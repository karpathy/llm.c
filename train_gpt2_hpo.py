
import time
import argparse
import tiktoken
import torch
import os 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from train_gpt2 import GPT, GPTConfig, DistributedDataLoader, write_model
from contextlib import nullcontext
import torch._inductor.config as config
import math
import torch.distributed as dist
from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
import pickle


def train_eval_hpo(
    config_space: ConfigurationSpace, 
    seed: int = 0,
    budget: int = 10,
    input_bin: str = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
    input_val_bin: str = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin",
    model: str = "d6",
    batch_size: int = 4,
    # sequence_length: int = 1024,
    total_batch_size: int = -1,
    # learning_rate: float = 1e-4,
    warmup_iters: int = 0,
    learning_rate_decay_frac: float = 1.0,
    # weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    val_max_steps: int = 20,
    dtype: str = "float32",
    zero_stage: int = 0,
    logger = None,
    multi_objective: bool = False,
):

    if os.path.exists("hp_details.pkl"):
        with open("hp_details.pkl", "rb") as f:
            hp_details = pickle.load(f)
    else:
        hp_details = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    section_tab = 3*"\t"
    print(f"Running train_eval_hpo with budget: {budget}, config_space: {config_space.get_dictionary()}, seed: {seed}, budget: {budget}")
    
    step = None
    time_elapsed = None
    learning_rate = config_space["learning_rate"]
    weight_decay = config_space["weight_decay"]
    sequence_length = config_space["sequence_length"]
    
    save_name = f"{model}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_seq_{sequence_length}_budget_{budget}"
       
    logger.info(f"{section_tab}== Running train_eval_hpo ==")
    # log all the arguments
    logger.info(f"{section_tab}== Arguments:\n"
                + section_tab + f"\tseed: {seed} \n"
                + section_tab + f"\tbudget: {budget} \n"
                + section_tab + f"\tinput_bin: {input_bin} \n"
                + section_tab + f"\tinput_val_bin: {input_val_bin} \n"
                + section_tab + f"\tmodel: {model} \n"
                + section_tab + f"\tbatch_size: {batch_size} \n"
                + section_tab + f"\ttotal_batch_size: {total_batch_size} \n"
                + section_tab + f"\twarmup_iters: {warmup_iters} \n"
                + section_tab + f"\tlearning_rate_decay_frac: {learning_rate_decay_frac} \n"
                + section_tab + f"\tgrad_clip: {grad_clip} \n"
                + section_tab + f"\tval_max_steps: {val_max_steps} \n"
                + section_tab + f"\tdtype: {dtype} \n"
                + section_tab + f"\tzero_stage: {zero_stage} \n")
        
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
    np.random.seed(seed)
            
    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration

    # args error checking and convenience variables
    B, T = batch_size, sequence_length
    assert 1 <= T <= 1024
    assert dtype in {"float32", "float16", "bfloat16"}
    assert model in {"gpt2-tiny", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d6", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # attempt to autodetect the device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    total_batch_size = total_batch_size if total_batch_size > 0 else tokens_per_fwdbwd
    
    logger.info(f"{section_tab}== Running setup:\n" 
                + section_tab + f"\tConfig_space:{config_space.get_dictionary()} \n"
                + section_tab + f"\tSeed: {seed} \n"
                + section_tab + f"\tBudget: {budget} \n"
                + section_tab + f"\ttokens_per_fwdbwd: {tokens_per_fwdbwd} \n"
                + section_tab + f"\ttotal_batch_size: {total_batch_size} \n"
                + section_tab + f"\tbatch_size: {B} \n"
                + section_tab + f"\tsequence_length: {T} \n")
                
    

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    model_name = model
    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d6":  GPTConfig(block_size=1024, vocab_size=50257, n_layer=6, n_head=6, n_embd=384),
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[model]
        model = GPT(model_config)
    else:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(model)
    model.train()
    model.to(device)

    # logger.info(f"{section_tab}== Model and device setup:\n"
    #             + section_tab + f"\tModel: {model} \n"
    #             + section_tab + f"\tDevice: {device} \n")

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if input_val_bin:
        val_loader = DistributedDataLoader(input_val_bin, B, 1024, ddp_rank, ddp_world_size)

    assert total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd
    
    num_iterations = train_loader.ntok_total // total_batch_size
    num_iterations = int(num_iterations)

    logger.info(f"{section_tab}== Data setup:\n"
                + section_tab + f"\ttrain_loader.ntok_total: {train_loader.ntok_total} \n"
                + section_tab + f"\tval_loader.ntok_total: {val_loader.ntok_total} \n"
                + section_tab + f"\tnum_iterations: {num_iterations} \n"
                + section_tab + f"\tval_max_steps: {val_max_steps} \n")

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay,
                                                learning_rate=learning_rate, betas=(0.9, 0.95),
                                                device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = learning_rate * learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it+1) / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (learning_rate - min_lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    section_tab = 5*"\t"
    logger.info(f"{section_tab}== Training loop ==\n")
    print(f"== Training loop ==")
    
    model_to_size = {"gpt2-tiny": "30M", "gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
    model_to_size.update({f"d{d}": f"d{d}" for d in [6, 12, 24, 36, 48]})
    
    print(f"saving model to: {save_name}")
    print(f"path does exist: {os.path.exists(f'./dev/models/{save_name}.pth')}")
    if os.path.exists(f"./dev/models/{save_name}.pth"):
        print(f'loading previous training states from: {save_name}')
        ckpt = torch.load(f"./dev/models/{save_name}.pth", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        step = ckpt['step']
        time_elapsed = ckpt['time_elapsed'] 
    
    
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    start_time = time.time()
    training_losses = []
    step = step if step is not None else 0
    time_elapsed = time_elapsed if time_elapsed is not None else 0.0    
    while time.time() + time_elapsed - start_time < budget:
        t0 = time.time()
        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            loss.backward()
            
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
    
        training_losses.append(lossf)
        if len(training_losses) > val_max_steps:
            training_losses.pop(0)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        
        if step % 200 == 0:
            logger.info(f"{section_tab} \tStep: {step}/{num_iterations} | Loss: {lossf} | LR: {get_lr(step)} | Norm: {norm}")
            print(f"Step: {step}/{num_iterations} | Loss: {lossf} | LR: {get_lr(step)} | Norm: {norm}")
        
        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()           
        # keep track of smooth timings, last 20 iterations
        timings.append(t1-t0)
        step += 1
        
        if step % 1000 == 0:
            # store scheduler and optimizer state and model weights
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'time_elapsed': time.time() - start_time,
                'step': step,
            }
            torch.save(ckpt, f"./dev/models/{save_name}.pth")
                    
    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    
    print(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    train_time = time.time() - start_time
    
    logger.info(f"{section_tab}== Training complete ==\n")
    
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss = 0.0
        for i in range(val_max_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y, return_logits=False)
            val_loss += loss.item()
            if i % 50 == 0:
                print(f"Validation step: {i}/{val_max_steps} | Loss: {val_loss / (i+1)}")
        val_loss /= val_max_steps

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
        
    logger.info(f"{section_tab}== Evaluation complete ==\n")
    logger.info(f"{section_tab}== Results ==\n"
                + section_tab + f"\tval_loss: {val_loss} \n"
                + section_tab + f"\ttrain_time: {train_time} \n")
    print(f"val_loss: {val_loss} \n train_time: {train_time}")
    train_loss = np.mean(training_losses)
    
    hp_details.append({"val_loss": val_loss, "train_loss": train_loss, "train_time": train_time, "model_size": model_to_size[model_name],
                        "batch_size": B, "learning_rate": learning_rate, "weight_decay": weight_decay, "sequence_length": T,
                       })    
    
    with open(f"hp_details.pkl", "wb") as f:
        pickle.dump(hp_details, f)
    
    # print(f"hp_details: {hp_details}")
    
    if multi_objective:
        return {"val_loss": val_loss, "train_time": train_time}
    else:
        return val_loss
    # return val_loss


