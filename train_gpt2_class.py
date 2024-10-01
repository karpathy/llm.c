
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
import logging
import datetime
import wandb


def setup_logger(name=None):

    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # File handler
    file_handler = logging.FileHandler(f'logs/smac_{formatted_date}.log')

    global log_file_name
    log_file_name = f'logs/smac_{formatted_date}'
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s',  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
    logger.propagate = False
    logging.captureWarnings(True)

def print_with_tabs(obj, num_tabs=1):
    # Convert the object to a string representation
    obj_str = str(obj)

    # Split the string representation into lines
    lines = obj_str.split('\n')

    # first line should not be tabbed
    tabbed_lines = ["\t"*num_tabs +lines[0] + "\n"]

    # Add a tab at the beginning of each line
    tabbed_lines = tabbed_lines + ['\t'*(num_tabs+1) + line + "\n" for line in lines[1:]]

    # Join the tabbed lines back into a single string and print
    return "".join(tabbed_lines).rstrip("\n")


# setup_logger('train_eval_hpo')
logger = logging.getLogger('HPO_gpt2')

import torch
import wandb
import time
import os
import numpy as np
import pickle
from contextlib import nullcontext

class Trainer:
    def __init__(self, config_space, seed, budget, 
                 input_bin: str = "dev/data/fineweb10B/fineweb_train_*.bin",
                 input_val_bin: str = "dev/data/fineweb10B/fineweb_val_*.bin",
                 batch_size: int = 8,
                 total_batch_size: int = -1,
                 warmup_iters: int = 700,
                 learning_rate_decay_frac: float = 0.0,
                 grad_clip: float = 1.0,
                 val_max_steps: int = 500,
                 dtype: str = "float32",
                 zero_stage: int = 1,
                 multi_objective: bool = False,):
        # Initialize parameters
        self.config_space = config_space
        self.seed = seed
        self.budget = budget
        self.model_name = config_space['model']
        self.input_bin = input_bin
        self.input_val_bin = input_val_bin
        self.batch_size = config_space["batch_size"] if "batch_size" in config_space else batch_size
        self.warmup_iters = warmup_iters
        self.learning_rate_decay_frac = learning_rate_decay_frac
        self.grad_clip = grad_clip
        self.val_max_steps = val_max_steps
        self.dtype = dtype
        self.zero_stage = zero_stage
        self.multi_objective = multi_objective
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.save_name = str(self.model_name)[:2] \
             + f"lr_{self.config_space['learning_rate']}_wd_{self.config_space['weight_decay']}_" \
             + f"sl_{self.config_space['sequence_length']}_bs_{self.batch_size}_h_{self.config_space['n_head']}_" \
             + f"l_{self.config_space['n_layer']}_em_{self.config_space['n_embd']}_seed_{self.seed}_bud_{self.budget}"
        self.step = 0
        self.time_elapsed = 0.0
        # Initialize WandB

        # Setup logging, device, and random seeds
        setup_logger()
        self.logger = logger
        self._setup_ddp()
        self._set_seeds()
        
        # Initialize model and data loaders
        self._init_model()
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        self.tokens_per_fwdbwd = self.config_space['sequence_length'] * self.batch_size * self.ddp_world_size
        self.total_batch_size = total_batch_size if total_batch_size > 0 else self.tokens_per_fwdbwd
        assert self.total_batch_size % self.tokens_per_fwdbwd == 0, "total_batch_size must be a multiple of tokens_per_fwdbwd"
        self.num_iterations = self.train_loader.ntok_total // self.total_batch_size
        self.run = self._init_wandb()
        
        if os.path.exists("hp_details_w_embeddings.pkl"):
            with open("hp_details_w_embeddings.pkl", "rb") as f:
                self.hp_details = pickle.load(f)
        else:
            self.hp_details = []

    def _log_setup(self):
        section_tab = 3*"\t"
        print(f"Running train_eval_hpo with budget: {self.budget}, config_space: {self.config_space.get_dictionary()}, seed: {self.seed}")
        logger.info(f"{section_tab}== Running train_eval_hpo ==")
        # log all the arguments
        logger.info(f"{section_tab}== Arguments:\n"
                    + section_tab + f"\tseed: {self.seed} \n"
                    + section_tab + f"\tbudget: {self.budget} \n"
                    + section_tab + f"\tinput_bin: {self.input_bin} \n"
                    + section_tab + f"\tinput_val_bin: {self.input_val_bin} \n"
                    + section_tab + f"\tmodel: {self.model_name} \n"
                    + section_tab + f"\tbatch_size: {self.batch_size} \n"
                    + section_tab + f"\ttotal_batch_size: {self.total_batch_size} \n"
                    + section_tab + f"\twarmup_iters: {self.warmup_iters} \n"
                    + section_tab + f"\tlearning_rate_decay_frac: {self.learning_rate_decay_frac} \n"
                    + section_tab + f"\tgrad_clip: {self.grad_clip} \n"
                    + section_tab + f"\tval_max_steps: {self.val_max_steps} \n"
                    + section_tab + f"\tdtype: {self.dtype} \n"
                    + section_tab + f"\tzero_stage: {self.zero_stage} \n")
        
        logger.info(f"{section_tab}== configuration space:\n" 
            + section_tab + f"\tConfig_space:{self.config_space.get_dictionary()} \n")
        
        logger.info(f"{section_tab}== Dataloaders setup:\n"
            + section_tab + f"\ttrain_loader.ntok_total: {self.train_loader.ntok_total} \n"
            + section_tab + f"\tval_loader.ntok_total: {self.val_loader.ntok_total} \n"
            + section_tab + f"\tnum_iterations: {self.num_iterations} \n"
            + section_tab + f"\tval_max_steps: {self.val_max_steps} \n")
        
    def _init_wandb(self):
        return wandb.init(
            project="LLMs",
            entity="o-swelam",
            config={
                "config_space": self.config_space.get_dictionary(),
                "seed": self.seed,
                "budget": self.budget,
                "input_bin": self.input_bin,
                "input_val_bin": self.input_val_bin,
                "model": self.model_name,
                "batch_size": self.batch_size,
                "total_batch_size": self.total_batch_size,
                "warmup_iters": self.warmup_iters,
                "learning_rate_decay_frac": self.learning_rate_decay_frac,
                "grad_clip": self.grad_clip,
                "val_max_steps": self.val_max_steps,
                "dtype": self.dtype,
                "zero_stage": self.zero_stage,
                "multi_objective": self.multi_objective
            },
            name= self.save_name,
        )

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _load_model(self):
        if os.path.exists(f"./dev/models/{self.save_name}.pth"):
            print(f'loading previous training states from: {self.save_name}')
            ckpt = torch.load(f"./dev/models/{self.save_name}.pth", map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.step = ckpt['step']
            self.time_elapsed = ckpt['time_elapsed'] 

    def _init_model(self):
        # Initialize the model from config
        model_name = self.model_name
        if model_name[0] == "d" or model_name == "custom":
            model_config = {
                "custom": GPTConfig(block_size=1024, vocab_size=50257, n_layer=self.config_space['n_layer'],
                                    n_head=self.config_space['n_head'], n_embd=self.config_space['n_embd']),
                "d6": GPTConfig(block_size=1024, vocab_size=50257, n_layer=6, n_head=6, n_embd=384),
                "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
                "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
                "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
                "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
            }[model_name]
            self.model = GPT(model_config)
        else:
            self.model = GPT.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        self.optimizer = self._get_optimizer()
        
        self._load_model()
        
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        
    def _setup_ddp(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.ddp:
            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = 0 # each process gets the exact same seed
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.zero_stage = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.seed_offset = 0
            # attempt to autodetect the device
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

    def _init_dataloaders(self):
        train_loader = DistributedDataLoader(self.input_bin, self.batch_size, 
                                             self.config_space['sequence_length'], ddp_rank=self.ddp_rank, ddp_world_size=self.ddp_world_size)
        val_loader = DistributedDataLoader(self.input_val_bin, 1, 
                                           1024, ddp_rank=0, ddp_world_size=1)
        return train_loader, val_loader

    def _get_optimizer(self):
        return self.model.configure_optimizers(weight_decay=self.config_space['weight_decay'],
                                               learning_rate=self.config_space['learning_rate'], betas=(0.9, 0.95),
                                               device_type=self.device, zero_stage=self.zero_stage)

    def _get_lr(self, step, num_iterations):
        min_lr = self.config_space["learning_rate"] * self.learning_rate_decay_frac
        if step < self.warmup_iters:
            return self.config_space["learning_rate"] * (step+1) / self.warmup_iters
        if step > num_iterations:
            return min_lr
        decay_ratio = (step - self.warmup_iters) / (num_iterations - self.warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_lr + coeff * (self.config_space["learning_rate"] - min_lr)

    def _save_model(self, step, start_time, optimizer):
        ckpt = {
            'model_state_dict': self.model.state_dict() if not self.ddp else self.model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time_elapsed': self.time_elapsed + time.time() - start_time,
            'step': step,
        }
        torch.save(ckpt, f"./dev/models/{self.save_name}.pth")

    def train(self):
        section_tab = 5*"\t"
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        ctx = torch.amp.autocast(device_type=self.device, dtype=self.ptdtype) if self.device == "cuda" else nullcontext()
        num_iterations = self.train_loader.ntok_total // self.total_batch_size
        grad_accum_steps = self.total_batch_size // self.tokens_per_fwdbwd
        
        self.validate(num_steps=1) # to make sure the memory works well under such conditions at max seq length
        timings = [0] * 200
        training_losses = [0] * 200
        start_time = time.time()
        while time.time() + self.time_elapsed - start_time < self.budget:
            self.step += 1
            t0 = time.time()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
            for micro_step in range(grad_accum_steps):
                # fetch a batch
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                if self.ddp:
                    # we want only the last micro-step to sync grads in a DDP model
                    # the official way to do this is with model.no_sync(), but that is a
                    # context manager that bloats the code, so we just toggle this variable
                    self.model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                # forward pass
                with ctx:
                    _, loss = self.model(x, y, return_logits=False)
                    # we have to scale the loss to account for gradient accumulation,
                    # because the gradients just add on each successive backward().
                    # addition of gradients corresponds to a SUM in the objective, but
                    # instead of a SUM we want MEAN, so we scale the loss here
                    loss = loss / grad_accum_steps
                    lossf += loss.detach() # keep track of the mean loss
                # backward pass
                loss.backward()

            if self.ddp:
                dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
            lossf = lossf.item()

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            lr = self._get_lr(self.step, self.num_iterations)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.step()
            
            # wait on the CPU for all device work to end so we get accurate per-iteration timings below
            if self.device == "mps":
                torch.mps.synchronize()
            elif self.device == "cuda":
                torch.cuda.synchronize()
            # time and print
            t1 = time.time()           

            training_losses.append(lossf)
            training_losses.pop(0)
            timings.append(t1-t0)
            timings.pop(0)

            if self.step % 200 == 0:
                logger.info(f"{section_tab} \tStep: {self.step}/{num_iterations} | Loss: {lossf} | LR: {lr} | Norm: {norm}")
                print(f"Step: {self.step}/{num_iterations} | Loss: {lossf} | LR: {lr} | Norm: {norm}")
                wandb.log({"avg train loss": np.mean(training_losses), "lr": lr, "step":self.step, "remaining_time": self.budget - (self.time_elapsed + time.time() - start_time)})
            
            if self.step % 2000 == 0:
                val_loss = self.validate()
                wandb.log({"val_loss": val_loss, "step": self.step})
                self._save_model(self.step, start_time, self.optimizer)
        
        logger.info(f"{section_tab}== Training complete ==\n")
        
        val_loss = self.validate()
        wandb.log({"val_loss": val_loss, "step": self.step})
        
        self.hp_details.append({
            "budget": self.budget,
            "val_loss": val_loss, 
            "train_loss": np.mean(training_losses), "train_time": self.time_elapsed + time.time() - start_time, 
            "batch_size": self.batch_size, 
            "learning_rate": self.config_space["learning_rate"],
            "weight_decay": self.config_space["weight_decay"], 
            "sequence_length": self.config_space["sequence_length"],
            "n_head": self.config_space["n_head"],
            "n_layer": self.config_space["n_layer"],
            "n_embd": self.config_space["n_embd"],
                       })
        
        with open(f"hp_details_w_embeddings.pkl", "wb") as f:
            pickle.dump(self.hp_details, f)
        wandb.finish()
        
        if self.ddp:
            destroy_process_group()
            
        return val_loss

    def validate(self, num_steps=None):
        # Validation logic
        val_loss = 0.0
        self.model.eval()
        num_steps = num_steps if num_steps is not None else self.val_max_steps
        with torch.no_grad():
            for i in range(num_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y, return_logits=False)
                val_loss += loss.item()
                if (i+1) % 20 == 0:
                    print(f"Validation step: {i}/{num_steps} | Loss: {val_loss / (i+1)}") 
        
        val_loss /= self.val_max_steps
        return val_loss
