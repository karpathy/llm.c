
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
                 model_name: str = "d6",
                 input_bin: str = "dev/data/fineweb10B/fineweb_train_*.bin",
                 input_val_bin: str = "dev/data/fineweb10B/fineweb_val_*.bin",
                 batch_size: int = 8,
                 total_batch_size: int = -1,
                 warmup_iters: int = 700,
                 learning_rate_decay_frac: float = 0.0,
                 grad_clip: float = 1.0,
                 val_max_steps: int = 200,
                 dtype: str = "float32",
                 zero_stage: int = 1,
                 multi_objective: bool = False,):
        # Initialize parameters
        self.config_space = config_space
        self.seed = seed
        self.budget = budget
        self.model_name = model_name
        self.input_bin = input_bin
        self.input_val_bin = input_val_bin
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.warmup_iters = warmup_iters
        self.learning_rate_decay_frac = learning_rate_decay_frac
        self.grad_clip = grad_clip
        self.val_max_steps = val_max_steps
        self.dtype = dtype
        self.zero_stage = zero_stage
        self.multi_objective = multi_objective
        
        # Initialize WandB
        self.run = self._init_wandb()

        # Setup logging, device, and random seeds
        self.logger = self._setup_logger()
        self.device, self.device_type = self._setup_device()
        self._set_seeds()
        
        # Initialize model and data loaders
        self.model = self._init_model()
        self.train_loader, self.val_loader = self._init_dataloaders()

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
            name=f"lr_{self.config_space['learning_rate']}_wd_{self.config_space['weight_decay']}_sl_{self.config_space['sequence_length']}_seed_{self.seed}_bud_{self.budget}"
        )

    def _setup_logger(self):
        # Customize this to set up your logging
        # logger = logging.getLogger(__name__)
        return None

    def _setup_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        return device, device_type

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _init_model(self):
        # Initialize the model from config
        model_name = self.model_name
        if model_name[0] == "d":
            model_config = {
                "d6": GPTConfig(block_size=1024, vocab_size=50257, n_layer=6, n_head=6, n_embd=384),
                "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
                "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
                "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
                "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
            }[model_name]
            model = GPT(model_config)
        else:
            model = GPT.from_pretrained(model_name)
        
        model.to(self.device)
        return model

    def _init_dataloaders(self):
        train_loader = DistributedDataLoader(self.input_bin, self.batch_size, 
                                             self.config_space['sequence_length'], ddp_rank=0, ddp_world_size=1)
        val_loader = DistributedDataLoader(self.input_val_bin, self.batch_size, 
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

    def _save_model(self, step, optimizer, save_name):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time_elapsed': time.time(),
            'step': step,
        }
        torch.save(ckpt, f"./dev/models/{save_name}.pth")

    def train(self):
        # Training loop logic
        optimizer = self._get_optimizer()
        num_iterations = self.train_loader.ntok_total // self.total_batch_size
        model = self.model
        step = 0

        while time.time() < self.budget:
            # Fetch a batch
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast(device_type=self.device_type, dtype=torch.float32):
                _, loss = model(x, y, return_logits=False)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                val_loss = self.validate()
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "step": step})
            
            if step % 2000 == 0:
                self._save_model(step, optimizer, save_name=f"{self.model_name}_step_{step}")
        
        wandb.finish()

    def validate(self):
        # Validation logic
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i in range(self.val_max_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y, return_logits=False)
                val_loss += loss.item()
        
        val_loss /= self.val_max_steps
        return val_loss
