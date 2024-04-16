"""
    Run this script from the root folder

        $python dev/prepare/gpt2-from-scratch.py
"""

from ast import Import
import os
import requests
from tqdm import tqdm
import struct
import torch
import numpy as np

import sys
sys.path.append('.')

from train_gpt2 import GPT, GPTConfig, write_model, write_tokenizer

if __name__ == "__main__":
    import time
    import argparse
    import math
    import tiktoken

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # save the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    write_tokenizer(enc, "gpt2_tokenizer.bin")

    model_type = "gpt2"
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

    # train a new model a from scratch
    config = GPTConfig(**config_args)
    model = GPT(config)

    model.train() # I don't think we need this

    n_params = model.get_num_params(False)
    print("number of parameters: %.2fM" % (n_params/1e6,))

    # save the new model to be trained from scratch using train_gpt2.cu
    write_model(model, "gpt2_124M.bin")