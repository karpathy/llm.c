"""
1. Prepare the shakespeare dataset for character-level language modeling.

    .\data\tiny_shakespeare_train.bin
    .\data\tiny_shakespeare_val.bin

2. Save the tokenizer file:

    .\gpt2_tokenizer.bin

3. Create a new GPT-2 10M model from scratch and save it

    .\gpt2_124M.bin # This is actually a 10M model saved as with '124M' in the filename so that we don't modify 
                    # the train_gpt2.c/cu files. If/when the train_gpt2 can accept input params 
                    # you can save to what ever file name you like.

    e.g. Run this script from the root folder

    $python dev/prepare/prepro_tinyshakespeare_char.py

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

from prepro_tinyshakespeare import DATA_CACHE_DIR, download
from train_gpt2 import GPT, GPTConfig, write_model

def write_tokenizer(tokens, end_of_text, filename):
    n = len(tokens)
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 1 # tokenizer version = 1
    header[2] = n # number of tokens
    header[3] = end_of_text # # <|endoftext|>
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for token in tokens:
            b = token.encode('utf-8')
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")
    
def tokenize():
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    # text = "Hello World,\n\n\n\nPeace ✌..."

    # get all the unique characters that occur in this text
    unique = set(text)
    unique.add('\n')
    unique_chars = sorted(list(unique))

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(unique_chars) }
    itos = { i:ch for i,ch in enumerate(unique_chars) }

    # make a full list of tokens to save to the tokenizer file
    enc = []
    for i in range(len(unique_chars)):
        enc.append(itos[i])
    enc.append("<|endoftext|>")

    end_of_line = enc.index('\n')
    end_of_text = enc.index('<|endoftext|>')

    write_tokenizer(enc, end_of_text, "gpt2_tokenizer.bin")

    encoded_data = []

    # # let's treat every person's statement in the dialog as a separate document
    for i in range(len(text)):
        if i > 0 and text[i] == '\n' and text[i - 1] == '\n':
            # '\n\n' > '\n\n<|endoftext|>'
            encoded_data.append(stoi[text[i]])
            encoded_data.append(end_of_text)
        else:
            encoded_data.append(stoi[text[i]])
    encoded_data.append(end_of_text)

    encoded_data_np = np.array(encoded_data, dtype=np.int32)
    # create the train and test splits
    n = len(encoded_data_np)
    val_encoded_data_np = encoded_data_np[:int(n*0.9)]
    train_encoded_data_np = encoded_data_np[int(n*0.9):]

    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    with open(val_filename, "wb") as f:
        f.write(val_encoded_data_np.tobytes())
    with open(train_filename, "wb") as f:
        f.write(train_encoded_data_np.tobytes())

    # prints
    print(f"Saved {len(val_encoded_data_np)} tokens to {val_filename}")
    print(f"Saved {len(train_encoded_data_np)} tokens to {train_filename}")
    
    return enc, end_of_text

if __name__ == "__main__":
    import time
    import argparse
    import math
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")

    args = parser.parse_args()

    # download the data
    download()

    # train a tokenizer
    enc, end_of_text = tokenize()

    config_args = dict(n_layer=6, n_head=6, n_embd=384) # tiny-gpt
    config_args['vocab_size'] = len(enc)
    config_args['block_size'] = 256
    learning_rate = 1e-3 # start with a large lr
    max_iters = 5000
    lr_decay_iters = max_iters
    min_lr = 1e-4 # don't go below this lr. see train_gpt2.c
    warmup_iters = 10
    weight_decay = 0
    beta1 = 0.9
    beta2 = 0.999
    decay_lr = True

    B = 16
    T = 256
    assert 1 <= T <= config_args['block_size']

    # select a reasonable device to run on
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # seed the random number generators
    torch.manual_seed(37)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(37)

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # train a new model a from scratch
    config = GPTConfig(**config_args)
    model = GPT(config)
    model.to(device)
    
    model.train()

    n_params = model.get_num_params(False)
    print("number of parameters: %.2fM" % (n_params/1e6,))

    write_model(model, "gpt2_124M.bin") # it's actually a 10M model...

    def decode(l):
        return ''.join([enc[i] for i in l]) # decoder: take a list of integers, output a string

    def generate():
        # we'll kick off the generation with "<|endoftext|>"
        start_ids = [end_of_text]
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        max_new_tokens = 64
        temperature = 1
        top_k = 40
        model.eval()
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print('---------------')
        print(decode(y[0].tolist()))


    # let's also do one round of inference on an untrained random model
    generate()

    tokens_bin = "data/tiny_shakespeare_train.bin"
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.int32)

    # np -> tensor, long, on device
    tokens = torch.tensor(tokens)
    tokens = tokens.to(torch.long)
    tokens = tokens.to(device)

    def get_batch():
        assert B*T+1 <= len(tokens), "not enough tokens"
        # for 338,025 tokens. E.g. with B=8 T=1024, this will yield 41 batches before looping
        i = 0
        while True:
            x = tokens[i:i+B*T].view(B, T)
            y = tokens[i+1:i+B*T+1].view(B, T)
            yield x, y
            i += B*T
            if i + B*T + 1 >= len(tokens):
                i = 0 # in prod we'd want to randomize the start point a bit

    data_iter = iter(get_batch())

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # forward backward for a few iterations
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 betas=(beta1, beta2),
                                 eps=1e-8,
                                 weight_decay=weight_decay)

    timings = []
    for iter_num in range(max_iters):
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        t0 = time.time()
        x, y = next(data_iter)
        logits, loss = model(x, y)
        if not args.inference_only:
            # save the raw model initialized from scratch so it can be trained using the C counter part from scratch
            if iter_num == 0 and args.write_tensors:
                write_model(model, "gpt2_124M.bin")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        print(f"iteration {iter_num}, lr: {lr:e} loss: {loss.item()}, time: {(t1-t0)*1000:.3f}ms")
    if len(timings) > 0:
        print(f"final {args.num_iterations} iters avg: {np.mean(timings)*1000:.3f}ms")

    # let's see what we get now after pre-training
    generate()

    # save the pre-trained model...
    if not args.inference_only:
        write_model(model, "gpt2_124M.bin")
        
    # now continue to fine tune with train-gpt2.c with a smaller lr    