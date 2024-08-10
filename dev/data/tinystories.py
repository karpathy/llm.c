"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is using either GPT-2 or LLaMA 3 tokenizer.

The output is written to a newly created tinystories/ folder.
The script prints:

For GPT-2:
Tokenizing val split...
Saved 19043638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
Saved 925653391 tokens to tinystories/TinyStories_train.bin

For LLaMA 3:
Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import glob
import json
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tiktoken
import numpy as np
from data_common import download_file, write_datafile
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename, model):
    if model == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    elif model == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        def encode(x):
            return tokenizer(x).input_ids
        eot = None
    else:
        raise ValueError(f"unknown model {model}")

    with open(shard_filename, "r") as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        if eot is not None:
            all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize(model):
    # shard 0 will be the val split, rest is train
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename, model)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens, model)

if __name__ == "__main__":
    model = "gpt-2"  # gpt-2 or llama
    download()
    tokenize(model)
