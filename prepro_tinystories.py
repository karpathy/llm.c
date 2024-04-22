"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Tokenizing val split...
Saved 19043638 tokens to data/TinyStories_val.bin
Tokenizing train split...
Saved 925653391 tokens to data/TinyStories_train.bin

And runs in 1-2 minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import glob
import json
import random
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import tiktoken
import numpy as np

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

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
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    #print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename):
    with open(shard_filename, "r") as f:
        data = json.load(f)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize():
    # shard 0 will be the val split, rest is train
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        all_tokens_np = np.array(all_tokens, dtype=np.int32)
        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        with open(split_filename, "wb") as f:
            f.write(all_tokens_np.tobytes())
        print(f"Saved {len(all_tokens_np)} tokens to {split_filename}")

if __name__ == "__main__":
    download()
    tokenize()

    # Prints:
    # Tokenizing val split...
    # Saved 19043638 tokens to data/TinyStories_val.bin
    # Tokenizing train split...
    # Saved 925653391 tokens to data/TinyStories_train.bin
