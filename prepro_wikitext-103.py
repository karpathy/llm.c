"""
Downloads and tokenizes the WikiText-103 dataset.
- The download is from "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
 following https://github.com/tysam-code/hlb-gpt/tree/main
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Saved 241185 tokens to data/wikitext-103_val.bin
Saved 114933466 tokens to data/wikitext-103_train.bin

And runs in 3-4 minutes (~1.5 minutes to download data 
+ ~2 minutes to preprocess) depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import re
import requests
import zipfile
from tqdm import tqdm

import tiktoken
import numpy as np

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

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
    """Downloads the WikiText-103 dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the WikiText-103 dataset, unless it's already downloaded
    data_url = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
    data_filename = os.path.join(DATA_CACHE_DIR, "WikiText-103.zip")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unzip the file
    data_dir = os.path.join(DATA_CACHE_DIR, "wikitext-103")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unzipping {data_filename}...")
        with zipfile.ZipFile(data_filename, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print(f"{data_dir} already exists, skipping unzipping...")

def tokenize():
    # special token
    eot = enc._special_tokens["<|endoftext|>"]

    # fetch training text
    train_data_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103/wikitext-103-raw/wiki.train.raw")
    train_text = open(train_data_filename, "r", encoding = "utf-8").read()

    print("Cleaning training data (this should take about 1 minute)...")
    # cleanup the training text
    train_text = train_text.strip() # remove leading and trailing whitespace
    train_text = train_text.replace(" \n \n ", "\n<|endoftext|>") # injecting special token in between sections
    train_text = "<|endoftext|>" + train_text # adding special token at start
    train_split = train_text.split("<|endoftext|>") # splitting the text by special token to remove the extraneous headers/titles

    # remove the awkward headers/titles that came from the original parquet format
    for i in reversed(range(len(train_split))):
        # if the chunk is of the form of the headers/titles we will pop this chunk out
        if bool(re.match(r"^\s*= +(.{1,}) +=\s*$", train_split[i])):
            train_split.pop(i)
    
    # now join the remaining chunks via eot
    train_text = "<|endoftext|>".join(train_split[i] for i in range(len(train_split)))
    train_tokens = encode(train_text)
    train_tokens_np = np.array(train_tokens, dtype = np.int32)
    print("Training data cleaned")

    # now repeat same cleanup process but for validation text
    val_data_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103/wikitext-103-raw/wiki.valid.raw")
    val_text = open(val_data_filename, "r", encoding = "utf-8").read()

    print("Cleaning validation data...")
    val_text = val_text.strip() 
    val_text = val_text.replace(" \n \n ", "\n<|endoftext|>")
    val_text = "<|endoftext|>" + val_text
    val_split = val_text.split("<|endoftext|>")

    for i in reversed(range(len(val_split))):
        if bool(re.match(r"^\s*= +(.{1,}) +=\s*$", val_split[i])):
            val_split.pop(i)

    val_text = "<|endoftext|>".join(val_split[i] for i in range(len(val_split)))
    val_tokens = encode(val_text)
    val_tokens_np = np.array(val_tokens, dtype = np.int32)
    print("Validation data cleaned")

    # now just dump the encoded tokens into binary files
    train_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103_train.bin")
    val_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103_val.bin")
    with open(train_filename, "wb") as f:
        for chunk in tqdm([train_tokens_np[i : i + 1024] for i in range(0, len(train_tokens_np), 1024)], desc = "Writing train data to wikitext-103_train.bin", unit = "iB"):
            f.write(chunk.tobytes())

    with open(val_filename, "wb") as f:
        for chunk in tqdm([val_tokens_np[i : i + 1024] for i in range(0, len(val_tokens_np), 1024)], desc = "Writing validation data to wikitext-103_val.bin", unit = "iB"):
            f.write(chunk.tobytes())
    
    print(f"Saved {len(val_tokens_np)} tokens to {val_filename}")
    print(f"Saved {len(train_tokens_np)} tokens to {train_filename}")

if __name__ == "__main__":
    download()
    tokenize()
