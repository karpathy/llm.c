"""
Downloads and tokenizes the WikiText-103 validation split.
- The download is from "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
 following https://github.com/tysam-code/hlb-gpt/tree/main
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.

Usage: python prepro_wikitext-103.py [-p|--preprocess]
"""

import os
import re
import requests
import argparse
import zipfile
import numpy as np
from tqdm import tqdm

import tiktoken

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special = {"<|endoftext|>"})

def download_file(url : str, fname : str, chunk_size = 1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream = True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc = fname,
        total = total,
        unit = "iB",
        unit_scale = True,
        unit_divisor = 1024,
    ) as bar:
        for data in resp.iter_content(chunk_size = chunk_size):
            size = file.write(data)
            bar.update(size)

def download():
    """Downloads the WikiText-103 dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok = True)

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
        os.makedirs(data_dir, exist_ok = True)
        print(f"Unzipping {data_filename}...")
        with zipfile.ZipFile(data_filename, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print(f"{data_dir} already exists, skipping unzipping...")

def tokenize(preprocess : bool):
    # special token
    eot = enc._special_tokens["<|endoftext|>"]

    # fetch validation text
    val_data_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103/wikitext-103-raw/wiki.valid.raw")
    val_text = open(val_data_filename, "r", encoding = "utf-8").read()

    if preprocess:
        print("Cleaning validation data...")
        # cleanup the validation text
        val_text = val_text.strip() # remove leading and trailing whitespace
        val_text = val_text.replace(" \n \n ", "\n<|endoftext|>") # injecting special token in between sections
        # removing some low hanging fruit artifacts
        val_text = val_text.replace("@-@", "-")
        val_text = val_text.replace("@.@", ".")
        val_text = val_text.replace("@,@", ",")
        val_text = "<|endoftext|>" + val_text # adding special token at start
        val_split = val_text.split("<|endoftext|>") # splitting the text by special token to remove the extraneous headers/titles

        # remove the awkward headers/titles that came from the original parquet format
        for chunk in tqdm([item for item in reversed(range(len(val_split)))], desc = "Removing artifacts", unit = "iB"):
            # if the chunk is of the form of the headers/titles we will pop this chunk out
            if bool(re.match(r"^\s*= +(.{1,}) +=\s*$", val_split[chunk])):
                val_split.pop(chunk)

        # now join the remaining chunks via eot
        val_text = "<|endoftext|>".join(val_split[i] for i in range(len(val_split)))
        print("Validation data cleaned")

    print("Tokenizing validation text...")
    val_tokens = encode(val_text)
    print("Validation text tokenized")
    val_tokens_np = np.array(val_tokens, dtype = np.int32)

    print("Dumping text into text files to observe readable output")
    with open(os.path.join(DATA_CACHE_DIR, "wikitext-103_val.txt"), "w") as f:
        f.write(val_text)

    # now just dump the encoded tokens into binary files
    val_filename = os.path.join(DATA_CACHE_DIR, "wikitext-103_val.bin")

    with open(val_filename, "wb") as f:
        for chunk in tqdm([val_tokens_np[i : i + 1024] for i in range(0, len(val_tokens_np), 1024)], desc = "Writing validation data to wikitext-103_val.bin", unit = "iB"):
            f.write(chunk.tobytes())
    
    print(f"Saved {len(val_tokens_np)} tokens to {val_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action = "store_true", help = "Preprocesses the dataset")
    args = parser.parse_args()
    download()
    tokenize(args.preprocess)