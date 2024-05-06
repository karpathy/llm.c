"""
Downloads and tokenizes the WikiText-103 validation split.
- The download is from "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
 following https://github.com/tysam-code/hlb-gpt/tree/main
 and is ~200MB file.
- The tokenization is GPT-2 tokenizer with tiktoken

Runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.

Usage:
python prepro_wikitext103.py
"""

import os
import re
import requests
import argparse
import zipfile
import numpy as np
from tqdm import tqdm

import tiktoken

# -------------------------------------
DATA_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
ZIP_FILENAME = "wikitext-103-raw-v1.zip"
DATA_DIR = "wikitext-103-raw"
VAL_FILENAME = "wikitext-103-raw/wiki.valid.raw"
BIN_FILENAME = "wikitext103_val.bin"
# -------------------------------------

def download(url : str, fname : str, chunk_size = 1024):
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

def unzip(fname : str):
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall()

def tokenize(fname : str, output_fname : str = None):
    """Tokenizes input text file, saves tokens .bin to disk"""
    val_text = open(fname, "r", encoding = "utf-8").read()
    print(f"Read {len(val_text)} characters from {fname}. Tokenizing...")
    enc = tiktoken.get_encoding("gpt2")
    val_tokens = enc.encode(val_text)
    print(f"Tokenized to {len(val_tokens)} tokens. Writing...")
    val_tokens_np = np.array(val_tokens, dtype = np.int32)
    with open(output_fname, "wb") as f:
        f.write(val_tokens_np.tobytes())
    print(f"Saved tokens to {output_fname}")

if __name__ == "__main__":

    # lazy download
    if not os.path.isfile(ZIP_FILENAME):
        print(f"Downloading {DATA_URL}...")
        download(DATA_URL, ZIP_FILENAME)
    else:
        print(f"{ZIP_FILENAME} already exists, skipping download...")

    # lazy unzip
    if not os.path.isdir(DATA_DIR):
        print(f"Unzipping {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
                zip_ref.extractall()
    else:
        print(f"{DATA_DIR} already exists, skipping unzipping...")

    # tokenize
    tokenize(VAL_FILENAME, BIN_FILENAME)
