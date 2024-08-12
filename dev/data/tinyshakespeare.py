"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

For GPT-2:
writing 32,768 tokens to tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to tinyshakespeare/tiny_shakespeare_train.bin

For LLaMA 3:
writing 32,768 tokens to tinyshakespeare/tiny_shakespeare_val.bin
writing 319,555 tokens to tinyshakespeare/tiny_shakespeare_train.bin

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

import argparse
import os

import tiktoken
from transformers import AutoTokenizer

from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # download the TinyShakespeare dataset, unless it's already downloaded
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def tokenize(model):
    if model == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
    elif model == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        def encode(x):
            return tokenizer(x).input_ids
    else:
        raise ValueError(f"unknown model {model}")

    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    # let's treat every person's statement in the dialog as a separate document
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    # encode the text
    tokens = encode(text)
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens, model)
    write_datafile(train_filename, train_tokens, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Shakespeare dataset preprocessing")
    parser.add_argument("-m", "--model", type=str, default="gpt-2", choices=["gpt-2", "llama"], help="Model type, gpt-2|llama")
    args = parser.parse_args()
    download()
    tokenize(args.model)