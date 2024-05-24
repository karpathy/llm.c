"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
# from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm
import argparse

from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

# create the cache directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "fineweb10B")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# todo is this needed? or just the load_dataset below?
# download 10B Tokens sample (~28GB on disk)
# folder = snapshot_download(
#     "HuggingFaceFW/fineweb",
#     repo_type="dataset",
#     local_dir="./data/fineweb/",
#     allow_patterns="sample/10BT/*"
# )
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

# helper functions
def tokenize(doc):
    # validate tokens in individual threads
    tokens = np.array([eot] + enc.encode_ordinary(doc["text"]))
    assert (0 <= tokens).all() and (tokens < 2**16).all(), "token dictionary too large for uint16"
    return tokens.astype(np.uint16)

# don't hog the entire system
nprocs = max(1, os.cpu_count() - 2)

# main loop write files
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # enough space to add this document fully?
        if token_count+len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")

            # split the last document
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None

            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
