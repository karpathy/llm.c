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
parser.add_argument("-s", "--shard_size", type=int, default=10**9, help="Size of each shard in tokens")
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
    return enc.encode_ordinary(doc["text"])

# main loop write files
with mp.Pool() as pool:
    shard_index = 0
    all_tokens = []
    progress_bar = None
    for tokens in pool.imap(tokenize, fw):

        # record the tokens and make sure to separate documents
        all_tokens.append(eot)
        all_tokens.extend(tokens)

        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))

        # if we reach shard_size tokens, write shard to disk
        if len(all_tokens) >= args.shard_size:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            write_tokens = all_tokens[:args.shard_size]
            rest_tokens = all_tokens[args.shard_size:]
            write_datafile(filename, write_tokens)
            shard_index += 1
            progress_bar = None
            # note: create a copy so Python can free the all_tokens memory above
            # the list rest_tokens is expected to be very small
            all_tokens = [t for t in rest_tokens]

    # write any remaining tokens as the last shard
    if len(all_tokens) > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens)
