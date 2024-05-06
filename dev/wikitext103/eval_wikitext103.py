"""
Evaluates GPT-2 on WikiText .bin files

Usage: python eval_wikitext103.py
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import GPT2LMHeadModel

BIN_FILENAME = "wikitext103_val.bin"

# detect a good device to use: cpu, or cuda, or mps (MacBook)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

models = {
    "124M": "gpt2",
    "350M": "gpt2-medium",
    "774M": "gpt2-large",
    "1558M": "gpt2-xl"
}

print("Loading dataset...")
with open(BIN_FILENAME, "rb") as f:
    tokens = np.frombuffer(f.read(), dtype=np.int32)
# note we'll move all tokens to device. the val of WikiText103 is only ~1MB
tokens = torch.tensor(tokens).long().unsqueeze(0).to(device)

# loop models and eval
for size in models:

    # load the model
    print(f"Loading gpt2-{size}...")
    model = GPT2LMHeadModel.from_pretrained(models[size]).to(device)

    # per Alec instructions on Reddit thread
    # https://www.reddit.com/r/MachineLearning/comments/oye64h/comment/h7ucco2/
    stride = 32
    # constants of perplexity score evaluation
    max_length = model.config.n_positions # context_length/block_size
    assert max_length == 1024 # being defensive for now, shouldn't be an issue
    seq_len = tokens.nelement()

    # DEBUG CROP, TODO REMOVE LATER
    seq_len = 8192

    # ref
    # https://huggingface.co/docs/transformers/perplexity
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = tokens[:, begin_loc:end_loc]
        # note: GPT2LMHeadModel internally shifts the targets
        # and the loss doesn't get evaluated on the very last time step
        # ¯\_(ツ)_/¯
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # get mean of negative log likelihoods and exponentiate to get perplexity
    nll = torch.stack(nlls).mean()
    ppl = torch.exp(nll)
    print(f"gpt2-{size} nll: {nll}, ppl: {ppl}")

