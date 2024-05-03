"""
Evauates GPT2 models. This code specifically
checks the perplexity score of the model on the wikitext-103
dataset. 

In the original paper, the perplexity scores on WikiText-103:
GPT2-small (117M): 37.50
GPT2-medium (345M): 26.37
GPT2-large (762M): 22.05
GPT2-xl (1542M): 17.48

Again from paper, the perplexity scores on WikiText-2:
GPT2-small (117M): 29.41
GPT2-medium (345M): 22.76
GPT2-large (762M): 19.93
GPT2-xl (1542M): 18.34

Usage: python target_eval_gpt2.py
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import GPT2LMHeadModel

assert os.path.isfile("data/wikitext-103_val.bin"), "you must first run python prepro_wikitext-103.py [-p|--preprocess]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "small": "openai-community/gpt2",
    "medium": "openai-community/gpt2-medium",
    "large": "openai-community/gpt2-large",
    "xl": "openai-community/gpt2-xl"
}

print("Loading dataset...")
with open("data/wikitext-103_val.bin", "rb") as f:
    eval_text = np.frombuffer(f.read(), dtype = np.int32)
    eval_text = torch.tensor(eval_text, dtype = torch.long).unsqueeze(0)
print("Dataset loaded")

# evaluate each model size
for size in models:
    print(f"Loading gpt2-{size}...")
    model = GPT2LMHeadModel.from_pretrained(models[size]).to(device)
    print(f"gpt2-{size} loaded")

    # constants of perplexity score evaluation
    max_length = model.config.n_positions # context_length/block_size
    stride = max_length # stride_length (GPT2 was likely evaluated on stride_length = max_length)
    seq_len = eval_text.shape[1] # size of evaluation text

    nlls = [] # negative log likelihoods
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc = f"Evaluating gpt2-{size}"):
        end_loc = min(begin_loc + max_length, seq_len) # always use full context length unless fewer remaining tokens
        trg_len = end_loc - prev_end_loc # this is used for when stride_length < context_length
        input_ids = eval_text[:, begin_loc:end_loc].to(device) # pick the tokens to eval on and move to device
        target_ids = input_ids.clone() # clone them to targets
        target_ids[:, :-trg_len] = -100 # using a label of -100 tells pytorch not to calculate loss for the label

        with torch.no_grad():
            outputs = model(input_ids, labels = target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    # get mean of negative log likelihoods and exponentiate to get perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"gpt2-{size} perplexity score:", ppl.item())