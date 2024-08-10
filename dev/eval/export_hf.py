"""
Script to convert GPT2 models from llm.c binary format to Hugging Face

It can optinally upload to your account on Hugging Face if you have the CLI:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login

Export to a local HF model:
  python export_hf.py --input input_file.bin --output output_dir

Export to a local HF model and also push to your account on Hugging Face:
  python export_hf.py --input input_file.bin --output output_dir --push true
"""

import numpy as np
import torch
import argparse, sys
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

def tensor_bf16(data_int16, transpose=False):
    if transpose:
        data_int16 = data_int16.transpose(1,0)
    return torch.tensor(data_int16).view(torch.bfloat16).to(torch.float32)

def tensor_fp32(data_float32, transpose=False):
    if transpose:
        data_float32 = data_float32.transpose(1,0)
    return torch.tensor(data_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

def convert(filepath, output, push_to_hub=False, out_dtype="bfloat16"):
    print(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if model_header[0] != 20240326:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    version = model_header[1]
    if not version in [3, 5]:
        print("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_header[2].item() # max sequence length
    V = model_header[3].item() # vocab size
    L =  model_header[4].item() # num layers
    H = model_header[5].item() # num heads
    C = model_header[6].item() # channels
    Vp = model_header[7].item() # padded vocab size

    print(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

    # Define the shapes of our parameters
    shapes = {
        'wte': (Vp, C),
        'wpe': (maxT, C),
        'ln1w': (L, C),
        'ln1b': (L, C),
        'qkvw': (L, 3 * C, C),
        'qkvb': (L, 3 * C),
        'attprojw': (L, C, C),
        'attprojb': (L, C),
        'ln2w': (L, C),
        'ln2b': (L, C),
        'fcw': (L, 4 * C, C),
        'fcb': (L, 4 * C),
        'fcprojw': (L, C, 4 * C),
        'fcprojb': (L, C),
        'lnfw': (C,),
        'lnfb': (C,),
    }

    # Load in our weights given our parameter shapes
    dtype = np.float32 if version == 3 else np.int16
    w = {}
    for key, shape in shapes.items():
        num_elements = np.prod(shape)
        data = np.frombuffer(f.read(num_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[key] = data.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        if shape[0] == Vp:
            w[key] = w[key].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    assert f.read() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[version]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    for i in range(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_size = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_head = H)
    model = GPT2LMHeadModel(config)
    if out_dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    # Set the model dict and save
    model.load_state_dict(model_dict)
    model.save_pretrained(output, max_shard_size="5GB", safe_serialization=True)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    if push_to_hub:
        print(f"Uploading {output} to Hugging Face")
        model.push_to_hub(output)
        tokenizer.push_to_hub(output)

def spin(output):
    print("Taking the exported model for a spin...")
    print('-'*80)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    tokens = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    tokens = tokens.to('cuda')
    output = model.generate(tokens, max_new_tokens=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    for sample in samples:
        print('-'*30)
        print(sample)

# -----------------------------------------------------------------------------

if __name__== '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=True)
    parser.add_argument("--output","-o",  help="The Hugging Face output model directory", type=str, required=True)
    parser.add_argument("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, default="bfloat16")
    parser.add_argument("--push", "-p", help="Push the model to your Hugging Face account", type=bool, default=False)
    parser.add_argument("--spin", "-s", help="Take the model for a spin at the end?", type=bool, default=True)
    args = parser.parse_args()
    convert(args.input, args.output, args.push, args.dtype)
    if args.spin:
        spin(args.output)
