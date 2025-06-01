import torch
import numpy as np
from train_gpt2 import GPTConfig, GPT
from transformers import GPT2LMHeadModel, GPT2Config

def read_bf16(tensor_size, file):
    bytes_to_read = np.prod(tensor_size) * 2  # 2 bytes per bfloat16
    raw_bytes = file.read(bytes_to_read)
    n = np.frombuffer(raw_bytes, dtype=np.int16).reshape(
        tensor_size)  # bfloat16 is stored as int16 for numpy
    t = torch.from_numpy(n).view(torch.bfloat16)  # convert to bfloat16
    return t


def read_fp32(tensor_size, file):
    bytes_to_read = np.prod(tensor_size) * 4  # 4 bytes per float32
    raw_bytes = file.read(bytes_to_read)
    n = np.frombuffer(raw_bytes, dtype=np.float32).reshape(
        tensor_size)
    t = torch.from_numpy(n).view(torch.float32)  # convert to torch
    return t


def read_tensors(state_dict, model_tensors, L, file, dtype):
    # Reads the GPT-2 model's weights from a binary file
    assert dtype in {"float32", "bfloat16"}
    read_fun = read_fp32 if dtype == "float32" else read_bf16
    state_dict["transformer.wte.weight"] = read_fun(
        model_tensors["transformer.wte.weight"], file)  # (V, C)
    state_dict["transformer.wpe.weight"] = read_fun(
        model_tensors["transformer.wpe.weight"], file)  # (T, C)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.ln_1.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.ln_1.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L):  # (L, 3C, C)
        state_dict[f"transformer.h.{i}.attn.c_attn.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L):  # (L, 3C)
        state_dict[f"transformer.h.{i}.attn.c_attn.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L):  # (L, C, C)
        state_dict[f"transformer.h.{i}.attn.c_proj.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.attn.c_proj.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.ln_2.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.ln_2.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L):  # (L, 4C, C)
        state_dict[f"transformer.h.{i}.mlp.c_fc.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L):  # (L, 4C)
        state_dict[f"transformer.h.{i}.mlp.c_fc.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L):  # (L, C, 4C)
        state_dict[f"transformer.h.{i}.mlp.c_proj.weight"] = read_fun(
            model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L):  # (L, C)
        state_dict[f"transformer.h.{i}.mlp.c_proj.bias"] = read_fun(
            model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    state_dict["transformer.ln_f.weight"] = read_fun(
        model_tensors["transformer.ln_f.weight"], file)  # (C, )
    state_dict["transformer.ln_f.bias"] = read_fun(
        model_tensors["transformer.ln_f.bias"], file)  # (C, )

    return state_dict


def get_state(filename,return_config=False):
    # Reads the GPT-2 model's weights from a binary file
    file = open(filename, 'rb')

    header_bytes = file.read(256*4)  # 256 int32 values
    model_header = np.frombuffer(header_bytes, dtype=np.int32)

    assert model_header[0] == 20240326  # magic number
    version = model_header[1]
    config = GPTConfig(block_size=model_header[2].item(), # convert to native python int. Otherwise it causes problems when used to initiate HF Model.
                       vocab_size=model_header[3].item(),
                       n_layer=model_header[4].item(),
                       n_head=model_header[5].item(),
                       n_embd=model_header[6].item(),
                       )
    padded_vocab_size = model_header[7]

    # Instantiate a dummy model to get the tensor shapes and state_dict keys
    model = GPT(config)

    dtype = {
        3: "float32",  # 3: all tensors are fp32, padded vocab
        5: "bfloat16",  # 5: all tensors are bf16, padded vocab
    }[version]

    if dtype == 'bfloat16':
        model = model.bfloat16()

    tensor_shapes = {name: list(param.shape)
                     for name, param in model.named_parameters()}

    # pad the vocab to padded_vocab_size
    wte_padded = [padded_vocab_size, config.n_embd]  # (Vp, C)
    tensor_shapes["transformer.wte.weight"] = wte_padded  # (Vp, C)

    state_dict = model.state_dict()
    state_dict = read_tensors(
        state_dict, tensor_shapes, config.n_layer, file, dtype)

    # Ensure extra padding is zero
    assert state_dict['transformer.wte.weight'][model_header[3]:].sum() == 0
    # Remove extra padding
    state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight'][:model_header[3]]

    # copy lm_head.weights from transformer.wte.weight
    state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']

    assert file.read() == b''  # Ensure the file is fully read
    file.close()

    return (state_dict, config) if return_config else state_dict

def get_hf_model(filename):
    # Reads the GPT-2 model's weights from a binary file
    sd,config = get_state(filename, return_config=True)
    
    # Construct a HF model with the same config
    hf_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    model_hf = GPT2LMHeadModel(hf_config)

    sd_hf = model_hf.state_dict()

    # Processing the parameters which need to be ignored and those that need to be transposed (c.f train_gpt2.py)
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    # parameters that need to be transposed
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k]=sd[k].t()
    
    # Load the state dict into the HF model
    m = model_hf.load_state_dict(sd,strict=False)
    
    assert len(m.missing_keys)==0
    
    for k in m.unexpected_keys: # these are OK, as we ignored them earlier
        assert k.endswith('.attn.masked_bias') or k.endswith('.attn.bias')
    
    return model_hf