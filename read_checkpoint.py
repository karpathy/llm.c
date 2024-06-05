import torch
import numpy as np
from train_gpt2 import GPTConfig, GPT

def read_bf16(tensor_size, file):
    bytes_to_read = np.prod(tensor_size) * 2  # 2 bytes per bfloat16
    raw_bytes = file.read(bytes_to_read)
    n = np.frombuffer(raw_bytes, dtype=np.int16).reshape(
        tensor_size)  # bfloat16 is stored as int16 for numpy
    t = torch.from_numpy(n).view(torch.bfloat16) # convert to bfloat16
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


def get_state(filename, dtype='float32'):
    assert dtype in {"float32", "bfloat16"}
    file = open(filename, 'rb')

    header_bytes = file.read(256*4)  # 256 int32 values
    model_header = np.frombuffer(header_bytes, dtype=np.int32)

    config = GPTConfig(block_size=model_header[2],
                       vocab_size=model_header[3],
                       n_layer=model_header[4],
                       n_head=model_header[5],
                       n_embd=model_header[6],
                       )
    padded_vocab_size = model_header[7]
    
    # Instantiate a dummy model to get the tensor shapes and state_dict keys
    model = GPT(config)
    
    if dtype == 'bfloat16':
        model = model.bfloat16()

    tensor_shapes = {name: list(param.shape) for name, param in model.named_parameters()}
    
    # pad the vocab to padded_vocab_size 
    wte_padded = [padded_vocab_size, config.n_embd]  # (Vp, C)
    tensor_shapes["transformer.wte.weight"] = wte_padded  # (Vp, C)

    state_dict = model.state_dict() 
    state_dict = read_tensors(state_dict, tensor_shapes, config.n_layer, file, dtype)
        
    # Ensure extra padding is zero
    assert state_dict['transformer.wte.weight'][model_header[3]:].sum() == 0
    # Remove extra padding
    state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight'][:model_header[3]]
    
    # copy lm_head.weights from transformer.wte.weight
    state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']
    
    assert file.read() == b''  # Ensure the file is fully read
    file.close()

    return state_dict
