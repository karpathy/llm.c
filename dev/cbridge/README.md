# cbridge

We'll use this directory for the PyTorch -> C bridge. So we have some PyTorch code and we'd like to have the equivalent C implementation (usually that one in turn serves as reference for the CUDA kernels later).

For starters we have RoPE. E.g. generate the reference with PyTorch and then match it in C:

```bash
python rope.py
gcc -o rope rope.c -lm
./rope
```

The .py file writes a `robe.bin` file with the intermediate tensors.
