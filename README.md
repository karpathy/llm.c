# llm.cpp
项目 fork 自 karpathy 的 [llm.c](https://github.com/karpathy/llm.c)，使用 C++(with Eigen) 来复现 GPT-2，支持 CPU/CUDA 计算。


This repo is forked from karpathy's [llm.c](https://github.com/karpathy/llm.c), using C++ (with Eigen) to reproduce GPT-2.


## quick start (CPU)

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
mkdir build && cd build
cmake ..
make train_gpt2_cpu
cd ../
./build/train_gpt2_cpu
```

The above lines 
- (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, 
tokenize it with the GPT-2 Tokenizer
- (2) download and save the GPT-2 (124M) weights
- (3) init from them in C++ and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. The output looks like this on my LMDE3 (Intel© Core™ i7-10700K CPU @ 3.80GHz × 8):

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904(474 MB)
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 82723584(315 MB)
val loss 5.325413
step 0: train loss 5.356086 (took 786.515755 ms)
step 1: train loss 4.300581 (took 677.340087 ms)
step 2: train loss 4.623053 (took 674.843167 ms)
step 3: train loss 4.599307 (took 673.189660 ms)
... (trunctated) ...
step 39: train loss 3.972404 (took 749.386021 ms)
val loss 4.017484
generating:
---
Requinetarius,
Which; supreme, but
Commands jest in vain for ever.

<|endoftext|>Lady:
No, heavens,
I were not to haste
To retire valorously and look nobly in the face,
Before this
UNHISILIUS UNDERDEINTS

---
step 40: train loss 4.378605 (took 692.830391 ms)
final 40 iters avg: 692.974 ms
```

## quick start (1 GPU, fp32 only)
```bash
mkdir build && cd build
cmake ..
make train_gpt2_gpu
cd ../
./build/train_gpt2_gpu
```


## datasets

The data files inside `/dev/data/(dataset).py` are responsible for downloading, tokenizing and saving the tokens to .bin files, readable easily from C. So for example when you run:

```bash
python dev/data/tinyshakespeare.py
```

We download and tokenize the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The output of this looks like this:

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```

The .bin files contain a short header (1024 bytes) and then a stream of tokens in uint16, indicating the token ids with the GPT-2 tokenizer. More datasets are available in `/dev/data`.

## test

I am also attaching a simple unit test for making sure our C++ code agrees with the PyTorch code. On the CPU as an example, compile and run with:

```bash
mkdir build && cd build
cmake ..
make test_gpt2_cpu
cd ../
./build/test_gpt2_cpu
```

This now loads the `gpt2_124M_debug_state.bin` file that gets written by train_gpt2.py, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch.
This tests both the fp32 path and the mixed precision path. The test should pass and print `overall okay: 1`.


## license

MIT
