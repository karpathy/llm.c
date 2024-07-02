# llm.cpp

This repo is a fork of [llm.c](https://github.com/karpathy/llm.c), using C++(with the Eigen library)  to reproduce GPT-2.


## quick start

The best introduction to the llm.cpp repo today is reproducing the GPT-2 (124M) model.

## quick start (CPU)

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
mkdir build && cd build
cmake ..
make train_gpt2_cpp
cd ../
./build/train_gpt2_cpp
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C++ and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. The output looks like this on my LMDE3 (Intel© Core™ i7-10700K CPU @ 3.80GHz × 8):

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
train dataset num_batches: 1192
val dataset num_batches: 128
val loss 5.325521
step 0: train loss 5.356185 (took 2623.140371 ms)
step 1: train loss 4.300968 (took 2248.935874 ms)
step 2: train loss 4.623280 (took 2287.073546 ms)
step 3: train loss 4.600363 (took 2481.456694 ms)
... (trunctated) ...
step 39: train loss 3.970914 (took 2299.972425 ms)
val loss 4.016771
generating:
---
Come discourse unto the
Governor -
Be in charge of this
friendly gentleman: if
you cannot utter the word
at Newark,
I will not do hence.

<|endoftext|>DECODE II: GOOD VALIDIUS!
Your parents, your govern's wife,
silver eyes, no pret
---
step 40: train loss 4.378000 (took 2420.432457 ms)
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
make test_gpt2_cpp
cd ../
./build/test_gpt2_cpp
```

This now loads the `gpt2_124M_debug_state.bin` file that gets written by train_gpt2.py, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch.
This tests both the fp32 path and the mixed precision path. The test should pass and print `overall okay: 1`.


## license

MIT
