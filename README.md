# llm.c

LLM training in simple, pure C/CUDA. There is no need for 245MB of PyTorch or 107MB of cPython. For example, training GPT-2 (CPU, fp32) is ~1,000 lines of clean code in a single file. It compiles and runs instantly, and exactly matches the PyTorch reference implementation. I chose GPT-2 as the first working example because it is the grand-daddy of LLMs, the first time the modern stack was put together.

Currently, I am working on:

- direct CUDA implementation, which will be significantly faster and probably come close to PyTorch.
- speed up the CPU version with SIMD instructions, AVX2 on x86 / NEON on ARM (e.g. Apple Silicon).
- more modern architectures, e.g. Llama2, Gemma, etc.

I'd like this repo to only maintain C and CUDA code. Ports of this repo to other programming languages are very welcome, but should be done in separate repos, and I am very happy to link to them below in the "notable forks" section, just like I did in [llama2.c notable forks](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#notable-forks).

## quick start

Download and tokenize a dataset. The [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset is the fastest to download and tokenize:

```bash
python prepro_tinyshakespeare.py
```

This prints:

```
Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespeare_train.bin
```

The .bin files are raw byte streams of int32 numbers indicating the token ids with the GPT-2 tokenizer. Alternatively you could also tokenize the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset with `prepro_tinystories.py`.

In principle we'd be ready to train the model right here. However the baseline CPU/fp32 reference code is so inefficient that it's not practical to train these models from scratch yet. Instead, we initialize with the GPT-2 weights released by OpenAI and just do finetuning. For that, we have to download the GPT-2 weights and save them as a checkpoint we can load in C:

```bash
python train_gpt2.py
```

You'll recognize this code from nanoGPT as a simple GPT-2 reference implementation in PyTorch. This script will download the GPT-2 (124M) model, overfit a single batch of data for 10 iterations, run a few steps of generation, and most importantly it will save three files: 1) the `gpt2_124M.bin` file that contains the raw model weights for loading in C, 2) the `gpt2_124M_debug_state.bin`, which also contains more debug state: the inputs, targets, logits and loss (useful for debugging and unit testing), and finally 3) the `gpt2_tokenizer.bin` which stores the vocabulary for the GPT-2 tokenizer, translating token ids to byte sequences of UTF-8 encoded string pieces. We can now initialize with these model weights and continue training in raw C. First compile the code:

```bash
make train_gpt2
```

You can have a look inside the `Makefile` and its comments. It will try to autodetect if OpenMP is available on your system, which is very helpful for speeding up the code at very low cost of code complexity. Some people seem to experience problems compiling on Ubuntu, have a look at [Issue 19](https://github.com/karpathy/llm.c/issues/19), TLDR you'd want to modify the `CFLAGS`:

```
# try this first
CFLAGS = -Ofast -fno-fast-math -Wno-unused-result
# try this second
CFLAGS = -O3 -Wno-unused-result
```

Once `train_gpt2` is compiled, you can run it:

```bash
OMP_NUM_THREADS=8 ./train_gpt2
```

You should tune the number of threads depending on how many cores your CPU has. The program will load the model weights, the tokens, it will run a finetuning loop for a few iterations with Adam lr 1e-4, and then generate a sample from the model. The file is (I think) very readable and you should have a look. Simply, there are implementations for the forward and backward pass of all the layers, and they get strung together into a large, manual, forward/backward/update loop. The output looks like this on my MacBook Pro (Apple Silicon M3 Max):

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73323776
val loss 5.252026
step 0: train loss 5.356189 (took 1452.121000 ms)
step 1: train loss 4.301069 (took 1288.673000 ms)
step 2: train loss 4.623322 (took 1369.394000 ms)
step 3: train loss 4.600470 (took 1290.761000 ms)
... (trunctated) ...
step 39: train loss 3.970751 (took 1323.779000 ms)
val loss 4.107781
generating:
---
Come Running Away,
Greater conquer
With the Imperial blood
the heaviest host of the gods
into this wondrous world beyond.
I will not back thee, for how sweet after birth
Netflix against repounder,
will not
flourish against the earlocks of
Allay
---
```

I like how Netflix comes up, it's clear that the shadow of the training past is still lurking in the model. I did not attempt to tune the finetuning hyperparameters so it's quite likely this can be improved quite a bit. I also noticed that slightly different platforms (e.g. MacOS / Linux) will (sadly) give very slightly different results, so perhaps don't expect to get the exact numbers or generation above. Also note that if you are seeing token ids instead of text in the generation, it might be because your code is out of date, as Tokenizer decoding was added April 14, 2024. `git pull` the updates, and then re-run `python train_gpt2.py`, which will now also save the tokenizer, which C can read and then use to print text instead of token ids.

## test

I am also attaching a simple unit test for making sure our C code agrees with the PyTorch code. Compile and run with:

```bash
make test_gpt2
./test_gpt2
```

This now loads the `gpt2_124M_debug_state.bin` file, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch.

## tutorial

I attached a very small tutorial here, in [doc/layernorm/layernorm.md](doc/layernorm/layernorm.md). It's a simple, step-by-step guide to implementing a single layer of the GPT-2 model, the layernorm layer. This is a good starting point to understand how the layers are implemented in C.

## cuda

CUDA port is WIP, I'm keeping the growing collection of kernels in the `dev` folder, e.g. see [dev/cuda/README.md](dev/cuda/README.md).

As of April 10, 2024 the full forward pass is now implemented in pure CUDA in one file. First we can check that all of the logits and the final loss matches the PyTorch reference:

```bash
make test_gpt2cu
./test_gpt2cu
```

This prints `overall okay: 1`. Now that we are calculating all the right values, we can time our code. We can't train yet because the backward pass + update are not implemented yet, but we can run the training loop and see the timings:

```bash
make train_gpt2cu
./train_gpt2cu
```

This will run GPT-2 (124M) in one file of pure CUDA (see [train_gpt2.cu](train_gpt2.cu)), using batch size 4 and sequence length 1024. This will print a bunch of hyperparameters and then the "training":

```
val loss 4.517179
step 0: train loss 4.367631 (took 25.868564 ms)
step 1: train loss 4.406341 (took 26.112257 ms)
step 2: train loss 4.484756 (took 26.124789 ms)
step 3: train loss 4.345182 (took 26.149837 ms)
...
```

The loss is changing because we are still loading real data batches from our dataset, but there is no training so they won't go down over time. In any case, on my A100 40GB PCIe GPU we are seeing about 28ms/iteration. We can compare this to PyTorch fp32 training by calling our python script like this:

```bash
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4
```

Which shows time per iteration with the same hyperparameters (batch 4, time 1024) at 104ms/iteration. We can then enable `torch.compile` by adding the `--compile 1` flag:

```bash
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4 --compile 1
```

And see that the first iteration now takes 20 seconds (compilation time), but all following iterations take ~86ms. And if we additionally turn on the use of fp32 tensorcores (only GPUs since Volta) with `--tensorcores 1`:

```bash
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4 --compile 1 --tensorcores 1
```

The time drops down to 26.2ms/iteration. So at the current 26.2ms/iteration we, amusingly and right now, have an identical running time. This somewhat makes sense because most of the FLOPs are in the matmul, and we both call about the same kernels. The remainder of the difference is likely our self-attention implementation, and possibly the round trips for GeLU, and permute/unpermute.

## notable forks

will go here, similar to [llama2.c notable forks](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#notable-forks)

## discussions

Ways of organizing development:

- Experiencing a concrete issue with the repo? Use [Issues](https://github.com/karpathy/llm.c/issues).
- Have some code to contribute? Open a [PR](https://github.com/karpathy/llm.c/pulls)
- Chat about the repo, ask questions, etc.? Look at [Discussions](https://github.com/karpathy/llm.c/discussions).
- Something faster? I created a new `#llmc` channel on my [Zero to Hero Discord channel](https://discord.gg/3zy8kqD9Cp).

## license

MIT
