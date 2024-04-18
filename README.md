# llm.c

LLM training in simple, pure C/CUDA. There is no need for 245MB of PyTorch or 107MB of cPython. For example, training GPT-2 (CPU, fp32) is ~1,000 lines of clean code in a single file. It compiles and runs instantly, and exactly matches the PyTorch reference implementation. I chose GPT-2 as the first working example because it is the grand-daddy of LLMs, the first time the modern stack was put together.

Currently, we are working on:

- optimize the CUDA implementation to compete with PyTorch
- lower the precision from fp32 to mixed precision training
- reproduce the GPT-2 training run (data, evals)
- more modern architectures, Llama 2, Gemma, Mistral, etc.

I'd like this repo to only maintain C and CUDA code. Ports of this repo to other languages are very welcome, but should be done in separate repos, and then I am happy to link to them below in the "notable forks" section, just like I did in [llama2.c notable forks](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#notable-forks).

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
CFLAGS="-Ofast -fno-finite-math-only -Wno-unused-result -march=native" make train_gpt2
# try this second
CFLAGS="-O3 -Wno-unused-result -march=native" make train_gpt2
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

As of April 17, 2024 the full training loop is now implemented in pure CUDA in one file, but has not been optimized yet. The way we organize code is that we have a growing collection of kernels of increasing complexity in the `dev/cuda` folder, see [dev/cuda/README.md](dev/cuda/README.md). We then copy paste the best kernels into the main training loop in the single training file `train_gpt2cu.cu`.

**Correctness**. First, we can do 10 iterations of training and verify that our code exactly matches and preproduces the numbers from PyTorch:

```bash
make test_gpt2cu
./test_gpt2cu
```

This prints `overall okay: 1`. So the forward activations, backward gradients, and the individual loss values for 10 iterations all match exactly.

**Training**. To train GPT-2 in a single file of CUDA, run the train script:

```bash
make train_gpt2cu
./train_gpt2cu
```

This will load the tiny_shakespeare dataset validation and training splits. At the default settings of B=4, T=1024, there are 8 validation batches and 74 training batches. The script is currently configured to do a single epoch of finetuning with learning rate 1e-4, and along the way it evaluates the validation performance and generates samples, e.g.:

```
step 1/74: train loss 4.367631 (424.545089 ms)
step 2/74: train loss 4.031261 (418.123055 ms)
step 3/74: train loss 4.034113 (418.248729 ms)
step 4/74: train loss 3.859862 (419.620396 ms)
...
step 72/74: train loss 3.085115 (430.124835 ms)
step 73/74: train loss 3.667760 (428.690927 ms)
step 74/74: train loss 3.467514 (428.090348 ms)
val loss 3.516422
generating:
---
?Where will you go?
I take you wherefore I can, myself, and must.
I sat down at a full's seven, and here was my mate;
And being gone to find some gentleman to his place
Zilled me with his eagle; and thus to find my light
```

So as currently configured, we come down to validation loss of 3.51 and can generate okay samples. This runs on my A100 in about ~30 seconds. Even so, this training loop in PyTorch is about 80ms/iteration, so we are currently ~5X slower. The kernels are being actively optimized and worked on and we aspire to reach the same vicinity soon^TM.

**Timing**. Finally we can time the code and compare the speed to PyTorch. Because the backward pass CUDA kernels are still very new and being optimized, I'd recommend turning off the backward + update in `train_gpt2.cu`:

```
// gpt2_zero_grad(&model);
// gpt2_backward(&model);
// gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
```

And then re-compile and run just the forward pass in a loop:

```bash
make train_gpt2cu
./train_gpt2cu
```

This will print a bunch of hyperparameters and then you'll see the loop:

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

## repo philosophy

A few more words on what I want this repo to be:

First, I want `llm.c` to be a place for education. E.g. our `dev/cuda` folder is a place for a library of kernels for all the layers that are manually hand-written and very well documented, starting from very simple kernels all the way to more complex / faster kernels. If you have a new kernel with various different tradeoffs, please feel free to contribute it here.

That said, I also want `llm.c` to be very fast too, even practically useful to train networks. E.g. to start, we should be able to reproduce the big GPT-2 (1.6B) training run. This requires that we incorporate whatever fastest kernels there are, including the use of libraries such as cuBLAS, cuBLASLt, CUTLASS, cuDNN, etc. I also think doing so serves an educational purpose to establish an expert upper bound, and a unit of measurement, e.g. you could say that your manually written kernels are 80% of cuBLAS speed, etc. Then you can choose to do a super fast run, or you can choose to "drag and drop" whatever manual kernels you wish to use, and run with those.

However, as a constraint, I want to keep the mainline `llm.c` in the root folder simple and readable. If there is a PR that e.g. improves performance by 2% but it "costs" 500 lines of complex C code, and maybe an exotic 3rd party dependency, I may reject the PR because the complexity is not worth it. In that sense I'd be ok to only be at e.g. 90% of PyTorch speed, if it means we can remain at ~2,000 readable lines of code with minimal exotic dependencies. As a concrete example - making cuBLAS for matmuls the default in the root training loop is a no-brainer: it makes the mainline code much faster, it is a single line of interpretable code, and it is a very common dependency. On the side of this, we can have manual implementations that can compete with cuBLAS in `dev/cuda`.

Lastly, I will be a lot more sensitive to complexity in the root folder of the project, which contains the main / default files of the project. In comparison, the `dev/` folder is a bit more of a scratch space for us to develop a library of kernels or classes and share useful or related or educational code, and some of this code could be ok to be (locally) complex.

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
