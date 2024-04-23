# llm.c

LLM training in simple, pure C/CUDA. There is no need for 245MB of PyTorch or 107MB of cPython. Training GPT-2 (CPU, fp32) is ~1,000 lines of clean code in the single file [train_gpt2.c](train_gpt2.c), and training it on GPU is ~2,000 lines (adds CUDA kernels) in [train_gpt2.cu](train_gpt2.cu). The code compiles and runs instantly, it exactly matches the PyTorch reference implementation, and it ~matches the speed of (compiled) PyTorch (fp32, no flash attention). I chose GPT-2 as the first working example because it is the grand-daddy of LLMs, the first time the modern stack was put together.

Our current goal is to reproduce GPT-2 with a multi-node, mixed-precision, efficient implementation. For an overview of current ongoing work, see the latest [State of the Union](https://github.com/karpathy/llm.c/discussions/224) post.

I'd like this repo to only maintain C and CUDA code. Ports of this repo to other languages are very welcome, but should be done in separate repos, and then I am happy to link to them below in the "notable forks" section, just like I did in [llama2.c notable forks](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#notable-forks).

## quick start (GPU)

The "I don't care about anything I just want to train and I have a GPU" section. Run:

```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2fp32cu
./train_gpt2fp32cu
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C/CUDA and train for one epoch on tineshakespeare with AdamW (using batch size 4, context length 1024, total of 74 steps), evaluate validation loss, and sample some text. Note that in this quickstart we are using the fp32 version [train_gpt2_fp32.cu](train_gpt2_fp32.cu) of the CUDA code. Below in the CUDA section we document the current "mainline" [train_gpt2.cu](train_gpt2.cu), which is still being very actively developed, uses mixed precision, and runs ~2X faster.

## quick start (CPU)

The "I am so GPU poor that I don't even have one" section. No worries, run:

```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. Honestly, unless you have a beefy CPU (and can crank up the number of OMP threads in the launch command), you're not going to get that far on CPU training LLMs, but it might be a good demo/reference.

## training: more detail

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

## CUDA

The full training loop is also implemented in pure CUDA in one file, but optimizations of the kernels are ongoing. Currently, we roughly match the speed of PyTorch. The way we organize code is that we have a growing collection of kernels of increasing complexity in the `dev/cuda` folder, see [dev/cuda/README.md](dev/cuda/README.md). We then copy paste the best kernels into the main training loop in the single training file `train_gpt2cu.cu`.

**WIP alert, April 23**. We merged the first version of mixed precision training code. I checkpointed the fp32 version to separate files that include `_fp32` in their filename, and would like to preserve this version in the root of the repo because it 1) doesn't require the most up to date CUDA and will a lot more likely compile and is more portable, 2) it is a lot simpler and acts as reference. The "mainline" development of the CUDA version will from here on move mostly to the [train_gpt2.cu](train_gpt2.cu) file, which includes mixed precision. In the descriptions below I will default to using the fp32 version for now because it is currently more portable and stable, then at the end I will cover to the new mixed precision version.

**Correctness**. First, we can do 10 iterations of training and verify that our code exactly matches and preproduces the numbers from PyTorch:

```bash
make test_gpt2fp32cu
./test_gpt2fp32cu
```

This prints `overall okay: 1`. So the forward activations, backward gradients, and the individual loss values for 10 iterations all match exactly.

**Training**. To train GPT-2 in a single file of CUDA, run the train script:

```bash
make train_gpt2fp32cu
./train_gpt2fp32cu
```

This will load the tiny_shakespeare dataset validation and training splits. At the default settings of B=4, T=1024, there are 8 validation batches and 74 training batches. The script is currently configured to do a single epoch of finetuning with learning rate 1e-4, and along the way it evaluates the validation performance and generates samples, e.g.:

```
step 1/74: train loss 4.367631 (80.639749 ms)
step 2/74: train loss 4.031242 (77.378867 ms)
step 3/74: train loss 4.034144 (77.315861 ms)
step 4/74: train loss 3.859865 (77.357575 ms)
...
step 72/74: train loss 3.085081 (78.850895 ms)
step 73/74: train loss 3.668018 (78.197064 ms)
step 74/74: train loss 3.467508 (78.009975 ms)
val loss 3.516490
generating:
---
?Where will you go?
I take you wherefore I can, myself, and must.
I cast off my beak, that I may look him up on the point;
For on his rock shall he be opencast.

<|endoftext|>My little nephew:
Keep on with me, my
```

This runs on my A100 in about ~10 seconds. This training loop in the PyTorch script is about 80ms/iteration, so we are slightly better than PyTorch here. However, this is measured with PyTorch that is a bit stale (I'm on 2.1.0) and we're not yet including FlashAttention or the PyTorch scaled_dot_product_attention fused operation.

We can compare to naive PyTorch like this, where we turn on `torch.compile` and the use of TensorCores, which use tf32 type:

```bash
python train_gpt2.py --write_tensors 0 --sequence_length 1024 --batch_size 4 --compile 1 --tensorcores 1
```

The compilation (first iteration) is ~27 seconds, but after that on my A100 this currently runs at ~80ms/iteration.

**Mixed precision**. The new CUDA mixed precision version, where most of the development will happen going forward, is [train_gpt2.cu](train_gpt2.cu), along with its test [test_gpt2.cu](test_gpt2.cu). Here, a lot of the calculations happen in lower-precision formats (fp16 or bf16), which allows us to run really fast (~2X of the TF32 performance above). Note that I describe the baseline implementation as `fp32` but, to be more accurate, it is actually a `tf32` (TensorFloat32). To train and test, it's the same commands just drop the fp32 parts:

```bash
make train_gpt2cu
./train_gpt2cu

make test_gpt2cu
./test_gpt2cu
```

If you have the latest CUDA you should expect this to compile OK, and you should see ~2X improved speed (~1.86X to be precise).

## experiments / sweeps

Now that the basic argparse and logging functionality is there in the .cu script, we can do our first learning rate sweeps. This is fairly manual right now, but just to document one example process to sweep learning rates on a machine with 4 GPUs on TinyStories. Run a shell script `sweep.sh` (after you of course `chmod u+x sweep.sh`):

```bash
#!/bin/bash

learning_rates=(3e-5 1e-4 3e-4 1e-3)

for i in {0..3}; do
    export CUDA_VISIBLE_DEVICES=$i
    screen -dmS "tr$i" bash -c "./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l ${learning_rates[$i]} -o stories$i.log"
done

# you can bring these down with
# screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
```

This example opens up 4 screen sessions and runs the four commands with different LRs. This writes the log files `stories$i.log` with all the losses, which you can plot as you wish in Python. Here's a quick example script to plot the losses in a Jupyter notebook, obviously can become more sophisticated later:

```python
import matplotlib.pyplot as plt
%matplotlib inline

def parse_log(logfile):
  # look for lines like e.g. "s:100 tel:1.6952", step 100, val 1.6952
    val_steps, val_losses = [], []
    with open(logfile, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "tel" in line:
            parts = line.split()
            step = parts[0].split(":")[1]
            loss = parts[1].split(":")[1]
            val_steps.append(int(step))
            val_losses.append(float(loss))
    return val_steps, val_losses

results = [parse_log(f"stories{i}.log") for i in range(0, 4)]
for i, (val_steps, val_losses) in enumerate(results):
    plt.plot(val_steps, val_losses, label="run {}".format(i))
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
```

## repo philosophy

A few more words on what I want this repo to be:

First, I want `llm.c` to be a place for education. E.g. our `dev/cuda` folder is a place for a library of kernels for all the layers that are manually hand-written and very well documented, starting from very simple kernels all the way to more complex / faster kernels. If you have a new kernel with various different tradeoffs, please feel free to contribute it here.

That said, I also want `llm.c` to be very fast too, even practically useful to train networks. E.g. to start, we should be able to reproduce the big GPT-2 (1.6B) training run. This requires that we incorporate whatever fastest kernels there are, including the use of libraries such as cuBLAS, cuBLASLt, CUTLASS, cuDNN, etc. I also think doing so serves an educational purpose to establish an expert upper bound, and a unit of measurement, e.g. you could say that your manually written kernels are 80% of cuBLAS speed, etc. Then you can choose to do a super fast run, or you can choose to "drag and drop" whatever manual kernels you wish to use, and run with those.

However, as a constraint, I want to keep the mainline `llm.c` in the root folder simple and readable. If there is a PR that e.g. improves performance by 2% but it "costs" 500 lines of complex C code, and maybe an exotic 3rd party dependency, I may reject the PR because the complexity is not worth it. In that sense I'd be ok to only be at e.g. 90% of PyTorch speed, if it means we can remain at ~2,000 readable lines of code with minimal exotic dependencies. As a concrete example - making cuBLAS for matmuls the default in the root training loop is a no-brainer: it makes the mainline code much faster, it is a single line of interpretable code, and it is a very common dependency. On the side of this, we can have manual implementations that can compete with cuBLAS in `dev/cuda`.

Lastly, I will be a lot more sensitive to complexity in the root folder of the project, which contains the main / default files of the project. In comparison, the `dev/` folder is a bit more of a scratch space for us to develop a library of kernels or classes and share useful or related or educational code, and some of this code could be ok to be (locally) complex.

## notable forks

- Mojo
  - [llm.🔥](https://github.com/dorjeduck/llm.mojo) by @[dorjeduck](https://github.com/dorjeduck): a Mojo port of this project

- C#
  - [llm.cs](https://github.com/azret/llm.cs) by @[azret](https://github.com/azret): a C# port of this project

- Rust
  -  [llm.rs](https://github.com/ToJen/llm.rs) by @[ToJen](https://github.com/ToJen): a Rust port of this project

- Metal
  - [llm.metal](https://github.com/regrettable-username/llm.metal) by @[regrettable-username](https://github.com/regrettable-username): LLM training in simple, raw C/Metal Shading Language
## discussions

Ways of organizing development:

- Experiencing a concrete issue with the repo? Use [Issues](https://github.com/karpathy/llm.c/issues).
- Have some code to contribute? Open a [PR](https://github.com/karpathy/llm.c/pulls)
- Chat about the repo, ask questions, etc.? Look at [Discussions](https://github.com/karpathy/llm.c/discussions).
- Something faster? I created a new `#llmc` channel on my [Zero to Hero Discord channel](https://discord.gg/3zy8kqD9Cp).

## license

MIT
