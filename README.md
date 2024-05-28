# llm.c

LLM training in simple, pure C/CUDA. There is no need for 245MB of PyTorch or 107MB of cPython. Training GPT-2 (CPU, fp32) is ~1,000 lines of clean code in the single file [train_gpt2.c](train_gpt2.c), and training it on GPU is ~3,000 lines (adds CUDA kernels) in [train_gpt2.cu](train_gpt2.cu). The code compiles and runs instantly, it exactly matches the PyTorch reference implementation, and currently slightly exceeds the speed of (compiled) PyTorch (with bf16, torch compile, and flash attention). I chose GPT-2 as the first working example because it is the grand-daddy of LLMs, the first time the modern stack was put together.

Our current goal is to reproduce GPT-2. For an overview of current ongoing work, see the latest [State of the Union](https://github.com/karpathy/llm.c/discussions/344) post.

Update May 28, 2024: A useful recent post may be ["Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20"](https://github.com/karpathy/llm.c/discussions/481) where I detail the steps to go from scratch to reproducing the GPT-2 miniseries for 124M/350M models. The files themselves that show the launch commands are [run124M.sh](run124M.sh) and [run350M.sh](run350M.sh).

I'd like this repo to only maintain C and CUDA code. Ports of this repo to other languages are very welcome, but should be done in separate repos, and then I am happy to link to them below in the "notable forks" section, just like I did in [llama2.c notable forks](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#notable-forks).

## quick start (GPU, slow but stable and educational)

The "I don't care about anything I just want to train and I have a GPU" section. Run:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
make train_gpt2fp32cu
./train_gpt2fp32cu
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C/CUDA and train for one epoch on tineshakespeare with AdamW (using batch size 4, context length 1024, total of 74 steps), evaluate validation loss, and sample some text. Note that in this quickstart we are using the fp32 version [train_gpt2_fp32.cu](train_gpt2_fp32.cu) of the CUDA code. In the next section we document the current "mainline" [train_gpt2.cu](train_gpt2.cu), which uses mixed precision, and runs ~2X faster.

## quick start (GPU, fast bleeding edge)

I want to see it go fast. In this case switch to our mainline, most optimized `train_gpt2.cu`. Run:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
make train_gpt2cu
./train_gpt2cu
```

If you additionally install cuDNN (see the CUDA section below), you can go even faster with flash attention. Adjust the make command as follows to compile with cudnn / flash attention:

```bash
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu
```

Note that the default batch size is very low (4). If you have enough memory on your GPU, I recommend you increase this to e.g. 32:

```bash
./train_gpt2cu -b 32
```

My standard single-GPU "prod" run (e.g. with a A100 40GB) trains on TinyStories instead of TinyShakespeare and looks like this, as an example:

```bash
python dev/data/tinystories.py
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu -i dev/data/tinystories/TinyStories_train.bin \
               -j dev/data/tinystories/TinyStories_val.bin \
               -v 250 -s 250 -g 144 -o stories.log -b 32
```

The `-i` flag is a glob pattern for the input data, `-j` for the val data. In addition I decrease the frequency of validation loss and sampling to every 250 steps, sample 144 tokens during sampling stage (to fit ~one story), and at batch size 32.

If you want to train on actual, real pretraining data, check out the recently added support for [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Unlike the datasets above where the train/val tokens fit into a single .bin file, we now have multiple data shards as well. Here is an example:

```
# write fineweb data in 100M token shards to dev/data/fineweb10B
python dev/data/fineweb.py -s 100000000
# compile and run
./train_gpt2cu -i "dev/data/fineweb10B/fineweb_train_*.bin" \
               -j "dev/data/fineweb10B/fineweb_val_*.bin" \
               -v 250 -s 250 -g 144 -o fineweb.log -b 32
```

Where you will notice the use of glob pattern `*` to match all the train shards.

## quick start (multiple GPUs)

Great, let's get even more serious. We're using MPI and NCCL for multi-GPU training. Everything in the section above applies, with the following changes:

```bash
# example to install MPI:
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
# the run command is now preceeded by `mpirun`:
mpirun -np <number of GPUs on your machine> ./train_gpt2cu
```

Sub in the number of GPUs you'd like to run on in the last command. All of the flags discussed in the section above apply here as well.

## quick start (CPU)

The "I am so GPU poor that I don't even have one" section. You can still train! But you won't go too far. You can still finetune a GPT-2 small (124M parameter model) to output Shakespeare-like text, as an example:

```bash
pip install -r requirements.txt
python dev/data/tinyshakespeare.py
python train_gpt2.py
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. Honestly, unless you have a beefy CPU (and can crank up the number of OMP threads in the launch command), you're not going to get that far on CPU training LLMs, but it might be a good demo/reference.

## training: more detail

The data files inside `/dev/data/(dataset).py` are responsible for downloading, tokenizing and saving the tokens to file. So for example when you run:

```bash
python dev/data/tinyshakespeare.py
```

We download and tokenize the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. The output of this looks like this:

```
writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin
writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin
```

The .bin files contain a short header (1024 bytes) and then a stream of tokens in uint16, indicating the token ids with the GPT-2 tokenizer. More datasets are available in `/dev/data`.

In principle, once we have the tokens, we'd be ready to train the model right here. However, current code can't start training from scratch just yet (coming very soon), so we initialize training from the pretrained models released by OpenAI and do finetuning. For that, we have to download the GPT-2 weights and save them as a checkpoint we can load in C. This is what happens when you run this script:

```bash
python train_gpt2.py
```

You'll recognize this code from nanoGPT as a simple GPT-2 reference implementation in PyTorch. This script will download the GPT-2 (124M) model, overfit a single batch of data for 10 iterations, run a few steps of generation, and most importantly it will save three files: 1) the `gpt2_124M.bin` file that contains the raw model weights for loading in C, 2) the `gpt2_124M_debug_state.bin`, which also contains more debug state: the inputs, targets, logits and loss (useful for debugging and unit testing), and finally 3) the `gpt2_tokenizer.bin` which stores the vocabulary for the GPT-2 tokenizer, translating token ids to byte sequences of UTF-8 encoded string pieces. The file also saves both the fp32 versions of the above, and the bfloat16 versions of them for mixed precision training. We can now initialize with these model weights and continue training in raw C. Then we compile the training programs with `make`. There are currently three parallel implementations:

```bash
# the simple, CPU, reference code version
make train_gpt2
# the single-GPU fp32 CUDA version
make train_gpt2fp32cu
# the multi-GPU mixed precision CUDA version
make train_gpt2cu
```

You can have a look inside the `Makefile` and its comments. It will try to autodetect a lot of tools and libraries (e.g. cuDNN, OpenMP, OpenMPI, nvcc), and you want to get as many checkmarks as possible. For example when I run `make train_gpt2cu USE_CUDNN=1` on my fully configured machine, we see:

```
✓ cuDNN found, will run with flash-attention
✓ OpenMP found
✓ OpenMPI found, OK to train with multiple GPUs
✓ nvcc found, including GPU/CUDA support
```

Some people seem to experience problems compiling on Ubuntu, have a look at [Issue 19](https://github.com/karpathy/llm.c/issues/19), TLDR you'd want to modify the `CFLAGS`:

```
# try this first
CFLAGS="-Ofast -fno-finite-math-only -Wno-unused-result -march=native" make train_gpt2
# try this second
CFLAGS="-O3 -Wno-unused-result -march=native" make train_gpt2
```

Once the binary is compiled, we can run it. For example the simplest CPU reference version runs as:

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

I like how Netflix comes up, it's clear that the shadow of the training past is still lurking in the model. I did not attempt to tune the finetuning hyperparameters so it's quite likely this can be improved quite a bit. I also noticed that slightly different platforms (e.g. MacOS / Linux) will (sadly) give very slightly different results, so perhaps don't expect to get the exact numbers or generation above.

Finally, the code is in flux. If anything weird happens that you didn't expect or that worked previously, try to `git pull`, re-run all the commands above, reference back to this README, etc.

## test

I am also attaching a simple unit test for making sure our C code agrees with the PyTorch code. On the CPU as an example, compile and run with:

```bash
make test_gpt2
./test_gpt2
```

This now loads the `gpt2_124M_debug_state.bin` file, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch. To test the GPU version I run:

```bash
# fp32 test (cudnn not supported)
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu
# mixed precision cudnn test
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu
```

## tutorial

I attached a very small tutorial here, in [doc/layernorm/layernorm.md](doc/layernorm/layernorm.md). It's a simple, step-by-step guide to implementing a single layer of the GPT-2 model, the layernorm layer. This is a good starting point to understand how the layers are implemented in C.

## CUDA

The full training loop is also implemented in pure CUDA in one file, but optimizations of the kernels are ongoing. Currently, we slightly exceed the speed of PyTorch Nightly. The way we organize code is that we have a growing collection of kernels of increasing complexity in the `dev/cuda` folder, see [dev/cuda/README.md](dev/cuda/README.md). We then copy paste the best kernels into the main training loop in the single training file `train_gpt2cu.cu`.

**WIP alert, April 23**. We merged the first version of mixed precision training code. I checkpointed the fp32 version to separate files that include `_fp32` in their filename, and would like to preserve this version in the root of the repo because it 1) doesn't require the most up to date CUDA and will a lot more likely compile and is more portable, 2) it is a lot simpler and acts as reference. In fact, we'd like to diverge the fp32 version in the direction of being pure CUDA (e.g. do not even call cuBLAS by default), to be used as an educational reference, maybe even a kernel of a course on CUDA. The "mainline" development concerned with speed will from there on move to the [train_gpt2.cu](train_gpt2.cu) file, which includes mixed precision training.

In the descriptions below I will default to using the fp32 version for now because it is currently more portable and stable, then at the end I will cover to the new mixed precision version.

**Correctness**. First, we can do 10 iterations of training and verify that our code exactly matches and preproduces the numbers from PyTorch:

```bash
make test_gpt2fp32cu
./test_gpt2fp32cu
```

This prints `overall okay: 1`. So the forward activations, backward gradients, and the individual loss values for 10 iterations all match exactly.

**Training**. To train on single GPU in fp32:

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

This runs on my A100 in about ~10 seconds. We can compare to naive PyTorch like this, where we turn on `torch.compile` and the use of TensorCores, which use tf32 type:

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

If you have the latest CUDA you should expect this to compile OK, and you should see a lot more improved performance.

**Flash Attention**. As of May 1, 2024 we now support the Flash Attention from cuDNN. Because cuDNN bloats the compile time from a few seconds to ~minute and this code path is right now very new, this is disabled by default. You can enable it by compiling like this:

```bash
make train_gpt2cu USE_CUDNN=1
```

This will try to compile with cudnn and run it. You have to have cuDNN installed on your system. The [cuDNN installation instructions](https://developer.nvidia.com/cudnn) with apt-get will grab the default set of cuDNN packages. For a minimal setup, the cuDNN dev package is sufficient, e.g. on Ubuntu 22.04 for CUDA 12.x:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

On top of this you need the [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/tree/main), but this is just header files. Simply clone the repo to your disk. The Makefile currently looks for it in either your home directory or the current directory. If you have put it elsewhere, add `CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include` to the `make` command-line.

**Multi-GPU training**. As of April 26, 2024 there is now also support for multi-GPU training using MPI and NCCL. Make sure you install MPI, e.g. on Linux:

```bash
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

and then:

```bash
make train_gpt2cu
mpirun -np <number of GPUs> ./train_gpt2cu
```

The fp32 version of the code does not support multi-GPU. This is because we want the GPT-2 fp32 version to become a nice educational endpoint of a CUDA optimization course. The mixed precision version is where we are doing the cutting edge development, so this is the version that supports multi-GPU training.

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

This example opens up 4 screen sessions and runs the four commands with different LRs. This writes the log files `stories$i.log` with all the losses, which you can plot as you wish in Python. A quick example of how to parse and plot these logfiles is in [dev/vislog.ipynb](dev/vislog.ipynb).

## repo philosophy

A few more words on what I want this repo to be:

First, I want `llm.c` to be a place for education. E.g. our `dev/cuda` folder is a place for a library of kernels for all the layers that are manually hand-written and very well documented, starting from very simple kernels all the way to more complex / faster kernels. If you have a new kernel with various different tradeoffs, please feel free to contribute it here.

That said, I also want `llm.c` to be very fast too, even practically useful to train networks. E.g. to start, we should be able to reproduce the big GPT-2 (1.6B) training run. This requires that we incorporate whatever fastest kernels there are, including the use of libraries such as cuBLAS, cuBLASLt, CUTLASS, cuDNN, etc. I also think doing so serves an educational purpose to establish an expert upper bound, and a unit of measurement, e.g. you could say that your manually written kernels are 80% of cuBLAS speed, etc. Then you can choose to do a super fast run, or you can choose to "drag and drop" whatever manual kernels you wish to use, and run with those.

However, as a constraint, I want to keep the mainline `llm.c` in the root folder simple and readable. If there is a PR that e.g. improves performance by 2% but it "costs" 500 lines of complex C code, and maybe an exotic 3rd party dependency, I may reject the PR because the complexity is not worth it. As a concrete example - making cuBLAS for matmuls the default in the root training loop is a no-brainer: it makes the mainline code much faster, it is a single line of interpretable code, and it is a very common dependency. On the side of this, we can have manual implementations that can compete with cuBLAS in `dev/cuda`.

Lastly, I will be a lot more sensitive to complexity in the root folder of the project, which contains the main / default files of the project. In comparison, the `dev/` folder is a bit more of a scratch space for us to develop a library of kernels or classes and share useful or related or educational code, and some of this code could be ok to be (locally) complex.

## notable forks

- AMD support
  - [llm.c](https://github.com/anthonix/llm.c) by @[anthonix](https://github.com/anthonix): support for AMD devices, such as the 7900 XTX

- C#
  - [llm.cs](https://github.com/azret/llm.cs) by @[azret](https://github.com/azret): a C# port of this project
  - [Llm.cs](https://github.com/nietras/Llm.cs) by @[nietras](https://github.com/nietras): a C# port of this project with focus on easy to get started on any platform. Clone and run ✅

- CUDA C++
  - [llm.cpp](https://github.com/gevtushenko/llm.c) by @[gevtushenko](https://github.com/gevtushenko): a port of this project using the [CUDA C++ Core Libraries](https://github.com/NVIDIA/cccl)
     - A presentation this fork was covered in [this lecture](https://www.youtube.com/watch?v=WiB_3Csfj_Q) in the [CUDA MODE Discord Server](https://discord.gg/cudamode)

- Go
  - [llm.go](https://github.com/joshcarp/llm.go) by @[joshcarp](https://github.com/joshcarp): a Go port of this project

- Java
  - [llm.java](https://github.com/harryjackson/llm.java) by @[harryjackson](https://github.com/harryjackson): a Java port of this project

- Metal
  - [llm.metal](https://github.com/regrettable-username/llm.metal) by @[regrettable-username](https://github.com/regrettable-username): LLM training in simple, raw C/Metal Shading Language

- Mojo
  - [llm.🔥](https://github.com/dorjeduck/llm.mojo) by @[dorjeduck](https://github.com/dorjeduck): a Mojo port of this project

- Rust
  -  [llm.rs](https://github.com/yijunyu/llm.rs) by @[Yijun Yu](https://github.com/yijunyu): a Rust rewrite with the aim to have same performance
  -  [llm.rs](https://github.com/ToJen/llm.rs) by @[ToJen](https://github.com/ToJen): a Rust port of this project

- Swift
  - [llm.swift](https://github.com/otabuzzman/llm.swift) by @[otabuzzman](https://github.com/otabuzzman): a Swift port of this project

- Zig
  - [llm.zig](https://github.com/Saimirbaci/llm.zig) by @[saimirbaci](https://github.com/Saimirbaci): a Zig port of this project


## major changes log

- **May 21, 2024: Dataset refactor**. I refactored the .bin files that hold the tokens to include a header like all the other .bin files that e.g. store the model weights. This was necessary to support multiple versions and future development. Unfortunately, this will brick everyone's master the next time you `git pull`, because the .bin files you've generated before are the legacy version. To fix this, you only have to re-generate the data in the new format. For example, for Tiny Shakespeare run: `python dev/data/tinyshakespeare.py`. For Tiny Stories, `python dev/data/tinystories.py`. Also notice that the location of these data files has changed. They used to just be "flat" and inside `data/` folder, but now all the data-related code was moved to `dev/data` files and sub-directories, to keep things organized. Apologies for breaking change, I'll try not to brick master too much in general.

## discussions

Ways of organizing development:

- Experiencing a concrete issue with the repo? Use [Issues](https://github.com/karpathy/llm.c/issues).
- Have some code to contribute? Open a [PR](https://github.com/karpathy/llm.c/pulls)
- Chat about the repo, ask questions, etc.? Look at [Discussions](https://github.com/karpathy/llm.c/discussions).
- Something faster? I created a new `#llmc` channel on my [Zero to Hero Discord channel](https://discord.gg/3zy8kqD9Cp).

## license

MIT
