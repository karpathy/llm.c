# scripts

These shell scripts hold the exact commands to llm.c that reproduce the GPT-2 and GPT-3 runs.

### pytorch reference runs

For all pyrun scripts, current restrictions:

- does not write checkpoint, only logs of the train/val losses
- does not evaluate hellaswag accuracy
- cannot "resume training" (i.e. the `-y 1` flag)

### memory considerations

In any of these scripts, if you are running out of memory on your GPU you'll want to meddle with two flags: the recompute setting `-r` and the microbatch size `-b`. Recompute throws away some activations during the forward pass and then recomputes them during the backward pass. This reduces the amount of memory we need to store and cache during the forward pass, but then increases the amount of computation we need to do during the backward pass. The microbatch size controls the number of token streams that are processed in a single forward/backward pass in parallel. Decreasing this number means we need to store less memory per microbatch, but then we have to increase the number of loops in the gradient accumulation to meet the same desired total batch size.

Long story short, try `-r 1` (recompute GeLU, trading off speed and memory) to conserve some memory. If that doesn't help, start dividing the micro batch size until things fit. For example if the deafult is `-b 64`, try `-b 32`, and then 16, 8, etc. until things fit. Once they do fit, experiment with dialing back the recompute flag `-r 0` to get some speed back. Alternatively to `-b`, if your application doesn't need a very long context length, you can dial back the number of max sequence length using `-t`. For example GPT-2 uses `-t 1024` and GPT-3 uses `-t 2048`. Your application may tolerate a lower context length.

### multi-gpu considerations

It might be that you only have one GPU and not a whole box of them. Every script is fairly easy to change for just a single GPU. For llm.c, simply change line 1 to line 2 and leave everything else the same:

```bash
mpirun -np 8 ./train_gpt2cu \
./train_gpt2cu \
```

For PyTorch, the same thing:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py \
python train_gpt2.py \
```

Both of these scripts automatically detect how many GPUs are available and adjust the gradient accumulation inner loop of the optimization accordingly, so the results come out the same, up to floating point error. Of course, you'll have to wait proportionally longer for the optimization to finish.

To run on multiple nodes of GPUs, have a look at this pending [PR](https://github.com/karpathy/llm.c/pull/426), alternatively for llm.c try something like this:

```bash
mpirun -np 16 --host node1:8,node2:8 ./train_gptcu ...
```

For PyTorch follow the torchrun docs.
