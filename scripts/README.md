# scripts

These shell scripts hold the exact commands to llm.c that reproduce the GPT-2 and GPT-3 runs.


### memory considerations

In any of these scripts, if you are running out of memory on your GPU you'll want to meddle with two flags: the recompute setting `-r` and the microbatch size `-b`. Recompute throws away some activations during the forward pass and then recomputes them during the backward pass. This reduces the amount of memory we need to store and cache during the forward pass, but then increases the amount of computation we need to do during the backward pass. The microbatch size controls the number of token streams that are processed in a single forward/backward pass in parallel. Decreasing this number means we need to store less memory per microbatch, but then we have to increase the number of loops in the gradient accumulation to meet the same desired total batch size.

Long story short, try `-r 1` (recompute GeLU, trading off speed and memory) to conserve some memory. If that doesn't help, start dividing the micro batch size until things fit. For example if the deafult is `-b 64`, try `-b 32`, and then 16, 8, etc. until things fit. Once they do fit, experiment with dialing back the recompute flag `-r 0` to get some speed back. Alternatively to `-b`, if your application doesn't need a very long context length, you can dial back the number of max sequence length using `-t`. For example GPT-2 uses `-t 1024` and GPT-3 uses `-t 2048`. Your application may tolerate a lower context length.
