
# layernorm

Quick tutorial. Let's look at how LayerNorm is handled, as one example layer in the model. We start with the [PyTorch docs for LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). LayerNorm of course comes from this original paper by [Ba et al. 2016](https://arxiv.org/abs/1607.06450), and was incorporated into the Transformer in [Vaswani et al.](https://arxiv.org/abs/1706.03762) famous paper Attention is All You Need. [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) picked up the same architecture as the Transformer, but the position of the LayerNorm was famously moved into what is now called the pre-normalization version. That is, the residual path of the Transformer is kept clean, and the LayerNorms are now the first layer of each block of the Transformer. This positively improves training stability.

The first thing to note when looking at [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) is that you will most likely not be able to find the actual implementation of the equation. That's because it is buried 30 layers deep in the code, behind an inscrutable dynamical dispatcher, in some possibly auto-generated CUDA code (for those who are interested in details, see [layer_norm.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp) and  [layer_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu)). This is done because PyTorch really really cares about efficiency, fair enough. For our purposes though, we have to start by first implementing LayerNorm manually using simpler PyTorch operations. This will be a lot less efficient than just forwarding a `LayerNorm` module, but it is algorithmically instructive. So here is the direct implementation of the math of LayerNorm using simpler PyTorch operations:

```python
import torch
eps = 1e-5

class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        # x is the input activations, of shape B,T,C
        # w are the weights, of shape C
        # b are the biases, of shape C
        B, T, C = x.size()
        # calculate the mean
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        # calculate the variance
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        # calculate the inverse standard deviation: **0.5 is sqrt, **-0.5 is 1/sqrt
        rstd = (var + eps) ** -0.5 # B,T,1
        # normalize the input activations
        norm = xshift * rstd # B,T,C
        # scale and shift the normalized activations at the end
        out = norm * w + b # B,T,C

        # return the output and the cache, of variables needed later during the backward pass
        cache = (x, w, mean, rstd)
        return out, cache
```

The activation tensors in the residual path of the Transformer during training are 3-dimensional arrays (tensors), of shape `B,T,C`. B is the batch size, T is time, and C is channels. For example, B=8, T=1024, C=768 is one setting you might see, for the smallest (124 million parameter) GPT-2 model.

We can forward this layer with some random numbers:

```python
B = 2 # some toy numbers here
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)
```

What we get out is the tensor `out`, also of shape `B,T,C`, where each C-dimensional "fibre" of activations (as we call them) is normalized and then scaled and at the end also shifted by the weights and biases of this layer. Notice that, importantly, we also return a variable `cache`, which is a tuple of the input activations `x`, the weights `w`, the mean `mean`, and the reciprocal standard deviation `rstd`. These are all variables we need during the backward pass.

PyTorch can of course do the backward pass of this layer for us with its Autograd. Let's do that first:

```python
dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()
```

You see here that we created a `fakeloss`, which simply takes a (random) weighted combination of all the outputs of our layernorm. All this is doing is projecting all of the `B,T,C` numbers into a single scalar value (loss), so that we have a single output of our "computational graph". Typically this would be the loss of the model, but here we're just doing a fake loss. We then call `backward()` on this scalar, and PyTorch will compute all the gradients for us on all the inputs to this graph - i.e. the input activations `x`, the weights `w`, and the biases `b`. If you don't know too much about autograd, I'd encourage you to watch my [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) video, where we build a tiny autograd engine. So the magic of PyTorch autograd is that after we call `.backward`, it will populate the `.grad` attribute of all the tensors that have `requires_grad=True` with the gradients of the loss with respect to that tensor. These gradients are telling us the slope of the loss for all of the input numbers in x,w,b. Therefore, the shape of `x.grad`, `w.grad`, and `b.grad` are exactly the same as the shape of `x`, `w`, and `b`.

But we don't want to use PyTorch Autograd. We want to do the backward pass manually. So we take out pen and paper and write out the expression for LayerNorm. The forward pass has the following mathematical form:

$\text{LayerNorm}(x) = w \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$

where $\odot$ is elementwise multiplication, $\mu$ is the mean, $\sigma^2$ is the variance, and $\epsilon$ is a small constant to avoid division by zero. Remembering the rules of differentiation from calculus, we now want to derive the gradients. For this part, my video [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) could be very helpful, as I work through (in detail) a similar layer - the Batch Normalization layer. When you work through the differentiation, you'll notice that the expressions simplify analytically and you can move the terms around and simplify the expression somehwat. So you don't have to manually backward every individual line in the forward pass. In particular, we get:

```python
    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db
```

So given the gradients on every individual output number stored in `dout`, and the `cache` from the forward pass, we can now backward through this layer into the inputs, to continue the chain rule of the backward pass. So now we can do our own backward pass and see that they match (the errors are tiny):

```python
dx, dw, db = LayerNorm.backward(dout, cache)
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())
```

Notice one more thing. Inside the backward pass we recomputed the variable `norm`. We already calculated this variable in the forward pass but then we threw it away! Couldn't we have made this also be a part of the `cache` and save this recompute? Actually, we very well could and you'd of course get the exact same results. The amount of stuff we save into our `cache` is completely up to us. We didn't even have to save `mean` and `rstd` either, and we could have recomputed them in the backward pass. The difference is that `mean` and `rstd` are very small, only of shape `B,T`, where as `norm` is of shape `B,T,C`. So this is simply a tradeoff between memory and compute. By not keeping `norm` in the cache, we are saving memory, but we are trading it off for a bit of compute later in the backward pass. This is very common in all the layers, and you'll see that different implementations of various layers in deep learning frameworks may all have different "checkpointing settings". Yes, confusingly enough, this is called checkpointing and has nothing to do with saving the model weights to disk. It's about saving intermediate variables in the forward pass to save compute in the backward pass.

Okay so that's the version with PyTorch tensors. Now we have to move this to C and get rid of the Tensor abstraction. Before I give you the full implementation of the forward pass, a brief word on Tensors. What are Tensors? They are 1) a 1D block of memory called Storage that holds the raw data, and 2) a View over that storage that holds its shape. [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) could be helpful here. So for example if we have the 3D tensor:

```python
torch.manual_seed(42)
B, T, C = 2, 3, 4
a = torch.randn(B, T, C)
print(a)

tensor([[[ 1.9269,  1.4873,  0.9007, -2.1055],
         [ 0.6784, -1.2345, -0.0431, -1.6047],
         [ 0.3559, -0.6866, -0.4934,  0.2415]],

        [[-1.1109,  0.0915, -2.3169, -0.2168],
         [-0.3097, -0.3957,  0.8034, -0.6216],
         [-0.5920, -0.0631, -0.8286,  0.3309]]])
```

This is 2x3x4 Tensor, but the underlying memory of it is just one single 1D array of size 2\*3\*4=24. The View is just a shape over this 1D array. So now when we index into this PyTorch tensor, for example `a[1,2,3]`, PyTorch computes the offset into the 1D array as `1*3*4 + 2*4 + 3 = 23`, and return the value at that offset. The general formula is that if you want to retrieve any element `b,t,c`, you compute the offset into Storage as `b*T*C + t*C + c`. So for example:

```python
b,t,c = 1,2,3
print(a[b,t,c])
print(a.view(-1)[b*T*C + t*C + c])
```

Both of these print 0.3309. So in this way, we know how to access all the individual elements, and how to offset all the pointers. Notice in particular that the channel dimension is the innermost dimension. So as we increase offset by 1, we are traversing the channel dimension. This is important to consider for the memory layout of our C implementation. The equivalent forward pass in C becomes:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
```

You'll see how I offset the pointer to the `inp[b,t]`, and then you know that the next `C` elements are the channels of that position in (batch, time). And the backward pass:

```c
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}
```

One additional detail to note is that we always += into the gradients. We never use = and we never use *=. This is important stylistically because if you have one variable used multiple times in a graph, the backward pass gradients always add up. In this repo this is not important because we don't have exotic branching, but it's proper. So during training we always first do `zero_grad` to set all the gradients to zero, and then we accumulate into them during backward pass.

One more note on differences between training and inference. Some of you may have already seen my earlier project [llama2.c](https://github.com/karpathy/llama2.c), which inferences Llama 2 architecture in pure C. Unlike GPT-2, Llama 2 swaps out LayerNorm for the much simpler RMSNorm. You can see the implementation of the [RMSNorm in llama2.c](https://github.com/karpathy/llama2.c/blob/master/run.c#L182), copy pasting it here:

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
```

How does this differ to our LayerNorm above?

- First, algorithmically, you'll notice that RMSNorm does not keep track of or subtract the mean, it only normalizes by the norm. Notice: norm, not standard deviation, because we did not subtract the mean. This is a simplification of the layer that has now become very trendy because it works just as well, if not slightly better. Also, the RMSNorm does not have biases, it only has a weight for scaling after normalization. In general, GPT-2 used way too many biases everywhere and it turns out you can remove these - from all the Linear Layers and from LayerNorms. The network can "simulate" biases if it needs them, e.g. by allocating one of the channel dimensions to be constant (data-independent), and then any weight multiplying that constant dimension will effectively work like a bias. This significantly simplies a lot of the code.
- Second, the inference code has no batch dimension B, i.e. the batch size is assumed to be 1. You could in principle have batched inference as well, especially if you wish to host an LLM that you expect many simultaneous queries to. But if you're just running an LLM locally, chances are you just want to have a single "stream" of generation, so there is no batch size for parallelism that could support multiple streams at once. To keep things simple, llama2.c is not batched, and therefore you won't see any loops that look like `for (int b = 0; b < B; b++)`.
- Third, this inference code has no time dimension T within this individual layer. During training, we can loop over time inside each layer and calculate the layernorm at all time steps. But during inference, we have to generate one token at a time, feeding the token predicted at time `t` into the forward pass of the Transformer at the next time step `t+1`. So this is why you don't see any loops that look like `for (int t = 0; t < T; t++)` inside individual layers. This loop over time [does exist](https://github.com/karpathy/llama2.c/blob/master/run.c#L747), but it is on the outside of the Transformer forward pass.
- You'll see that we don't keep track of any intermediate calculations, memory, or cache. That's because during inference, there is no `.backward` pass that will follow. We only need to calculate the output, and we don't need to keep any intermediate variables around. As a result, the memory consumption of inference is significantly lower than that of training. We can afford to just discard activations, and only keep memory for the "activation frontier". Similarly, there is no need to implement the `backward` function for this RMSNorm anywhere, as there is no backward pass.

As a result of all these difference, training is significantly more complex and involved, both algorithmically and computationally, and that's partly why I started by writing inference (llama2.c) before I implemented training (llm.c, here). Finally, I am attaching two helper files to this same directory that have the complete code. First:

```
python layernorm.py
```

To write out the reference data from PyTorch. Then compile and run the C version:

```
gcc layernorm.c -o layernorm -lm
./layernorm
```

You'll see that everything matches ok.

This was just the LayerNorm. We go through the exact same process for all the other layers. Most of the other layers are actually easier than LayerNorm. Hope that helps!
