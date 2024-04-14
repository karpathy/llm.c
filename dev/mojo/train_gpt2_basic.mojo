from collections.vector import InlinedFixedVector
from math import sqrt,rsqrt,exp,tanh,cosh,log,pow
from memory import memset_zero,memcpy
from python import Python
from time import now

alias RU32_HEX = 0x2545F4914F6CDD1D
alias RF32_DIV = 16777216.0

alias dtype = DType.float32
alias FLOAT = SIMD[dtype,1]

alias dtype_int = DType.int32
alias INT = SIMD[dtype_int, 1]

alias NULL = DTypePointer[dtype]()
alias NULL_INT = DTypePointer[dtype_int]()
alias M_PI:FLOAT = 3.14159265358979323846

alias GPT2_EOT=50256

alias EXIT_1 = external_call["exit",Int](1)

alias SIZEOF_INT = sizeof[DType.int32]()
alias SIZEOF_FLOAT = sizeof[DType.float32]()

## ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes

fn encoder_forward(out:DTypePointer[dtype], inp:DTypePointer[dtype_int], wte:DTypePointer[dtype], wpe:DTypePointer[dtype],B:Int32,T:Int32,C:Int32):
    for b in range(B):
        for t in range(T):
            # seek to the output position in out[b,t,:]
            var out_bt:DTypePointer[dtype] = out + b * T * C + t * C
            # get the index of the token at inp[b, t]
            var ix:Int32 = inp[b * T + t]
            # seek to the position in wte corresponding to the token
            var wte_ix:DTypePointer[dtype] = wte + ix * C
            # seek to the position in wpe corresponding to the position
            var wpe_t:DTypePointer[dtype] = wpe + t * C
            # add the two vectors and store the result in out[b,t,:]
            for i in range(C):
                out_bt[i] = wte_ix[i] + wpe_t[i]
            
fn encoder_backward(dwte:DTypePointer[dtype], dwpe:DTypePointer[dtype],dout:DTypePointer[dtype], inp:DTypePointer[dtype_int],B:Int32,T:Int32,C:Int32):
    for b in range(B):
        for t in range(T):
            var dout_bt:DTypePointer[dtype] = dout + b * T * C + t * C
            var ix:Int32 = inp[b * T + t]
            var dwte_ix:DTypePointer[dtype] = dwte + ix * C
            var dwpe_t:DTypePointer[dtype] = dwpe + t * C
            for i in range(C):
                var d:FLOAT = dout_bt[i]
                dwte_ix[i] += d
                dwpe_t[i] += d
            

fn layernorm_forward(inout out:DTypePointer[dtype], mean:DTypePointer[dtype], rstd:DTypePointer[dtype],inp:DTypePointer[dtype], weight:DTypePointer[dtype], bias:DTypePointer[dtype],B:Int32,T:Int32,C:Int32):
   
    var eps:FLOAT = 1e-5

    for b in range(B):
        for t in range(T):
            # seek to the input position inp[b,t,:]
            var x:DTypePointer[dtype] = inp + b * T * C + t * C
            # calculate the mean
            var m:FLOAT = 0.0
            for i in range(C):
                m += x[i]
            
            m = m/C.to_int()
            # calculate the variance (without any bias correction)
            var v:FLOAT = 0.0
            for i in range(C):
                var xshift:FLOAT = x[i] - m
                v += xshift * xshift
            
            v = v/C.to_int()
           
            # calculate the rstd
            var s:FLOAT = 1.0 * rsqrt(v + eps)

            # seek to the output position in out[b,t,:]
            var out_bt:DTypePointer[dtype] = out + b * T * C + t * C
            for i in range(C):
                var n:FLOAT = (s * (x[i] - m)) # normalized output
                var o:FLOAT = n * weight[i] + bias[i] # scale and shift it
                out_bt[i] = o # write

            # cache the mean and rstd for the backward pass later
            mean[b * T + t] = m
            rstd[b * T + t] = s
        
fn layernorm_backward( dinp:DTypePointer[dtype], dweight:DTypePointer[dtype], dbias:DTypePointer[dtype],
                        dout:DTypePointer[dtype], inp:DTypePointer[dtype], weight:DTypePointer[dtype], mean:DTypePointer[dtype], rstd:DTypePointer[dtype],
                        B:Int32,T:Int32,C:Int32):
    for b in range(B):
        for t in range(T):
            var dout_bt:DTypePointer[dtype] = dout + b * T * C + t * C
            var inp_bt:DTypePointer[dtype] = inp + b * T * C + t * C
            var dinp_bt:DTypePointer[dtype] = dinp + b * T * C + t * C
            var mean_bt:FLOAT = mean[b * T + t]
            var rstd_bt:FLOAT = rstd[b * T + t]

            # first: two reduce operations
            var dnorm_mean:FLOAT = 0.0
            var dnorm_norm_mean:FLOAT = 0.0
            for i in range(C):
                var norm_bti:FLOAT = (inp_bt[i] - mean_bt) * rstd_bt
                var dnorm_i:FLOAT = weight[i] * dout_bt[i]
                dnorm_mean += dnorm_i
                dnorm_norm_mean += dnorm_i * norm_bti
            
            dnorm_mean = dnorm_mean / C.to_int()
            dnorm_norm_mean = dnorm_norm_mean / C.to_int()

            # now iterate again and accumulate all the gradients
            for i in range(C):
                var norm_bti:FLOAT = (inp_bt[i] - mean_bt) * rstd_bt
                var dnorm_i:FLOAT = weight[i] * dout_bt[i]
                # gradient contribution to bias
                dbias[i] += dout_bt[i]
                # gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i]
                # gradient contribution to input
                var dval:FLOAT = 0.0
                dval += dnorm_i # term 1
                dval -= dnorm_mean # term 2
                dval -= norm_bti * dnorm_norm_mean # term 3
                dval *= rstd_bt # final scale
                dinp_bt[i] += dval
            
fn matmul_forward( out:DTypePointer[dtype],
                    inp:DTypePointer[dtype], weight:DTypePointer[dtype], bias:DTypePointer[dtype],
                    B:Int32,T:Int32,C:Int32,OC:Int32):
    # most of the running time is spent here and in matmul_backward
    # OC is short for "output channels"
    # inp is (B,T,C), weight is (OC, C), bias is (OC)
    # out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            var out_bt:DTypePointer[dtype] = out + b * T * OC + t * OC
            var inp_bt:DTypePointer[dtype] = inp + b * T * C + t * C
            for o in range(OC):
                var val:FLOAT =  0.0
                if bias != NULL:
                    val = bias[o]
                var wrow:DTypePointer[dtype] = weight + o*C
                for i in range(C):
                    val += inp_bt[i] * wrow[i]
                
                out_bt[o] = val

fn matmul_backward( dinp:DTypePointer[dtype], dweight:DTypePointer[dtype], dbias:DTypePointer[dtype],
                      dout:DTypePointer[dtype], inp:DTypePointer[dtype], weight:DTypePointer[dtype],
                     B:Int32,T:Int32,C:Int32,OC:Int32):
    # most of the running time is spent here and in matmul_forward
    # this backward could be done in a single "round" of loops
    # but that doesn't afford an efficient parallelization strategy

    # backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            var dout_bt:DTypePointer[dtype] = dout + b * T * OC + t * OC
            var dinp_bt:DTypePointer[dtype] = dinp + b * T * C + t * C
            for o in range(OC):
                var wrow:DTypePointer[dtype] = weight + o*C
                var d:FLOAT = dout_bt[o]
                for i in range(C):
                    dinp_bt[i] += wrow[i] * d
                
    # backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for o in range(OC):
        for b in range(B):
            for t in range(T):
                var dout_bt:DTypePointer[dtype] = dout + b * T * OC + t * OC
                var inp_bt:DTypePointer[dtype] = inp + b * T * C + t * C
                var dwrow:DTypePointer[dtype] = dweight + o*C
                var d:FLOAT = dout_bt[o]
                if (dbias != NULL): 
                    dbias[o] += d 
                for i in range(C):
                    dwrow[i] += inp_bt[i] * d
                
fn attention_forward( out:DTypePointer[dtype], preatt:DTypePointer[dtype], att:DTypePointer[dtype],
                        inp:DTypePointer[dtype],
                       B:Int32,T:Int32,C:Int32,NH:Int32):
    # input is (B, T, 3C) Q,K,V
    # preatt, att are (B, NH, T, T)
    # output is (B, T, C)
    var C3:Int32 = C*3
    var hs:Int32 = C / NH # head size
    var scale:FLOAT = 1.0 * rsqrt(hs.cast[dtype]())

    #pragma omp parallel for collapse(3)
    for b in range(B):
        for t in range(T):
            for h in range(NH):
                var query_t:DTypePointer[dtype] = inp + b * T * C3 + t * C3 + h * hs
                var preatt_bth:DTypePointer[dtype] = preatt + b*NH*T*T + h*T*T + t*T
                var att_bth:DTypePointer[dtype] = att + b*NH*T*T + h*T*T + t*T

                # pass 1: calculate query dot key and maxval
                var maxval:FLOAT = -10000.0 # TODO something better
                
                for t2 in range(t+1):
                    var key_t2:DTypePointer[dtype] = inp + b * T * C3 + t2 * C3 + h * hs + C # +C because it's key

                    # (query_t) dot (key_t2)
                    var val:FLOAT = 0.0
                    for i in range(hs):
                        val += query_t[i] * key_t2[i]
                    
                    val *= scale
                    if (val > maxval):
                        maxval = val
                    
                    preatt_bth[t2] = val
                
                # pass 2: calculate the exp and keep track of sum
                var expsum:FLOAT = 0.0
                
                for t2 in range(t+1):
                    var expv:FLOAT = exp(preatt_bth[t2] - maxval)
                    expsum += expv
                    att_bth[t2] = expv
                
                var expsum_inv:FLOAT =  1.0 / expsum
                if expsum == 0.0:
                    expsum_inv = 0.0

                # pass 3: normalize to get the softmax
                for t2 in range(T):
                    if (t2 <= t):
                        att_bth[t2] *= expsum_inv
                    else:
                        # causal attention mask. not strictly necessary to set to zero here
                        # only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0
                    
                # pass 4: accumulate weighted values into the output of attention
                var out_bth:DTypePointer[dtype] = out + b * T * C + t * C + h * hs
                for i in range(hs): 
                    out_bth[i] = 0.0 
                for t2 in range(t+1):
                    var value_t2:DTypePointer[dtype] = inp + b * T * C3 + t2 * C3 + h * hs + C*2 # +C*2 because it's value
                    var att_btht2:FLOAT = att_bth[t2]
                    for i in range(hs):
                        out_bth[i] += att_btht2 * value_t2[i]
                   
fn attention_backward( dinp:DTypePointer[dtype], dpreatt:DTypePointer[dtype], datt:DTypePointer[dtype],
                        dout:DTypePointer[dtype], inp:DTypePointer[dtype], att:DTypePointer[dtype],
                        B:Int32,T:Int32,C:Int32,NH:Int32):
    # inp/dinp are (B, T, 3C) Q,K,V
    # att/datt/dpreatt are (B, NH, T, T)
    # dout is (B, T, C)
    var C3:Int32 = C*3
    var hs:Int32 = C / NH # head size
    var scale:FLOAT = 1.0 * rsqrt(hs.cast[dtype]())

    for b in range(B):
        for t in range(T):
            for h in range(NH):
                var att_bth:DTypePointer[dtype] = att + b*NH*T*T + h*T*T + t*T
                var datt_bth:DTypePointer[dtype] = datt + b*NH*T*T + h*T*T + t*T
                var dpreatt_bth:DTypePointer[dtype] = dpreatt + b*NH*T*T + h*T*T + t*T
                var dquery_t:DTypePointer[dtype] = dinp + b * T * C3 + t * C3 + h * hs
                var query_t:DTypePointer[dtype] = inp + b * T * C3 + t * C3 + h * hs

                # backward pass 4, through the value accumulation
                var dout_bth:DTypePointer[dtype] = dout + b * T * C + t * C + h * hs
                for t2 in range(t+1):
                    var value_t2:DTypePointer[dtype] = inp + b * T * C3 + t2 * C3 + h * hs + C*2 # +C*2 because it's value
                    var dvalue_t2:DTypePointer[dtype] = dinp + b * T * C3 + t2 * C3 + h * hs + C*2
                    for i in range(hs):
                        # in the forward pass this was:
                        # out_bth[i] += att_bth[t2] * value_t2[i]
                        # so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i]
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i]
                    
                # backward pass 2 & 3, the softmax
                # note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in range(t+1):
                    for t3 in range(t+1):
                        var indicator:FLOAT = 0.0
                        if t2 == t3:
                            indicator = 1.0

                        var local_derivative:FLOAT = att_bth[t2] * (indicator - att_bth[t3])
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2]
                    
                # backward pass 1, the query @ key matmul
                for t2 in range(t+1):
                    var key_t2:DTypePointer[dtype] = inp + b * T * C3 + t2 * C3 + h * hs + C # +C because it's key
                    var dkey_t2:DTypePointer[dtype] = dinp + b * T * C3 + t2 * C3 + h * hs + C # +C because it's key
                    for i in range(hs):
                        # in the forward pass this was:
                        # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
                        # so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale
                    
fn gelu_forward( out:DTypePointer[dtype], inp:DTypePointer[dtype],N:Int32):
    var s:FLOAT = sqrt(2.0 / M_PI)
    for i in range(N):
        var x:FLOAT = inp[i]
        var cube:FLOAT = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1.0 + tanh(s * (x + cube)))
   
fn gelu_backward( dinp:DTypePointer[dtype], inp:DTypePointer[dtype], dout:DTypePointer[dtype],N:Int32):
    var s:FLOAT = sqrt(2.0 / M_PI)
    for i in range(N):
        var x:FLOAT = inp[i]
        var cube:FLOAT = 0.044715 * x * x * x
        var tanh_arg:FLOAT = s * (x + cube)
        var tanh_out:FLOAT = tanh(tanh_arg)
        var coshf_out:FLOAT = cosh(tanh_arg)
        var sech_out:FLOAT = 1.0 / (coshf_out * coshf_out)
        var local_grad:FLOAT = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s * (1.0 + 3.0 * 0.044715 * x * x)
        dinp[i] += local_grad * dout[i]

fn residual_forward( out:DTypePointer[dtype], inp1:DTypePointer[dtype], inp2:DTypePointer[dtype],N:Int32):
    for i in range(N):
        out[i] = inp1[i] + inp2[i]
   
fn residual_backward( dinp1:DTypePointer[dtype], dinp2:DTypePointer[dtype], dout:DTypePointer[dtype],N:Int32):
    for i in range(N):
        dinp1[i] += dout[i]
        dinp2[i] += dout[i]

fn softmax_forward( probs:DTypePointer[dtype], logits:DTypePointer[dtype],B:Int32,T:Int32,V:Int32):
    # output: probs are (B,T,V) of the probabilities
    # input: logits is (B,T,V) of the unnormalized log probabilities
    #pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            # probs <- softmax(logits)
            var logits_bt:DTypePointer[dtype] = logits + b * T * V + t * V
            var probs_bt:DTypePointer[dtype] = probs + b * T * V + t * V

            var maxval:FLOAT = -10000.0 # TODO something better
            for i in range(V):
                if (logits_bt[i] > maxval):
                    maxval = logits_bt[i]
                
            var sum:FLOAT = 0.0
            for i in range(V):
                probs_bt[i] = exp(logits_bt[i] - maxval)
                sum += probs_bt[i]
            
            for i in range(V):
                probs_bt[i] /= sum
            
fn crossentropy_forward( losses:DTypePointer[dtype],
                        probs:DTypePointer[dtype], targets:DTypePointer[dtype_int],
                          B:Int32,T:Int32,V:Int32):
    # output: losses is (B,T) of the individual losses at each position
    # input: probs are (B,T,V) of the probabilities
    # input: targets is (B,T) of integers giving the correct index in logits
    for b in range(B):
        for t in range(T):
            # loss = -log(probs[target])
            var probs_bt:DTypePointer[dtype] = probs + b * T * V + t * V
            var ix:Int32 = targets[b * T + t]
            losses[b * T + t] = -log(probs_bt[ix])

fn crossentropy_softmax_backward( dlogits:DTypePointer[dtype],
                        dlosses:DTypePointer[dtype], probs:DTypePointer[dtype], targets:DTypePointer[dtype_int],
                           B:Int32,T:Int32,V:Int32):
    # backwards through both softmax and crossentropy
    for b in range(B):
        for t in range(T):
            var dlogits_bt:DTypePointer[dtype] = dlogits + b * T * V + t * V
            var probs_bt:DTypePointer[dtype] = probs + b * T * V + t * V
            var dloss:FLOAT = dlosses[b * T + t]
            var ix:Int32 = targets[b * T + t]
            for i in range(V):
                var p:FLOAT = probs_bt[i]
                var indicator:FLOAT = 0.0
                if ix == i:
                    indicator = 1.0 
                dlogits_bt[i] += (p - indicator) * dloss
            
# ----------------------------------------------------------------------------
# GPT-2 model definition

# the parameters of the model

alias NUM_PARAMETER_TENSORS = 16

struct ParameterTensors:
    var params_memory: DTypePointer[dtype]

    var wte: DTypePointer[dtype]  # (V, C)
    var wpe: DTypePointer[dtype]  # (maxT, C)
    var ln1w: DTypePointer[dtype]  # (L, C)
    var ln1b: DTypePointer[dtype]  # (L, C)
    var qkvw: DTypePointer[dtype]  # (L, 3*C, C)
    var qkvb: DTypePointer[dtype]  # (L, 3*C)
    var attprojw: DTypePointer[dtype]  # (L, C, C)
    var attprojb: DTypePointer[dtype]  # (L, C)
    var ln2w: DTypePointer[dtype]  # (L, C)
    var ln2b: DTypePointer[dtype]  # (L, C)
    var fcw: DTypePointer[dtype]  # (L, 4*C, C)
    var fcb: DTypePointer[dtype]  # (L, 4*C)
    var fcprojw: DTypePointer[dtype]  # (L, C, 4*C)
    var fcprojb: DTypePointer[dtype]  # (L, C)
    var lnfw: DTypePointer[dtype]  # (C)
    var lnfb: DTypePointer[dtype]  # (C)

    fn __init__(
        inout self,
    ):
        self.params_memory = DTypePointer[dtype]()
       
        self.wte = DTypePointer[dtype]()
        self.wpe = DTypePointer[dtype]()
        self.ln1w = DTypePointer[dtype]()
        self.ln1b = DTypePointer[dtype]()
        self.qkvw = DTypePointer[dtype]()
        self.qkvb = DTypePointer[dtype]()
        self.attprojw = DTypePointer[dtype]()
        self.attprojb = DTypePointer[dtype]()
        self.ln2w = DTypePointer[dtype]()
        self.ln2b = DTypePointer[dtype]()
        self.fcw = DTypePointer[dtype]()
        self.fcb = DTypePointer[dtype]()
        self.fcprojw = DTypePointer[dtype]()
        self.fcprojb = DTypePointer[dtype]()
        self.lnfw = DTypePointer[dtype]()
        self.lnfb = DTypePointer[dtype]()

    fn alloc_and_point_parameters(inout self,param_sizes: InlinedFixedVector[type=Int32, size=NUM_PARAMETER_TENSORS]) -> DTypePointer[dtype]:

        var num_parameters: Int32 = 0
        var i: Int

        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += param_sizes[i]

        # malloc all parameters all at once
        self.params_memory = DTypePointer[dtype]().alloc(num_parameters.to_int())
        # assign all the tensors

        var ptrs = List(
            Pointer.address_of(self.wte),
            Pointer.address_of(self.wpe),
            Pointer.address_of(self.ln1w),
            Pointer.address_of(self.ln1b),
            Pointer.address_of(self.qkvw),
            Pointer.address_of(self.qkvb),
            Pointer.address_of(self.attprojw),
            Pointer.address_of(self.attprojb),
            Pointer.address_of(self.ln2w),
            Pointer.address_of(self.ln2b),
            Pointer.address_of(self.fcw),
            Pointer.address_of(self.fcb),
            Pointer.address_of(self.fcprojw),
            Pointer.address_of(self.fcprojb),
            Pointer.address_of(self.lnfw),
            Pointer.address_of(self.lnfb),
        )

        var params_memory_iterator: DTypePointer[dtype] = self.params_memory

        for i in range(NUM_PARAMETER_TENSORS):
            ptrs[i][] = params_memory_iterator
            params_memory_iterator += param_sizes[i]

        return self.params_memory


alias NUM_ACTIVATION_TENSORS = 23

@register_passable("trivial")
struct ActivationTensors:
    var encoded: DTypePointer[dtype]  # (B, T, C)
    var ln1: DTypePointer[dtype]  # (L, B, T, C)
    var ln1_mean: DTypePointer[dtype]  # (L, B, T)
    var ln1_rstd: DTypePointer[dtype]  # (L, B, T)
    var qkv: DTypePointer[dtype]  # (L, B, T, 3*C)
    var atty: DTypePointer[dtype]  # (L, B, T, C)
    var preatt: DTypePointer[dtype]  # (L, B, NH, T, T)
    var att: DTypePointer[dtype]  # (L, B, NH, T, T)
    var attproj: DTypePointer[dtype]  # (L, B, T, C)
    var residual2: DTypePointer[dtype]  # (L, B, T, C)
    var ln2: DTypePointer[dtype]  # (L, B, T, C)
    var ln2_mean: DTypePointer[dtype]  # (L, B, T)
    var ln2_rstd: DTypePointer[dtype]  # (L, B, T)
    var fch: DTypePointer[dtype]  # (L, B, T, 4*C)
    var fch_gelu: DTypePointer[dtype]  # (L, B, T, 4*C)
    var fcproj: DTypePointer[dtype]  # (L, B, T, C)
    var residual3: DTypePointer[dtype]  # (L, B, T, C)
    var lnf: DTypePointer[dtype]  # (B, T, C)
    var lnf_mean: DTypePointer[dtype]  # (B, T)
    var lnf_rstd: DTypePointer[dtype]  # (B, T)
    var logits: DTypePointer[dtype]  # (B, T, V)
    var probs: DTypePointer[dtype]  # (B, T, V)
    var losses: DTypePointer[dtype]  # (B, T)

    fn __init__(
        inout self,
        
    ):
        self.encoded = DTypePointer[dtype]()
        self.ln1 = DTypePointer[dtype]()
        self.ln1_mean = DTypePointer[dtype]()
        self.ln1_rstd = DTypePointer[dtype]()
        self.qkv = DTypePointer[dtype]()
        self.atty = DTypePointer[dtype]()
        self.preatt = DTypePointer[dtype]()
        self.att = DTypePointer[dtype]()
        self.attproj = DTypePointer[dtype]()
        self.residual2 = DTypePointer[dtype]()
        self.ln2 = DTypePointer[dtype]()
        self.ln2_mean = DTypePointer[dtype]()
        self.ln2_rstd = DTypePointer[dtype]()
        self.fch = DTypePointer[dtype]()
        self.fch_gelu = DTypePointer[dtype]()
        self.fcproj = DTypePointer[dtype]()
        self.residual3 = DTypePointer[dtype]()
        self.lnf = DTypePointer[dtype]()
        self.lnf_mean = DTypePointer[dtype]()
        self.lnf_rstd = DTypePointer[dtype]()
        self.logits = DTypePointer[dtype]()
        self.probs = DTypePointer[dtype]()
        self.losses = DTypePointer[dtype]()
    
    fn alloc_and_point_activations(inout self,act_sizes: InlinedFixedVector[type=Int32, size=NUM_ACTIVATION_TENSORS]) -> DTypePointer[dtype]:

        var ptrs = List(
            Pointer.address_of(self.encoded),
            Pointer.address_of(self.ln1),
            Pointer.address_of(self.ln1_mean),
            Pointer.address_of(self.ln1_rstd),
            Pointer.address_of(self.qkv),
            Pointer.address_of(self.atty),
            Pointer.address_of(self.preatt),
            Pointer.address_of(self.att),
            Pointer.address_of(self.attproj),
            Pointer.address_of(self.residual2),
            Pointer.address_of(self.ln2),
            Pointer.address_of(self.ln2_mean),
            Pointer.address_of(self.ln2_rstd),
            Pointer.address_of(self.fch),
            Pointer.address_of(self.fch_gelu),
            Pointer.address_of(self.fcproj),
            Pointer.address_of(self.residual3),
            Pointer.address_of(self.lnf),
            Pointer.address_of(self.lnf_mean),
            Pointer.address_of(self.lnf_rstd),
            Pointer.address_of(self.logits),
            Pointer.address_of(self.probs),
            Pointer.address_of(self.losses),
        )

        var num_activations: Int32 = 0

        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += act_sizes[i]

        var acts_memory = DTypePointer[dtype]().alloc(num_activations.to_int())

        var acts_memory_iterator: DTypePointer[dtype] = acts_memory
        for i in range(NUM_ACTIVATION_TENSORS):
            ptrs[i][] = acts_memory_iterator
            acts_memory_iterator += act_sizes[i]
        
        return acts_memory

@value
struct GPT2Config:
    var max_seq_len: Int32  # max sequence length, e.g. 1024
    var vocab_size: Int32  # vocab size, e.g. 50257
    var num_layers: Int32  # number of layers, e.g. 12
    var num_heads: Int32  # number of heads in attention, e.g. 12
    var channels: Int32  # number of channels, e.g. 768

struct GPT2:
    var config: GPT2Config
    # the weights of the model, and their sizes
    var params: ParameterTensors
    var param_sizes: InlinedFixedVector[type=Int32, size=NUM_PARAMETER_TENSORS]
    var params_memory: DTypePointer[dtype]
    var num_parameters: Int32
    # gradients of the weights
    var grads: ParameterTensors
    var grads_memory: DTypePointer[dtype]
    # buffers for the AdamW optimizer
    var m_memory: DTypePointer[dtype]
    var v_memory: DTypePointer[dtype]
    # the activations of the model, and their sizes
    var acts: ActivationTensors
    var act_sizes: InlinedFixedVector[type=Int32, size=NUM_ACTIVATION_TENSORS]
    var acts_memory: DTypePointer[dtype]
    var num_activations: Int32
    # gradients of the activations
    var grads_acts: ActivationTensors
    var grads_acts_memory: DTypePointer[dtype]
    # other run state configuration
    var batch_size: INT  # the batch size (B) of current forward pass
    var seq_len: INT  # the sequence length (T) of current forward pass
    var inputs: DTypePointer[dtype_int]  # the input tokens for the current forward pass
    var targets: DTypePointer[
        dtype_int
    ]  # the target tokens for the current forward pass
    var mean_loss: FLOAT  # after a forward pass with targets, will be populated with the mean loss

    var checkpoint_path: StringRef

    fn __init__(inout self, checkpoint_path: StringRef) raises:
        self.checkpoint_path = checkpoint_path

        self.param_sizes = InlinedFixedVector[type=Int32, size=NUM_PARAMETER_TENSORS](
            NUM_PARAMETER_TENSORS
        )
        self.act_sizes = InlinedFixedVector[type=Int32, size=NUM_ACTIVATION_TENSORS](
            NUM_ACTIVATION_TENSORS
        )

        var model_file = open(checkpoint_path, "r")

        var bytes_of_config_params = 256 * sizeof[DType.int32]()
        # config_data_raw id Tensor[DType.int8] with bytes_of_config_params elements
        var config_data_raw = model_file.read(bytes_of_config_params)
       
        var model_header = config_data_raw._steal_ptr().bitcast[DType.int32]()
       
        if model_header[0] != 20240326:
            print("Bad magic model file")
            # EXIT_1
        if model_header[1] != 1:
            print("Bad version in model file")
            # EXIT_1

        # read in hyperparameters

        self.config = GPT2Config(
            model_header[2].cast[DType.int32](),
            model_header[3].cast[DType.int32](),
            model_header[4].cast[DType.int32](),
            model_header[5].cast[DType.int32](),
            model_header[6].cast[DType.int32](),
        )

        var maxT: Int32 = self.config.max_seq_len
        var V: Int32 = self.config.vocab_size
        var L: Int32 = self.config.num_layers
        var NH: Int32 = self.config.num_heads
        var C: Int32 = self.config.channels

        print("[GPT-2]")
        print("max_seq_len:", self.config.max_seq_len)
        print("vocab_size:", self.config.vocab_size)
        print("num_layers:", self.config.num_layers)
        print("num_heads:", self.config.num_heads)
        print("channels:", self.config.channels)

        # allocate space for all the parameters and read them in
        self.param_sizes[0] = V * C
        self.param_sizes[1] = maxT * C
        self.param_sizes[2] = L * C
        self.param_sizes[3] = L * C
        self.param_sizes[4] = L * (3 * C) * C
        self.param_sizes[5] = L * (3 * C)
        self.param_sizes[6] = L * C * C
        self.param_sizes[7] = L * C
        self.param_sizes[8] = L * C
        self.param_sizes[9] = L * C
        self.param_sizes[10] = L * (4 * C) * C
        self.param_sizes[11] = L * (4 * C)
        self.param_sizes[12] = L * C * (4 * C)
        self.param_sizes[13] = L * C
        self.param_sizes[14] = C
        self.param_sizes[15] = C

        # cound the number of paramaters
        var num_parameters: Int32 = 0

        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += self.param_sizes[i]

        print("num_parameters:", num_parameters)
        self.num_parameters = num_parameters

        # read in all the parameters from file
        self.params = ParameterTensors()
        self.params_memory = self.params.alloc_and_point_parameters(self.param_sizes)
        
        var data_raw = model_file.read( (num_parameters * SIZEOF_FLOAT).to_int())
        
        model_file.close()

        var float32_ptr= data_raw._steal_ptr().bitcast[DType.float32]()
        memcpy(dest=self.params_memory,src=float32_ptr,count=(num_parameters).to_int())

        # other inits
        self.acts = ActivationTensors()
        self.num_activations = 0 # for now

        self.acts_memory = NULL
        self.grads_memory = NULL
        self.m_memory = NULL
        self.v_memory = NULL
        self.grads_acts_memory = NULL
        self.inputs = NULL_INT
        self.targets = NULL_INT
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0 will designate no loss

        self.grads = ParameterTensors()
        self.grads_acts = ActivationTensors()


fn gpt2_forward(inout model:GPT2, inputs:DTypePointer[dtype_int], targets:DTypePointer[dtype_int],B:Int32,T:Int32):
    # targets are optional and could be NULL

    # ensure the model was initialized or error out
    if (model.params_memory == NULL):
        print("Error: model was not initialized properly.")
        
    # convenience parameters
    var V:Int32 = model.config.vocab_size
    var L:Int32 = model.config.num_layers
    var NH:Int32 = model.config.num_heads
    var C:Int32 = model.config.channels

    # allocate space for all the activations if needed (done here, lazily)
    if(model.acts_memory == NULL):
        # record the current B,T as well
        model.batch_size = B
        model.seq_len = T

        # and now allocate the space
        model.act_sizes[0] = B * T * C
        model.act_sizes[1] = L * B * T * C
        model.act_sizes[2] = L * B * T
        model.act_sizes[3] = L * B * T
        model.act_sizes[4] = L * B * T * 3*C
        model.act_sizes[5] = L * B * T * C
        model.act_sizes[6] = L * B * NH * T * T
        model.act_sizes[7] = L * B * NH * T * T
        model.act_sizes[8] = L * B * T * C
        model.act_sizes[9] = L * B * T * C
        model.act_sizes[10] = L * B * T * C
        model.act_sizes[11] = L * B * T
        model.act_sizes[12] = L * B * T
        model.act_sizes[13] = L * B * T * 4*C
        model.act_sizes[14] = L * B * T * 4*C
        model.act_sizes[15] = L * B * T * C
        model.act_sizes[16] = L * B * T * C
        model.act_sizes[17] = B * T * C
        model.act_sizes[18] = B * T
        model.act_sizes[19] = B * T
        model.act_sizes[20] = B * T * V
        model.act_sizes[21] = B * T * V
        model.act_sizes[22] = B * T

        var num_activations:Int32 = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += model.act_sizes[i]
        
        print("num_activations:", num_activations)
    
        model.acts_memory = model.acts.alloc_and_point_activations(model.act_sizes)     
        model.num_activations = num_activations
        # also create memory for caching inputs and targets
       
        model.inputs = DTypePointer[dtype_int]().alloc((B * T).to_int() )
        model.targets = DTypePointer[dtype_int]().alloc((B * T).to_int() )
    
    else:
        # validate B,T is no larger than what was previously allocated
        # in principle, we could re-allocate a larger chunk of memory, for now we just error out
        if B > model.batch_size.to_int() or T > model.seq_len.to_int():
            print("Error: batch size or sequence length is inadequately large")
            #print("Model: B=%d T=%d, Desired: B=%d T=%d\n", model.batch_size, model.seq_len, B, T)
            
   
    # cache the inputs/targets
    memcpy(model.inputs, inputs, (B * T).to_int())

    if targets != NULL_INT:
        memcpy(model.targets, targets, (B * T).to_int())
    
    # forward pass
    
    var residual:DTypePointer[dtype]
    encoder_forward(model.acts.encoded, inputs, model.params.wte, model.params.wpe, B, T, C) # encoding goes into residual[0]
    
    for l in range(L):
        
        residual =  model.acts.residual3 + (l-1) * B * T * C

        if  l == 0:
            residual = model.acts.encoded

        # get the pointers of the weights for this layer
        var l_ln1w:DTypePointer[dtype] = model.params.ln1w + l * C
        var l_ln1b:DTypePointer[dtype] = model.params.ln1b + l * C
        var l_qkvw:DTypePointer[dtype] = model.params.qkvw + l * 3*C * C
        var l_qkvb:DTypePointer[dtype] = model.params.qkvb + l * 3*C
        var l_attprojw:DTypePointer[dtype] = model.params.attprojw + l * C * C
        var l_attprojb:DTypePointer[dtype] = model.params.attprojb + l * C
        var l_ln2w:DTypePointer[dtype] = model.params.ln2w + l * C
        var l_ln2b:DTypePointer[dtype] = model.params.ln2b + l * C
        var l_fcw:DTypePointer[dtype] = model.params.fcw + l * 4*C * C
        var l_fcb:DTypePointer[dtype] = model.params.fcb + l * 4*C
        var l_fcprojw:DTypePointer[dtype] = model.params.fcprojw + l * C * 4*C
        var l_fcprojb:DTypePointer[dtype] = model.params.fcprojb + l * C

        # get the pointers of the activations for this layer
        var l_ln1:DTypePointer[dtype] = model.acts.ln1 + l * B * T * C
        var l_ln1_mean:DTypePointer[dtype] = model.acts.ln1_mean + l * B * T
        var l_ln1_rstd:DTypePointer[dtype] = model.acts.ln1_rstd + l * B * T
        var l_qkv:DTypePointer[dtype] = model.acts.qkv + l * B * T * 3*C
        var l_atty:DTypePointer[dtype] = model.acts.atty + l * B * T * C
        var l_preatt:DTypePointer[dtype] = model.acts.preatt + l * B * NH * T * T
        var l_att:DTypePointer[dtype] = model.acts.att + l * B * NH * T * T
        var l_attproj:DTypePointer[dtype] = model.acts.attproj + l * B * T * C
        var l_residual2:DTypePointer[dtype] = model.acts.residual2 + l * B * T * C
        var l_ln2:DTypePointer[dtype] = model.acts.ln2 + l * B * T * C
        var l_ln2_mean:DTypePointer[dtype] = model.acts.ln2_mean + l * B * T
        var l_ln2_rstd:DTypePointer[dtype] = model.acts.ln2_rstd + l * B * T
        var l_fch:DTypePointer[dtype] = model.acts.fch + l * B * T * 4*C
        var l_fch_gelu:DTypePointer[dtype] = model.acts.fch_gelu + l * B * T * 4*C
        var l_fcproj:DTypePointer[dtype] = model.acts.fcproj + l * B * T * C
        var l_residual3:DTypePointer[dtype] = model.acts.residual3 + l * B * T * C

        # now do the forward pass

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C)
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH)
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
        residual_forward(l_residual2, residual, l_attproj, B*T*C)
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C)
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C)
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C)
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C)
   
    residual = model.acts.residual3 + (L-1) * B * T * C # last residual is in residual3
    layernorm_forward(model.acts.lnf, model.acts.lnf_mean, model.acts.lnf_rstd, residual, model.params.lnfw, model.params.lnfb, B, T, C)
    matmul_forward(model.acts.logits, model.acts.lnf, model.params.wte, NULL, B, T, C, V)
    softmax_forward(model.acts.probs, model.acts.logits, B, T, V)
   
    # also forward the cross-entropy loss function if we have the targets
    if targets != NULL_INT:
        crossentropy_forward(model.acts.losses, model.acts.probs, targets, B, T, V)
        # for convenience also evaluate the mean loss
        var mean_loss:FLOAT = 0.0
        for i in range(B*T):
             mean_loss += model.acts.losses[i] 
        mean_loss /= (B*T).to_int()
        model.mean_loss = mean_loss
    else:
        # if we don't have targets, we don't have a loss
        model.mean_loss = -1.0
    
fn gpt2_zero_grad(inout model:GPT2):
    if(model.grads_memory != NULL): 
        memset_zero(model.grads_memory, model.num_parameters.to_int()) 

    if(model.grads_acts_memory != NULL): 
        memset_zero(model.grads_acts_memory, model.num_activations.to_int()) 

fn gpt2_backward(inout model:GPT2):

    # double check we forwarded previously, with targets
    if (model.mean_loss == -1.0):
        print("Error: must forward with targets before backward\n")
        
    # lazily allocate the memory for gradients of the weights and activations, if needed
    if (model.grads_memory == NULL):
        model.grads_memory = model.grads.alloc_and_point_parameters(model.param_sizes)
        model.grads_acts_memory = model.grads_acts.alloc_and_point_activations( model.act_sizes)
        gpt2_zero_grad(model)
    
    # convenience shortcuts
    var B:Int32 = model.batch_size
    var T:Int32 = model.seq_len
    var V:Int32 = model.config.vocab_size
    var L:Int32 = model.config.num_layers
    var NH:Int32 = model.config.num_heads
    var C:Int32 = model.config.channels

    # backward pass

    # we kick off the chain by filling in dlosses with 1.0/(B*T), to get the mean loss
    var dloss_mean:FLOAT = 1.0 / (B*T).to_int()
    
    for i in range(B*T):
        model.grads_acts.losses[i] = dloss_mean 

    crossentropy_softmax_backward(model.grads_acts.logits, model.grads_acts.losses, model.acts.probs, model.targets, B, T, V)
    matmul_backward(model.grads_acts.lnf, model.grads.wte, NULL, model.grads_acts.logits, model.acts.lnf, model.params.wte, B, T, C, V)
    var residual:DTypePointer[dtype] = model.acts.residual3 + (L-1) * B * T * C # last layer's residual
    var dresidual:DTypePointer[dtype] = model.grads_acts.residual3 + (L-1) * B * T * C # write to last layer's residual
    layernorm_backward(dresidual, model.grads.lnfw, model.grads.lnfb, model.grads_acts.lnf, residual, model.params.lnfw, model.acts.lnf_mean, model.acts.lnf_rstd, B, T, C)

    for l in range(L-1,-1,-1):

        var residual = model.acts.encoded
        var dresidual =  model.grads_acts.encoded
        if l != 0: 
            residual = model.acts.residual3 + (l-1) * B * T * C
            dresidual =  model.grads_acts.residual3 + (l-1) * B * T * C
       
        # get the pointers of the weights for this layer
        var l_ln1w:DTypePointer[dtype] = model.params.ln1w + l * C
        var l_qkvw:DTypePointer[dtype] = model.params.qkvw + l * 3*C * C
        var l_attprojw:DTypePointer[dtype] = model.params.attprojw + l * C * C
        var l_ln2w:DTypePointer[dtype] = model.params.ln2w + l * C
        var l_fcw:DTypePointer[dtype] = model.params.fcw + l * 4*C * C
        var l_fcprojw:DTypePointer[dtype] = model.params.fcprojw + l * C * 4*C
        # get the pointers of the gradients of the weights for this layer
        var dl_ln1w:DTypePointer[dtype] = model.grads.ln1w + l * C
        var dl_ln1b:DTypePointer[dtype] = model.grads.ln1b + l * C
        var dl_qkvw:DTypePointer[dtype] = model.grads.qkvw + l * 3*C * C
        var dl_qkvb:DTypePointer[dtype] = model.grads.qkvb + l * 3*C
        var dl_attprojw:DTypePointer[dtype] = model.grads.attprojw + l * C * C
        var dl_attprojb:DTypePointer[dtype] = model.grads.attprojb + l * C
        var dl_ln2w:DTypePointer[dtype] = model.grads.ln2w + l * C
        var dl_ln2b:DTypePointer[dtype] = model.grads.ln2b + l * C
        var dl_fcw:DTypePointer[dtype] = model.grads.fcw + l * 4*C * C
        var dl_fcb:DTypePointer[dtype] = model.grads.fcb + l * 4*C
        var dl_fcprojw:DTypePointer[dtype] = model.grads.fcprojw + l * C * 4*C
        var dl_fcprojb:DTypePointer[dtype] = model.grads.fcprojb + l * C
        # get the pointers of the activations for this layer
        var l_ln1:DTypePointer[dtype] = model.acts.ln1 + l * B * T * C
        var l_ln1_mean:DTypePointer[dtype] = model.acts.ln1_mean + l * B * T
        var l_ln1_rstd:DTypePointer[dtype] = model.acts.ln1_rstd + l * B * T
        var l_qkv:DTypePointer[dtype] = model.acts.qkv + l * B * T * 3*C
        var l_atty:DTypePointer[dtype] = model.acts.atty + l * B * T * C
        var l_att:DTypePointer[dtype] = model.acts.att + l * B * NH * T * T
        var l_residual2:DTypePointer[dtype] = model.acts.residual2 + l * B * T * C
        var l_ln2:DTypePointer[dtype] = model.acts.ln2 + l * B * T * C
        var l_ln2_mean:DTypePointer[dtype] = model.acts.ln2_mean + l * B * T
        var l_ln2_rstd:DTypePointer[dtype] = model.acts.ln2_rstd + l * B * T
        var l_fch:DTypePointer[dtype] = model.acts.fch + l * B * T * 4*C
        var l_fch_gelu:DTypePointer[dtype] = model.acts.fch_gelu + l * B * T * 4*C
        # get the pointers of the gradients of the activations for this layer
        var dl_ln1:DTypePointer[dtype] = model.grads_acts.ln1 + l * B * T * C
        var dl_qkv:DTypePointer[dtype] = model.grads_acts.qkv + l * B * T * 3*C
        var dl_atty:DTypePointer[dtype] = model.grads_acts.atty + l * B * T * C
        var dl_preatt:DTypePointer[dtype] = model.grads_acts.preatt + l * B * NH * T * T
        var dl_att:DTypePointer[dtype] = model.grads_acts.att + l * B * NH * T * T
        var dl_attproj:DTypePointer[dtype] = model.grads_acts.attproj + l * B * T * C
        var dl_residual2:DTypePointer[dtype] = model.grads_acts.residual2 + l * B * T * C
        var dl_ln2:DTypePointer[dtype] = model.grads_acts.ln2 + l * B * T * C
        var dl_fch:DTypePointer[dtype] = model.grads_acts.fch + l * B * T * 4*C
        var dl_fch_gelu:DTypePointer[dtype] = model.grads_acts.fch_gelu + l * B * T * 4*C
        var dl_fcproj:DTypePointer[dtype] = model.grads_acts.fcproj + l * B * T * C
        var dl_residual3:DTypePointer[dtype] = model.grads_acts.residual3 + l * B * T * C

        # backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C)
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C)
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C)
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C)
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C)
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C)
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH)
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C)
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C)
    
    encoder_backward(model.grads.wte, model.grads.wpe, model.grads_acts.encoded, model.inputs, B, T, C)


fn gpt2_update(inout model:GPT2, learning_rate:FLOAT, beta1:FLOAT, beta2:FLOAT, eps:FLOAT, weight_decay:FLOAT,t:Int32):
    # reference: https:#pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    # lazily allocate the memory for m_memory and v_memory
    if (model.m_memory == NULL):
        model.m_memory = DTypePointer[dtype]().alloc(model.num_parameters.to_int())
        model.v_memory = DTypePointer[dtype]().alloc(model.num_parameters.to_int())

        memset_zero(model.m_memory,model.num_parameters.to_int())
        memset_zero(model.v_memory,model.num_parameters.to_int())

    for i in range(model.num_parameters):
        var param:FLOAT = model.params_memory[i]
        var grad:FLOAT = model.grads_memory[i]

        # update the first moment (momentum)
        var m:FLOAT = beta1 * model.m_memory[i] + (1.0 - beta1) * grad
        # update the second moment (RMSprop)
        var v:FLOAT = beta2 * model.v_memory[i] + (1.0 - beta2) * grad * grad
        # bias-correct both moments
        var m_hat:FLOAT = m / (1.0 - pow(beta1, t))
        var v_hat:FLOAT = v / (1.0 - pow(beta2, t))

        # update
        model.m_memory[i] = m
        model.v_memory[i] = v
        model.params_memory[i] -= learning_rate * (m_hat / (rsqrt(v_hat) + eps) + weight_decay * param)
    

fn gpt2_free(inout model:GPT2):
    model.params_memory.free()
    model.grads_memory.free()
    model.m_memory.free()
    model.v_memory.free()
    model.acts_memory.free()
    model.grads_acts_memory.free()
    model.inputs.free()
    model.targets.free()

#ifndef TESTING
# if we are TESTING (see test_gpt2.c), we'll skip the maiN:Int32 below

# ----------------------------------------------------------------------------
# data loader lite
# returns random batches of data from a file of integers

struct DataLoader:
    # hyperparameters
    var B:Int32
    var T:Int32
    # input handling and its state
    var filename:StringRef
    var tokens_file:FileHandle
    var file_size:Int32
    var current_position:Int32
    # output memory
    var batch:DTypePointer[dtype_int]
    var inputs:DTypePointer[dtype_int]
    var targets:DTypePointer[dtype_int]
    # convenience variables
    var num_batches:Int32

    fn __init__(inout self):
        self.B = 0
        self.T = 0
        self.filename = ""
        self.tokens_file=FileHandle()
        self.file_size = 0
        self.current_position = 0
        self.batch = DTypePointer[dtype_int]()
        self.inputs = DTypePointer[dtype_int]()
        self.targets = DTypePointer[dtype_int]()
        self.num_batches = 0

fn dataloader_init(inout loader:DataLoader,filename:StringRef,B:Int32,T:Int32) raises:
    loader.B = B
    loader.T = T

    # open the input file for reading
    try:
        loader.tokens_file = open(filename, "rb")
    except e:
        print("Error opening tokens file",e)
       
    # determine the file size
    var _os = Python.import_module("os")  
    loader.file_size = int(_os.path.getsize(filename))

    if (loader.file_size < (B * T + 1).to_int() * 4):
        print("Error: file size is too small for the batch size and sequence length\n")
         
    loader.current_position = 0 # start at the beginning

    # allocate space for B*T + 1 integers to store the inputs and targets loader.batch = (int*) malloc((B * T + 1) * sizeof(int))
    
    loader.batch = DTypePointer[dtype_int]().alloc((B * T + 1).to_int())
    loader.inputs = loader.batch
    loader.targets = loader.batch + 1 # targets are shifted by one
    loader.num_batches = loader.file_size.to_int() / (B * T * SIZEOF_INT).to_int()

fn dataloader_reset(inout loader:DataLoader):
    loader.current_position = 0

fn dataloader_next_batch(inout loader:DataLoader) raises:
    var B:Int32 = loader.B
    var T:Int32 = loader.T

    # if we are at the end of the file, loop back to the beginning
    if loader.current_position + ((B*T+1) * SIZEOF_INT).to_int() > loader.file_size:
        loader.current_position = 0
        
    # read the B*T+1 integers from the file into batch
    _ = loader.tokens_file.seek( loader.current_position.to_int())
    
    # config_data_raw id Tensor[DType.int8] with bytes_of_config_params elements
    var data_raw = loader.tokens_file.read(((B*T+1) * SIZEOF_INT).to_int())
    var int32_ptr= data_raw._steal_ptr().bitcast[DType.int32]()

    memcpy(dest=loader.batch,src=int32_ptr,count=(B*T+1).to_int())
       
    # advance the current position by B*T integers
    loader.current_position += B*T * SIZEOF_INT

fn dataloader_free(inout loader:DataLoader) raises:
    loader.tokens_file.close()
    loader.batch.free()

# ----------------------------------------------------------------------------
# sampler

fn random_u32(inout state:Int32) -> Int32:
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return ((state * RU32_HEX) >> 32).cast[DType.int32]();

fn random_f32(inout state:Int32) -> Float32:    
    return (random_u32(state) >> 8).cast[DType.float32]() / RF32_DIV

fn sample_mult( probabilities:DTypePointer[dtype],n:Int32, coin:FLOAT) -> Int32:
    # sample index from probabilities (they must sum to 1!)
    # coin is a random number in [0, 1), usually from random_f32()
    var cdf:FLOAT = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if (coin < cdf):
            return i
    return n - 1

# ----------------------------------------------------------------------------
# main training loop

fn main() raises:

    # build the GPT-2 model from a checkpoint
    var model = GPT2("../../gpt2_124M.bin")

    # build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    var tiny_stories_train:StringRef = "../../data/TinyStories_train.bin"
    var tiny_stories_val:StringRef = "../../data/TinyStories_val.bin"
    var tiny_shakespeare_train:StringRef = "../../data/tiny_shakespeare_train.bin"
    var tiny_shakespeare_val:StringRef = "../../data/tiny_shakespeare_val.bin"
    ##var train_tokens:StringRef = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train
    ##var val_tokens:StringRef = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val
    var train_tokens:StringRef = tiny_shakespeare_train
    var val_tokens:StringRef = tiny_shakespeare_val

    var B:Int32 = 4
    var T:Int32 = 64
    var train_loader = DataLoader() 
    dataloader_init(train_loader, train_tokens, B, T)
    print("train dataset num_batches:", train_loader.num_batches)
    var val_loader = DataLoader() 
    dataloader_init(val_loader, val_tokens, B, T)
    print("val dataset num_batches:", val_loader.num_batches)
    var val_num_batches:Int32 = 10

    # some memory for generating samples from the model
    var rng_state:Int32 = 1337
    var gen_max_length:Int32 = 64
    var gen_tokens = DTypePointer[dtype_int]().alloc(gen_max_length.to_int())

    # train

    var elapsed_time_ms_total = 0.0
     
    for step in range(41):
       
        # once in a while estimate the validation loss
        if step % 10 == 20:
            var val_loss:FLOAT = 0.0
            dataloader_reset(val_loader)
            for i in range(val_num_batches):
                dataloader_next_batch(val_loader)
                gpt2_forward(model, val_loader.inputs, val_loader.targets, B, T)
                val_loss += model.mean_loss
            
            val_loss /= val_num_batches.to_int()
            print("val loss", val_loss)
        
        # once in a while do model inference to prgenerated INT32 text
        if step > 0 and step % 20 == 0:
            gen_tokens[0] = GPT2_EOT # the GPT-2 EOT token kicks off the generation
            
            for t in range(1,gen_max_length):
                # note that inference is wasteful here because
                # for each t, we re-compute all activations between 0 and t
                # leaving this alone because you want separate code for inference anyway
                # the inference here is just for sanity checking purposes
                gpt2_forward(model, gen_tokens, NULL_INT, 1, t)
                var probs:DTypePointer[dtype] = model.acts.probs + (t-1) * model.config.vocab_size
                var coin:FLOAT = random_f32(rng_state)
                var next_token:Int32 = sample_mult(probs, model.config.vocab_size, coin)
                gen_tokens[t] = next_token
            
            print("generated: ",end="")
            for t in range(gen_max_length):
                print( gen_tokens[t],end=" ")
            print("")
        
        # do a training step
     
        var start_time = now()

        dataloader_next_batch(train_loader)
        gpt2_forward(model, train_loader.inputs, train_loader.targets, B, T)
        gpt2_zero_grad(model)
        gpt2_backward(model)
        gpt2_update(model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step+1)
          
        var elapsed_time_ms = (now() - start_time)/1_000_000.
    
        elapsed_time_ms_total += elapsed_time_ms
        
        print("step " + str(step) + ": train loss " + str(model.mean_loss) + " (took " + int(elapsed_time_ms) + " ms, average: " + int(elapsed_time_ms_total/(step+1)) + " ms)")
    
    # free
    dataloader_free(train_loader)
    dataloader_free(val_loader)
    gpt2_free(model)
