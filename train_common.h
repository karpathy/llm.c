

#ifndef TRAIN_CUDA
#define TRAIN_CPU
#endif

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the GPT-2 end-of-text token id
#define GPT2_EOT 50256

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;


#ifdef TRAIN_CPU
#define NUM_ACTIVATION_TENSORS 23
#endif

#ifdef TRAIN_CUDA
#define NUM_ACTIVATION_TENSORS 25
#endif

typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
#ifdef TRAIN_CUDA
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    float* v_accum; // (L, B, T, C)
#endif
} ActivationTensors;

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
#ifdef TRAIN_CUDA
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
#endif
} GPT2;



// ----------------------------------------------------------------------------
// check function helpers

void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T);


// ----------------------------------------------------------------------------
// check function helpers

#ifdef TRAIN_CUDA
#define cudaCheck(err) (cudaDoCheck(err, __FILE__, __LINE__))
#define cublasCheck(status) { cublasDoCheck((status), __FILE__, __LINE__); }

void cudaDoCheck(cudaError_t error, const char *file, int line);
void cublasDoCheck(cublasStatus_t status, const char *file, int line);
#endif

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

FILE *fopen_check(const char *path, const char *mode, const char *file, int line);
void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);
void fclose_check(FILE *fp, const char *file, int line);
void *malloc_check(size_t size, const char *file, int line);


// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample_mult(float* probabilities, int n, float coin);

// ----------------------------------------------------------------------------
// tokenizer

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
} Tokenizer;


void safe_printf(const char *piece);
void tokenizer_init(Tokenizer *tokenizer, const char *filename);
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id);
void tokenizer_free(Tokenizer *tokenizer);

void gen_forward(GPT2 *modelp, Tokenizer *tokenizerp, int genT, int B, int T );
