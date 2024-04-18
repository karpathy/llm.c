
#ifndef _TRAIN_COMMON_H_INCLUDED
#define _TRAIN_COMMON_H_INCLUDED

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

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;



// ----------------------------------------------------------------------------
// check function helpers

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

#endif // _TRAIN_COMMON_H_INCLUDED
