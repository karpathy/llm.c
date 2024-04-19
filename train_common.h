
#ifndef _TRAIN_COMMON_H_INCLUDED
#define _TRAIN_COMMON_H_INCLUDED


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
