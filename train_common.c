
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#ifdef OMP
#include <omp.h>
#endif

#ifdef TRAIN_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

#include "train_common.h"



#ifdef TRAIN_CUDA

// CUDA error checking
void cudaDoCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

// cuBLAS error checking
void cublasDoCheck(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}

#endif


// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose with fopenCheck, freadCheck, fcloseCheck

FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return fp;
}

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}


// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}


// ----------------------------------------------------------------------------
// random and sampler

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}


// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    assert(header[1] == 1);
    tokenizer->vocab_size = header[2];
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %d!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}


// ----------------------------------------------------------------------------
// Occasional gen_forward step

void gen_forward(GPT2 *modelp, Tokenizer *tokenizerp, int genT, int B, int T  ) {

    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
#ifdef TRAIN_CUDA
    float* cpu_probs = (float*)mallocCheck(modelp->config.vocab_size * sizeof(float));
#endif

	static unsigned long long rng_state = 1337;

	// fill up gen_tokens with the GPT2_EOT, which kicks off the generation
	for(int i = 0; i < B * T; ++i) {
		gen_tokens[i] = GPT2_EOT;
	}
	// now sample from the model autoregressively
	printf("generating:\n---\n");
	for (int t = 1; t < genT; t++) {
		// note that inference is very wasteful here because for each token
		// we re-calculate the forward pass for all of (B,T) positions from scratch
		// but the inference here is just for sanity checking anyway
		// and we can maybe optimize a bit more later, with careful tests
		gpt2_forward(modelp, gen_tokens, NULL, B, T);
		// furthermore, below we're only using b=0 (i.e. the first row) of all B rows
		// we're in principle running B "inference streams" in parallel here
		// only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
		// get the V-dimensional vector probs[0, t-1, :]

		float coin = random_f32(&rng_state);
		float* model_probs = modelp->acts.probs + (t-1) * modelp->config.vocab_size;
#ifdef TRAIN_CUDA
		// move cuda model_probs back to CPU and sample
		cudaCheck(cudaMemcpy(cpu_probs, model_probs, modelp->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
		float *sample_probs = cpu_probs;
#endif
#ifdef TRAIN_CPU
		float *sample_probs = model_probs;
#endif
		int next_token = sample_mult(sample_probs, modelp->config.vocab_size, coin);
		gen_tokens[t] = next_token;
		// print the generated token, either using the Tokenizer or a fallback
		if (tokenizerp->init_ok) {
			const char* token_str = tokenizer_decode(tokenizerp, next_token);
			safe_printf(token_str);
		} else {
			// fall back to printing the token id
			printf("%d ", next_token);
		}
		fflush(stdout);
	}
	printf("\n---\n");

#ifdef TRAIN_CUDA
	free(cpu_probs);
#endif
}


