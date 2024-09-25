/*
Our goal here is to load the .bin files generated by rope.py and match
the implementation in C and get the same results as in rope.py.

Compile and run simply with:

gcc -o rope rope.c -lm
./rope
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// ----------------------------------------------------------------------------
// a few utils for safety
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
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
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

extern inline void *malloc_check(size_t size, const char *file, int line) {
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

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

int compare_arrays(const float *arr1, const float *arr2, size_t size, const char *name, float epsilon) {
    for (size_t i = 0; i < size; i++) {
        // print 10 elements that are equally spaced out, for qualitative check
        if (i % (size / 10) == 0) {
            printf("arr1[%zu] = %f, arr2[%zu] = %f\n", i, arr1[i], i, arr2[i]);
        }
        if (fabsf(arr1[i] - arr2[i]) > epsilon) {
            printf("Error: %s[%zu] = %f, expected %f (diff: %f)\n",
                   name, i, arr1[i], arr2[i], fabsf(arr1[i] - arr2[i]));
            return 0;
        }
    }
    printf("OK: %s\n", name);
    return 1;
}

// ----------------------------------------------------------------------------
// all the functions we need

void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta, int use_scaled) {
    // same as precompute_freqs_cis_real in rope.py
    for (int i = 0; i < dim / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float freq = 1.0f / powf(theta, (float)(2 * i) / dim);
        if (use_scaled) {
            const int scale_factor = 8;
            const int low_freq_factor = 1;
            const int high_freq_factor = 4;
            const int old_context_len = 8192;  // original llama3 length
            const float low_freq_wavelen = (float)old_context_len / low_freq_factor;
            const float high_freq_wavelen = (float)old_context_len / high_freq_factor;
            float wavelen = 2.0f * M_PI / freq;
            if (wavelen < high_freq_wavelen) {
                // skip; keep freq as is
            } else if (wavelen > low_freq_wavelen) {
                // scale down by scale_factor
                freq /= scale_factor;
            } else {
                // smooth transition between scaled and unscaled
                float smooth = ((float)old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                freq = (1.0f - smooth) * freq / scale_factor + smooth * freq;
            }
        }

        // iterate over all time steps, calculate the angle, and store the cos/sin
        for (int t = 0; t < end; t++) {
            float angle = (float)t * freq;
            freqs_cis[t * dim + 2 * i] = cosf(angle);     // real part
            freqs_cis[t * dim + 2 * i + 1] = sinf(angle); // imaginary part
        }
    }
}

void apply_rotary_emb_forward(float *out, const float *inp, const float *freqs_cis, int B, int T, int n_head, int head_dim) {
    // same as apply_rotary_emb_real in rope.py
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
            for (int h = 0; h < n_head; h++) {
                int idx_bth = idx_bt + h * head_dim;
                for (int d = 0; d < head_dim / 2; d++) {
                    // fetch a tuple of activations, which we imagine as a complex number
                    int idx = idx_bth + 2 * d;
                    float x_real = inp[idx];
                    float x_imag = inp[idx + 1];
                    // fetch the angle from freqs_cis
                    int freqs_idx = t * head_dim + 2 * d;
                    float freqs_cos = freqs_cis[freqs_idx];
                    float freqs_sin = freqs_cis[freqs_idx + 1];
                    // apply the rotation
                    out[idx] = x_real * freqs_cos - x_imag * freqs_sin;
                    out[idx + 1] = x_real * freqs_sin + x_imag * freqs_cos;
                }
            }
        }
    }
}

void apply_rotary_emb_backward(float *dinp, const float *dout, const float *inp, const float *freqs_cis, int B, int T, int n_head, int head_dim) {
    // backward pass of the RoPE embedding
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
            for (int h = 0; h < n_head; h++) {
                int idx_bth = idx_bt + h * head_dim;
                for (int d = 0; d < head_dim / 2; d++) {
                    // fetch the angle from freqs_cis
                    int freqs_idx = t * head_dim + 2 * d;
                    float freqs_cos = freqs_cis[freqs_idx];
                    float freqs_sin = freqs_cis[freqs_idx + 1];
                    // and the input index we'll be updating
                    int idx = idx_bth + 2 * d;
                    // backward pass is simple because freqs_cis is just scaling by a constant
                    dinp[idx] += dout[idx] * freqs_cos + dout[idx + 1] * freqs_sin;
                    dinp[idx + 1] += -dout[idx] * freqs_sin + dout[idx + 1] * freqs_cos;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------

int main() {

    // load the .bin file
    FILE *file = fopen("rope.bin", "rb");
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return 1;
    }
    // read the header
    int int_header[16];
    float float_header[16];
    freadCheck(int_header, sizeof(int), 16, file);
    freadCheck(float_header, sizeof(float), 16, file);
    // check the magic number
    if (int_header[0] != 20240924) {
        printf("Error: Invalid magic number.\n");
        fclose(file);
        return 1;
    }
    // extract the hyperparameters
    int B = int_header[1];
    int T = int_header[2];
    int n_embd = int_header[3];
    int n_head = int_header[4];
    int use_scaled_rope = int_header[5];
    float rope_theta = float_header[0];
    int head_dim = n_embd / n_head;
    // read the inputs
    float *inp = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    freadCheck(inp, sizeof(float), B * T * n_head * head_dim, file);
    // read the freqs_cis
    float *freqs_cis_target = (float *)mallocCheck(T * head_dim * sizeof(float));
    freadCheck(freqs_cis_target, sizeof(float), T * head_dim, file);
    // read the output
    float *out_target = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    freadCheck(out_target, sizeof(float), B * T * n_head * head_dim, file);
    // read the weights for the loss function
    float *wei = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    freadCheck(wei, sizeof(float), B * T * n_head * head_dim, file);
    // read the input gradients
    float *inp_grad_target = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    freadCheck(inp_grad_target, sizeof(float), B * T * n_head * head_dim, file);
    // ensure we exactly exhausted the file
    long current_position = ftell(file);
    // Get the file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    // check if we read the whole file
    if (current_position != file_size) {
        printf("Error: File was not read properly; %ld bytes left unread.\n", file_size - current_position);
        fclose(file);
        return 1;
    }
    fclose(file);

    // print the hyperparameters
    printf("B: %d, T: %d, n_embd: %d, n_head: %d, use_scaled_rope: %d, rope_theta: %f\n",
            B, T, n_embd, n_head, use_scaled_rope, rope_theta);

    // Step 1) Calculate freqs_cis in C and compare with the Python one
    float *freqs_cis = (float *)mallocCheck(T * head_dim * sizeof(float));
    precompute_freqs_cis(freqs_cis, head_dim, T, rope_theta, use_scaled_rope);
    if (!compare_arrays(freqs_cis, freqs_cis_target, T * head_dim, "freqs_cis", 1e-6f)) { return 1; }

    // Step 2) Apply the RoPE embedding in C and compare with the Python one
    float *out = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    apply_rotary_emb_forward(out, inp, freqs_cis, B, T, n_head, head_dim);
    if (!compare_arrays(out, out_target, B * T * n_head * head_dim, "out", 1e-6f)) { return 1; }

    // Step 3) Calculate the loss and gradients in C and compare with the Python one
    float *dout = wei; // wei is dout because the loss is just a dot product of out and wei
    float *dinp = (float *)mallocCheck(B * T * n_head * head_dim * sizeof(float));
    apply_rotary_emb_backward(dinp, dout, inp, freqs_cis, B, T, n_head, head_dim);
    if (!compare_arrays(dinp, inp_grad_target, B * T * n_head * head_dim, "dinp", 1e-6f)) { return 1; }

    printf("✅ ALL OK\n");

    // clean up
    free(inp);
    free(freqs_cis_target);
    free(out_target);
    free(wei);
    free(inp_grad_target);
    free(freqs_cis);
    free(out);
    free(dinp);

    return 0;
}