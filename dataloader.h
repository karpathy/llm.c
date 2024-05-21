/*
Implements a medium simple DataLoader for a distributed training setup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef struct {
    // Distributed data parallel specifics.
    // Each worker loads it's own chunk of data.
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B;
    size_t T;
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // outputs
    uint16_t* buffer; // used to fread data from file into
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    // convenience variables
    size_t num_batches;
} DataLoader;

void dataloader_reset(DataLoader *loader) {
    // each process starts at a different offset in the file
    long header_bytes = HEADER_SIZE * sizeof(int);
    long token_bytes_offset = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);
    loader->current_position = header_bytes + token_bytes_offset;
}

void dataloader_init(DataLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    if (header[0] != 20240520) { printf("Bad magic in data file\n"); exit(EXIT_FAILURE); }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    long ntok = header[2]; // number of tokens in the file

    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    long expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    if (ntok < num_processes * B * T + 1) {
        // being too defensive/lazy, we could tolerate as low as T+1 tokens in principle
        printf("Error: there are too few tokens\n");
        exit(EXIT_FAILURE);
    }

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->buffer = (uint16_t*)malloc((B * T + 1) * sizeof(uint16_t));
    loader->inputs = (int*)malloc(B * T * sizeof(int));
    loader->targets = (int*)malloc(B * T * sizeof(int));
    // note: we definitely want to advance by B * T; That is the "stride" by which we move
    // the window of tokens. We only load B * T + 1 tokens because our targets are offset by 1
    loader->num_batches = ntok / (num_processes * B * T);

    // reset the loader to the beginning of the file
    dataloader_reset(loader);
}

void dataloader_next_batch(DataLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (loader->num_processes * B * T + 1) * sizeof(uint16_t) > loader->file_size) {
        dataloader_reset(loader);
    }
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, loader->current_position, SEEK_SET);
    freadCheck(loader->buffer, sizeof(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    for (int i = 0; i < B*T; i++) {
        loader->inputs[i] = (int)loader->buffer[i];
        loader->targets[i] = (int)loader->buffer[i+1];
    }
    // advance the current position by B*T*num_processes integers
    // note: the "stride" of tokens by which we move each time is definitely B * T
    loader->current_position += loader->num_processes * B * T * sizeof(uint16_t);
}

void dataloader_free(DataLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    fcloseCheck(loader->tokens_file);
}
