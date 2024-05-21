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
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    size_t num_batches;
} DataLoader;

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

    // determine the file size
    fseekCheck(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseekCheck(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(EXIT_FAILURE);
    }
    loader->current_position = loader->process_rank * B * T * sizeof(int); // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->batch = (int*)malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    // note: we definitely want to advance by B * T; That is the "stride" by which we move
    // the window of tokens. We only load B * T + 1 tokens because our targets are offset by 1
    loader->num_batches = loader->file_size / (loader->num_processes * B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (loader->num_processes * B * T + 1) * sizeof(int) > loader->file_size) {
        loader->current_position = loader->process_rank * B * T * sizeof(int);
    }
    // read the B*T+1 integers from the file into batch
    fseekCheck(loader->tokens_file, loader->current_position, SEEK_SET);
    freadCheck(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T*num_processes integers
    // note: the "stride" of tokens by which we move each time is definitely B * T
    loader->current_position += loader->num_processes * B * T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    free(loader->batch);
    fcloseCheck(loader->tokens_file);
}
