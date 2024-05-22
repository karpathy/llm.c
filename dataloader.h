/*
Implements a medium simple DataLoader for a distributed training setup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------
// we need glob to list files matching a pattern
// windows does not have glob, so we fall back on a very simple implementation
// this implementation doesn't actually do a glob, it assumes that the "pattern"
// is exactly the single file of interest
#ifndef _WIN32
#include <glob.h>
#else

typedef struct glob_t {
    size_t gl_pathc;
    char **gl_pathv;
} glob_t;

int glob(const char *pattern, int flags, void *unused, glob_t *pglob) {
    assert(strstr(pattern, "*") == NULL); // we don't support * here
    pglob->gl_pathc = 1;
    pglob->gl_pathv = (char **)malloc(sizeof(char *));
    if (pglob->gl_pathv == NULL) { exit(EXIT_FAILURE); } // ??? oom?
    pglob->gl_pathv[0] = (char *)pattern;
    return 0;
}

void globfree(glob_t* pglob) {
    free(pglob->gl_pathv);
}
#endif

// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B;
    size_t T;
    // input handling and its state
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    int current_shard; // the current shard we are reading from
    FILE* tokens_file;
    long file_size;
    long current_position;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    size_t num_batches;
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
} DataLoader;

long dataloader_load_shard_(DataLoader *loader, int shard_index) {
    // use the first glob match as the filename for now
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    long ntok = header[2]; // number of tokens in the file
    assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
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
    return ntok;
}

void dataloader_reset(DataLoader *loader) {
    // fully resets the DataLoader object to init configuration
    // each process starts at a different offset in the file
    long header_bytes = HEADER_SIZE * sizeof(int);
    long token_bytes_offset = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);
    loader->current_shard = 0;
    loader->current_position = header_bytes + token_bytes_offset;
    dataloader_load_shard_(loader, loader->current_shard);
}

void dataloader_advance_(DataLoader *loader) {
    // advance the loader by loading the next data shard and resetting the position
    if (loader->glob_result.gl_pathc > 1) {
        // if we have more than one shard, advance to the next one
        loader->current_shard = (loader->current_shard + 1) % loader->glob_result.gl_pathc;
        dataloader_load_shard_(loader, loader->current_shard);
    }
    long header_bytes = HEADER_SIZE * sizeof(int);
    long token_bytes_offset = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);
    loader->current_position = header_bytes + token_bytes_offset;
}

void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = NULL;

    // glob to get the list of files matching the pattern, these are our data shards
    int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
    if (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    long ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        long shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        assert(shard_ntok >= num_processes * B * T + 1);
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)malloc((B * T + 1) * sizeof(uint16_t));
    loader->inputs = (int*)malloc(B * T * sizeof(int));
    loader->targets = (int*)malloc(B * T * sizeof(int));
    loader->num_batches = ntok_total / (num_processes * B * T); // useful to know

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

void dataloader_next_batch(DataLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
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
    // we only load B * T + 1 tokens at each iteration because the targets are offset by 1
    loader->current_position += loader->num_processes * B * T * sizeof(uint16_t);
    // if the next batch would go past the end of the file, advance the loader
    if (loader->current_position + (loader->num_processes * B * T + 1) * sizeof(uint16_t) > loader->file_size) {
        dataloader_advance_(loader);
    }
}

void dataloader_free(DataLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
}
