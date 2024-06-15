/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

*/

#include "../../llmc/dataloader.h"

char shard_name[64];
int num_tokens = 140;
int num_shards = 4;

void checkRange(const int *tokens, const int start, const int end) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    int n = end - start;
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != start + i) {
            printf("Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}

void checkEquals(const int *tokens, const int n, const int expected) {
    // checks that the tokens[0, ... n] are all equal to expected
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != expected) {
            printf("Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}

void test_simple() {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    int B = 4;
    int T = 8;
    int process_rank = 0;
    int num_processes = 1;
    int should_shuffle = 0;
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    int batches_fit = num_tokens / (B * T); // number of batches that fit per shard
    int BT = B * T;
    int num_epochs = 4;
    for (int e = 0; e < num_epochs; e ++) { // epoch
        for (int s = 0; s < num_shards; s++) { // shard
            int start = s * num_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                checkRange(loader.inputs, start, start + BT);
                checkRange(loader.targets, start + 1, start + BT + 1);
                start += BT;
            }
        }
    }
    dataloader_free(&loader);
    printf("OK\n");
}

void test_shuffled() {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    printf("test_shuffled... ");
    int B = 4;
    int T = 8;
    int process_rank = 0;
    int num_processes = 1;
    int should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    int total_tokens = num_shards * num_tokens;
    int *num_seen_inputs = (int *)calloc(total_tokens, sizeof(int));
    int *num_seen_targets = (int *)calloc(total_tokens, sizeof(int));
    int batches_fit = num_tokens / (B * T); // number of batches that fit per shard
    int BT = B * T;
    int num_epochs = 4;
    for (int e = 0; e < num_epochs; e ++) { // epoch
        for (int s = 0; s < num_shards; s++) { // shard
            int start = s * num_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                for (int i = 0; i < BT; i++) {
                    int input_token = loader.inputs[i];
                    int target_token = loader.targets[i];
                    assert(input_token >= 0 && input_token < total_tokens);
                    assert(target_token >= 0 && target_token < total_tokens);
                    num_seen_inputs[input_token]++;
                    num_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    int tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    for (int s = 0; s < num_shards; s++) {
        int start = s * num_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(num_seen_inputs + start, tokens_fit, num_epochs);
        checkEquals(num_seen_inputs + start + tokens_fit, num_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(num_seen_targets + start + 1, tokens_fit, num_epochs);
        checkEquals(num_seen_targets + start + 1 + tokens_fit, num_tokens - tokens_fit, 0);
    }

    dataloader_free(&loader);
    free(num_seen_inputs);
    free(num_seen_targets);
    printf("OK\n");
}

int main(void) {

    // generate a few dummy shards of data with incrementing tokens
    int header[HEADER_SIZE];
    uint16_t tokens[num_tokens];
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        int token_offset = shard_id * num_tokens;
        for (int i = 0; i < num_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, 64, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = num_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, sizeof(int), HEADER_SIZE, shard_file);
        fwrite(tokens, sizeof(uint16_t), num_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_shuffled();

    // clean up the shards
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        snprintf(shard_name, 64, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

}