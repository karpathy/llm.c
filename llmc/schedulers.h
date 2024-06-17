/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H

#define SCHEDULERS_H

#include <assert.h>
#include <math.h>

typedef struct {
    float learning_rate;
    int warmup_iterations;
    int train_num_batches;
    float final_learning_rate_frac;
} CosineLearningRateScheduler;

// learning rate schedule: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
float get_learning_rate(CosineLearningRateScheduler *scheduler, int step) {
    float step_learning_rate = scheduler->learning_rate;
    if (step < scheduler->warmup_iterations) {
        step_learning_rate = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        step_learning_rate = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    return step_learning_rate;
}

void lr_scheduler_init(CosineLearningRateScheduler *scheduler, float learning_rate, int warmup_iterations, int train_num_batches, float final_learning_rate_frac) {
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

#endif // SCHEDULERS_H