/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H

#define SCHEDULERS_H

#include <assert.h>
#include <math.h>

//
// Learning rate scheduler structs
//

typedef struct {
    float learning_rate;
    int warmup_iterations;
    int train_num_batches;
    float final_learning_rate_frac;
} CosineLearningRateScheduler;

typedef struct {
    float min_lr;
    float max_lr;
    int step_size;
} CyclicTriangularLearningRateScheduler;

//
// Learning rate scheduler functions
//

// cosine learning rate schedule: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
float get_learning_rate_cosine(CosineLearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

// cyclic triangular learning rate schedule: linearly increase LR from min LR to max LR, then linearly decrease LR to min LR (repeat)
float get_learning_rate_triangular(CyclicTriangularLearningRateScheduler *scheduler, int step) {
    int cycle = 1 + step / (2 * scheduler->step_size);
    float x = fabsf((float)step / scheduler->step_size - 2 * cycle + 1);
    float lr = scheduler->min_lr + (scheduler->max_lr - scheduler->min_lr) * fmaxf(0, (1 - x));
    return lr;
}

//
// Init functions
//

void lr_scheduler_init_cosine(CosineLearningRateScheduler *scheduler, float learning_rate, int warmup_iterations, int train_num_batches, float final_learning_rate_frac) {
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

void lr_scheduler_init_triangular(CyclicTriangularLearningRateScheduler *scheduler, float min_lr, float max_lr, int step_size) {
    scheduler->min_lr = min_lr;
    scheduler->max_lr = max_lr;
    scheduler->step_size = step_size;
}

#endif // SCHEDULERS_H