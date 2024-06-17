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

// Linear with warmup learning rate scheduler
typedef struct {
    float learning_rate;
    int warmup_iterations;
    int train_num_batches;
    float final_learning_rate_frac;
} LinearLearningRateScheduler;

typedef struct {
    float min_lr;
    float max_lr;
    int step_size;
} CyclicTriangularLearningRateScheduler;

// Constant learning rate scheduler
typedef struct {
    float learning_rate;
} ConstantLearningRateScheduler;

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

// linear warmup learning rate schedule: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
float get_learning_rate_linear(LinearLearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

// cyclic triangular learning rate schedule: linearly increase LR from min LR to max LR, then linearly decrease LR to min LR (repeat)
float get_learning_rate_triangular(CyclicTriangularLearningRateScheduler *scheduler, int step) {
    int cycle_index = 1 + step / (2 * scheduler->step_size);  // tells us which cycle we are in, starting at 1
    float x = fabsf((float)step / scheduler->step_size - 2 * cycle_index + 1);  // goes from 0 to 1 to 0
    float lr = scheduler->min_lr + (scheduler->max_lr - scheduler->min_lr) * fmaxf(0, (1 - x));
    return lr;
}

// constant learning rate schedule
float get_learning_rate_constant(ConstantLearningRateScheduler *scheduler, int step) {
    return scheduler->learning_rate;
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

void lr_scheduler_init_linear(LinearLearningRateScheduler *scheduler, float learning_rate, int warmup_iterations, int train_num_batches, float final_learning_rate_frac) {
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

void lr_scheduler_init_constant(ConstantLearningRateScheduler *scheduler, float learning_rate) {
    scheduler->learning_rate = learning_rate;
}

#endif // SCHEDULERS_H