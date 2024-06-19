/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H

#define SCHEDULERS_H

#include <assert.h>
#include <math.h>

typedef enum {
    LR_SCHEDULER_COSINE,
    LR_SCHEDULER_LINEAR,
    LR_SCHEDULER_TRIANGULAR,
    LR_SCHEDULER_CONSTANT,
    NUM_LR_SCHEDULERS   // To keep track of the number of schedulers
} LRSchedulerType;

const char* lr_scheduler_names[] = {
    "cosine",
    "linear",
    "triangular",
    "constant",
};

const char* get_lr_scheduler_name(LRSchedulerType type) {
    if (type < 0 || type >= NUM_LR_SCHEDULERS) {
        exit(EXIT_FAILURE);
    }
    return lr_scheduler_names[type];
}

LRSchedulerType get_lr_scheduler_type_from_name(const char* name) {
    for (int i = 0; i < NUM_LR_SCHEDULERS; ++i) {
        if (strcmp(name, lr_scheduler_names[i]) == 0) {
            return (LRSchedulerType)i;
        }
    }
    printf("Warning: Unknown learning rate scheduler name: %s\n. Using cosine as default.", name);
    return LR_SCHEDULER_COSINE;  // Default to cosine if not found
}

//
// Learning rate scheduler structs and init
//

typedef struct {
    float learning_rate;
    int warmup_iterations;
    int train_num_batches;
    float final_learning_rate_frac;
} LearningRateScheduler;

void lr_scheduler_init(LearningRateScheduler *scheduler, float learning_rate, int warmup_iterations, int train_num_batches, float final_learning_rate_frac) {
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

//
// Learning rate scheduler functions
//

// switch to the appropriate learning rate scheduler
float get_learning_rate(LRSchedulerType lr_scheduler_type, LearningRateScheduler *scheduler, int step) {
    float step_learning_rate;
    if (lr_scheduler_type == LR_SCHEDULER_COSINE) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } else if (lr_scheduler_type == LR_SCHEDULER_LINEAR) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } else if (lr_scheduler_type == LR_SCHEDULER_TRIANGULAR) {
        step_learning_rate = get_learning_rate_triangular(scheduler, step);
    } else if (lr_scheduler_type == LR_SCHEDULER_CONSTANT) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } else {
        printf("Unknown learning rate scheduler type\n");
        exit(EXIT_FAILURE);
    }
    return step_learning_rate;
}

// cosine learning rate schedule: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
float get_learning_rate_cosine(LearningRateScheduler *scheduler, int step) {
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
float get_learning_rate_linear(LearningRateScheduler *scheduler, int step) {
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
// currently hardcoded to support only a single cycle
float get_learning_rate_triangular(LearningRateScheduler *scheduler, int step) {
    int step_size = scheduler->train_num_batches / 2;  // number of steps in half a cycle
    float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
    float max_lr = scheduler->learning_rate;

    int cycle_index = 1 + step / (2 * step_size);  // tells us which cycle we are in, starting at 1
    float x = fabsf((float)step / step_size - 2 * cycle_index + 1);  // goes from 0 to 1 to 0
    float lr = min_lr + (max_lr - min_lr) * fmaxf(0, (1 - x));
    return lr;
}

// constant learning rate schedule
float get_learning_rate_constant(LearningRateScheduler *scheduler, int step) {
    return scheduler->learning_rate;
}

#endif // SCHEDULERS_H