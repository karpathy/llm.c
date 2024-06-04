/*
Defines the schedule for the hyperparameters.
Only supports Cosine and WSD for now.
Planning on adding batch size schedule later.

lr_schedule_type = 0 if cosine schedule, and 1 if WSD schedule.

Guide on best practices when using WSD:
- The maximum learning rate should be around half the optimal one for cosine.
- The final_learning_rate_frac should be 0.0.
- For the number of decay_iterations, 20% of max_iterations seems like a good value. However, you can achieve good results (almost matching cosine) with 10% of max_iterations.
For more information, see this paper: https://arxiv.org/abs/2405.18392
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

typedef struct {
    float learning_rate;
    float max_learning_rate;
    float final_learning_rate_frac;
    float min_learning_rate;
    int lr_schedule_type; //cos (0) or wsd (1).
    int max_iterations;
    int warmup_iterations;
    int decay_iterations; // -1 if cos.
} LRSchedule;


void lr_schedule_init(LRSchedule *lr_schedule, float max_learning_rate, int lr_schedule_type, int max_iterations, int warmup_iterations, float final_learning_rate_frac, int decay_iterations) {
    lr_schedule->max_learning_rate = max_learning_rate;
    lr_schedule->final_learning_rate_frac = final_learning_rate_frac;
    lr_schedule->min_learning_rate = lr_schedule->max_learning_rate * lr_schedule->final_learning_rate_frac;
    lr_schedule->lr_schedule_type = lr_schedule_type;
    lr_schedule->max_iterations = max_iterations;
    lr_schedule->warmup_iterations = warmup_iterations;
    lr_schedule->decay_iterations = decay_iterations;
    lr_schedule->learning_rate= 0.0f;
    assert(!(lr_schedule->decay_iterations == -1 && lr_schedule->lr_schedule_type == 1) && "decay_iterations must be defined.");
}

void lr_step(LRSchedule *lr_schedule, int step) {
    if (lr_schedule->lr_schedule_type == 0) {
        // cosine learning rate schedule: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
        if (step < lr_schedule->warmup_iterations) {
            lr_schedule->learning_rate = lr_schedule->max_learning_rate * ((float)(step + 1)) / lr_schedule->warmup_iterations;
        } else {
            float decay_ratio = ((float)(step - lr_schedule->warmup_iterations)) / (lr_schedule->max_iterations - lr_schedule->warmup_iterations);
            assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
            float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
            assert(0.0f <= coeff && coeff <= 1.0f);
            //float min_lr = learning_rate * final_learning_rate_frac;
            lr_schedule->learning_rate = lr_schedule->min_learning_rate + coeff * (lr_schedule->max_learning_rate - lr_schedule->min_learning_rate );
        }
    } else if (lr_schedule->lr_schedule_type == 1) {
        // wsd learning rate schedule: warmup linearly, then constant learning rate, then "1-sqrt" shape decay to LR * final_learning_rate_frac (should be 0 for optimal perf)
        if (step < lr_schedule->warmup_iterations) {
            // warmup phase: linearly increase learning rate
            lr_schedule->learning_rate = lr_schedule->max_learning_rate * ((float)(step + 1)) / lr_schedule->warmup_iterations;
        } else if (step < lr_schedule->max_iterations - lr_schedule->decay_iterations) {
            // constant learning rate phase
            lr_schedule->learning_rate = lr_schedule->max_learning_rate;
        } else {
            // decay phase: 1 - square root decay
            float decay_ratio = ((float)(step - lr_schedule->max_iterations + lr_schedule->decay_iterations)) / lr_schedule->decay_iterations;
            assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
            lr_schedule->learning_rate = lr_schedule->min_learning_rate + (1.0f - sqrtf(decay_ratio)) * (lr_schedule->max_learning_rate - lr_schedule->min_learning_rate);
        }
    }
}