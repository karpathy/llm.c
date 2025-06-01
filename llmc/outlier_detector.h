/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#include <stdio.h>
#include <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 100

typedef struct {
    double buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    int count;
    int index;
    int skipped_in_a_row;
    double sum;
    double sum_sq;
} OutlierDetector;

void init_detector(OutlierDetector *detector) {
    for (int i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->count = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

double update_detector(OutlierDetector *detector, double new_value, double skip_update_threshold) {

    if (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->count] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        detector->skipped_in_a_row = 0;
        detector->count++;
        return nan(""); // not enough data yet

    } else {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        double old_value = detector->buffer[detector->index];
        detector->sum -= old_value;
        detector->sum_sq -= old_value * old_value;
        // push the new value into the window
        detector->buffer[detector->index] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        double mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        double variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        double std_dev = sqrt(variance);
        if (std_dev == 0.0) {
            return 0.0;
        }
        double z = (new_value - mean) / std_dev;

        if (skip_update_threshold != 0.0 && z > skip_update_threshold) {
            // let's go back in time and pretend this never happened
            // i.e. don't let bad outliers affect the threshold for detecting future outliers
            // otherwise the detector will get less picky and accept things it really shouldn't!
            // but we do update on consecutive outliers, to avoid getting stuck completely
            detector->skipped_in_a_row++;
            if (detector->skipped_in_a_row <= 1) {
                detector->index = (detector->index - 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
                detector->sum += old_value - new_value;
                detector->sum_sq += (old_value * old_value) - (new_value * new_value);
                detector->buffer[detector->index] = old_value;
            }
        } else {
            detector->skipped_in_a_row = 0;
        }

        return z;
    }
}
