// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <math.h>
//#define gen_max_length 64 // compile as C++ to skip this VLA issue
#include <time.h>

#define CLOCK_MONOTONIC 0
int clock_gettime(int ignore_variable, struct timespec* tv)
{
    return timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#include  <io.h> /* needed for access below */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, push )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(pop)) // Restore file's default settings

#endif
