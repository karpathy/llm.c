#ifndef _WIN_H_
#define _WIN_H_

#define WIN32_LEAN_AND_MEAN      // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // pi
#endif

#define CLOCK_MONOTONIC  0
#define access _access
#define F_OK 0

#include <sys/types.h>

int clock_gettime(int clk_id, struct timespec* tp) {
    uint32_t ticks = GetTickCount();
    tp->tv_sec = ticks / 1000;
    tp->tv_nsec = (ticks % 1000) * 1000000;
    return 0;
}

#endif /*  _WIN_H_ */
