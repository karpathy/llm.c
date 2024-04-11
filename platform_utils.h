#ifndef PLATFORM_UTILS_H
#define PLATFORM_UTILS_H

// ----------------------------------------------------------------------------
// platform-specific utilities

#if defined(__unix__)
#include <unistd.h>
#include <time.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1e6);
}

int file_exists(const char* filename) {
    return access(filename, F_OK) != -1;
}

#elif defined(_WIN32)
#include <windows.h>

double get_time_ms() {
    LARGE_INTEGER freq;
    LARGE_INTEGER time;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / (double)freq.QuadPart * 1000.0;
}

int file_exists(const char* filename) {
    DWORD dwAttrib = GetFileAttributes(filename);
    if (dwAttrib != INVALID_FILE_ATTRIBUTES && 
        !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY)) {
        return 1;
    }

    return 0;
}

#else
#error "Unsupported platform"
#endif

#endif // PLATFORM_UTILS_H
