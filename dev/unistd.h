// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h> // for malloc and free
#include <string.h>
#include <direct.h> // for _mkdir and _stat
#include <io.h> // needed for _access below and _findfirst, _findnext, _findclose

#define CLOCK_MONOTONIC 0
static inline int clock_gettime(int ignore_variable, struct timespec* tv)
{
    return timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, push )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(pop)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

typedef struct glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    char **gl_pathv;    // List of matched pathnames
} glob_t;

static inline void replace_forward_slashes(char* str) {
    while (*str) {
        if (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

static inline void globfree(glob_t *pglob) {
    for (size_t i = 0; i < pglob->gl_pathc; ++i) {
        free(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    free(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

static inline int glob(const char* pattern, int ignored_flags, int (*ignored_errfunc)(const char* epath, int eerrno), glob_t* pglob){
    struct _finddata_t find_file_data;
    char full_path[576]; // stored in pglob->gl_pathv[n]
    char directory_path[512] = {0}; // Store the directory path from the pattern
    char pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, sizeof(pattern_copy) - 1, pattern, sizeof(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    if (strchr(pattern_copy, '\\') != (void*) NULL) {
        strncpy_s(directory_path, sizeof(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    if (find_handle == -1) {
        return 1; // No files found
    }

    size_t file_count = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (char **) malloc(max_files * sizeof(char*)); // freed in globfree

    if (pglob->gl_pathv == NULL) {
        _findclose(find_handle);
        return 2; // Memory allocation failed
    }

    do {
        if (file_count >= max_files) {
            _findclose(find_handle);
            return 2; // Too many files found
            }

        snprintf(full_path, sizeof(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_count] = _strdup(full_path); // freed in globfree

        if (pglob->gl_pathv[file_count] == NULL) {
            _findclose(find_handle);
            return 2; // Memory allocation for filename failed
        }
        file_count++;
    } while (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_count;
    return 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
typedef struct dirent {
    char d_name[MAX_PATH_LENGTH];
} dirent;

typedef struct DIR {
    intptr_t handle;
    struct _finddata_t findFileData;
    int firstRead;
} DIR;

static inline DIR *opendir(const char *name) {
    DIR *dir = (DIR *)malloc(sizeof(DIR));
    if (dir == NULL) {
        return NULL;
    }

    char searchPath[MAX_PATH_LENGTH];

    snprintf(searchPath, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(searchPath, &dir->findFileData);
    if (dir->handle == -1) {
        free(dir);
        return NULL;
    }

    dir->firstRead = 1;
    return dir;
}

static inline struct dirent *readdir(DIR *directory) {
    static struct dirent result;

    if (directory->firstRead) {
        directory->firstRead = 0;
    } else {
        if (_findnext(directory->handle, &directory->findFileData) != 0) {
            return NULL;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    return &result;
}

static inline int closedir(DIR *directory) {
    if (directory == NULL) {
        return -1;
    }

    if (_findclose(directory->handle) != 0) {
        return -1;
    }

    free(directory);
    return 0;
}
#endif // UNISTD_H
