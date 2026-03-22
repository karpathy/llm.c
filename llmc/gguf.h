/*
GGUF file format parser.
Supports GGUF v3, tensor types F32/F16/Q4_0/Q8_0.
*/
#ifndef GGUF_H
#define GGUF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// ----------------------------------------------------------------------------
// GGUF types

#define GGUF_MAGIC 0x46554747  // "GGUF"
#define GGUF_ALIGNMENT 32

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFValueType;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8,
} GGMLType;

#define GGML_TYPE_COUNT 16

static const int ggml_blk_size[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = 1,
    [GGML_TYPE_F16]  = 1,
    [GGML_TYPE_Q4_0] = 32,
    [GGML_TYPE_Q4_1] = 32,
    [6]  = 32,   // Q5_0
    [7]  = 32,   // Q5_1
    [GGML_TYPE_Q8_0] = 32,
    [10] = 256,  // Q2_K
    [11] = 256,  // Q3_K
    [12] = 256,  // Q4_K
    [13] = 256,  // Q5_K
    [14] = 256,  // Q6_K
};

static const int ggml_type_size[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = 4,
    [GGML_TYPE_F16]  = 2,
    [GGML_TYPE_Q4_0] = 18,  // 2 (f16 scale) + 16 (32 x 4bit)
    [GGML_TYPE_Q4_1] = 20,  // 2 + 2 + 16
    [6]  = 22,   // Q5_0
    [7]  = 24,   // Q5_1
    [GGML_TYPE_Q8_0] = 34,  // 2 (f16 scale) + 32 (int8 quants)
    [10] = 84,   // Q2_K
    [11] = 110,  // Q3_K
    [12] = 144,  // Q4_K
    [13] = 176,  // Q5_K
    [14] = 210,  // Q6_K
};

// ----------------------------------------------------------------------------
// GGUF value (tagged union)

typedef struct GGUFValue GGUFValue;

typedef struct {
    GGUFValueType type;
    uint64_t count;
    GGUFValue *items;
} GGUFArray;

struct GGUFValue {
    GGUFValueType type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint8_t  bool_;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        char    *str;     // null-terminated, heap allocated
        GGUFArray arr;
    };
};

// ----------------------------------------------------------------------------
// GGUF metadata key-value pair

typedef struct {
    char      *key;   // null-terminated
    GGUFValue  value;
} GGUFMetaKV;

// ----------------------------------------------------------------------------
// GGUF tensor info

#define GGUF_MAX_DIMS 4

typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[GGUF_MAX_DIMS];
    GGMLType  type;
    uint64_t  offset;   // offset from start of tensor data section
    void     *data;     // pointer into mmap'd or malloc'd buffer
    size_t    data_size;
} GGUFTensor;

// ----------------------------------------------------------------------------
// GGUF context

typedef struct {
    uint32_t     version;
    uint64_t     n_tensors;
    uint64_t     n_kv;
    GGUFMetaKV  *kv;
    GGUFTensor  *tensors;

    // raw file data (malloc'd)
    void        *data;
    size_t       data_size;
    size_t       data_offset;  // offset in file where tensor data begins
} GGUFContext;

// ----------------------------------------------------------------------------
// Internal read helpers

static void gguf_read_check(FILE *f, void *dst, size_t n, const char *label) {
    if (fread(dst, 1, n, f) != n) {
        fprintf(stderr, "GGUF: failed to read %s\n", label);
        exit(EXIT_FAILURE);
    }
}

static char *gguf_read_string(FILE *f) {
    uint64_t len;
    gguf_read_check(f, &len, sizeof(len), "string length");
    char *s = (char *)malloc(len + 1);
    if (!s) { fprintf(stderr, "GGUF: OOM\n"); exit(EXIT_FAILURE); }
    gguf_read_check(f, s, len, "string data");
    s[len] = '\0';
    return s;
}

static void gguf_read_value(FILE *f, GGUFValue *v, GGUFValueType type);

static void gguf_read_array(FILE *f, GGUFArray *arr) {
    uint32_t elem_type;
    gguf_read_check(f, &elem_type, sizeof(elem_type), "array elem type");
    arr->type = (GGUFValueType)elem_type;
    gguf_read_check(f, &arr->count, sizeof(arr->count), "array count");
    arr->items = (GGUFValue *)calloc(arr->count, sizeof(GGUFValue));
    if (!arr->items) { fprintf(stderr, "GGUF: OOM\n"); exit(EXIT_FAILURE); }
    for (uint64_t i = 0; i < arr->count; i++) {
        gguf_read_value(f, &arr->items[i], arr->type);
    }
}

static void gguf_read_value(FILE *f, GGUFValue *v, GGUFValueType type) {
    v->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   gguf_read_check(f, &v->u8,   1, "u8");   break;
        case GGUF_TYPE_INT8:    gguf_read_check(f, &v->i8,   1, "i8");   break;
        case GGUF_TYPE_UINT16:  gguf_read_check(f, &v->u16,  2, "u16");  break;
        case GGUF_TYPE_INT16:   gguf_read_check(f, &v->i16,  2, "i16");  break;
        case GGUF_TYPE_UINT32:  gguf_read_check(f, &v->u32,  4, "u32");  break;
        case GGUF_TYPE_INT32:   gguf_read_check(f, &v->i32,  4, "i32");  break;
        case GGUF_TYPE_FLOAT32: gguf_read_check(f, &v->f32,  4, "f32");  break;
        case GGUF_TYPE_BOOL:    gguf_read_check(f, &v->bool_,1, "bool"); break;
        case GGUF_TYPE_UINT64:  gguf_read_check(f, &v->u64,  8, "u64");  break;
        case GGUF_TYPE_INT64:   gguf_read_check(f, &v->i64,  8, "i64");  break;
        case GGUF_TYPE_FLOAT64: gguf_read_check(f, &v->f64,  8, "f64");  break;
        case GGUF_TYPE_STRING:  v->str = gguf_read_string(f);             break;
        case GGUF_TYPE_ARRAY:   gguf_read_array(f, &v->arr);              break;
        default:
            fprintf(stderr, "GGUF: unknown value type %d\n", type);
            exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------
// Main load function

static GGUFContext *gguf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "GGUF: cannot open '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    // magic
    uint32_t magic;
    gguf_read_check(f, &magic, 4, "magic");
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: bad magic 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        exit(EXIT_FAILURE);
    }

    GGUFContext *ctx = (GGUFContext *)calloc(1, sizeof(GGUFContext));
    if (!ctx) { fprintf(stderr, "GGUF: OOM\n"); exit(EXIT_FAILURE); }

    gguf_read_check(f, &ctx->version,  4, "version");
    gguf_read_check(f, &ctx->n_tensors, 8, "n_tensors");
    gguf_read_check(f, &ctx->n_kv,      8, "n_kv");

    // metadata key-value pairs
    ctx->kv = (GGUFMetaKV *)calloc(ctx->n_kv, sizeof(GGUFMetaKV));
    if (!ctx->kv) { fprintf(stderr, "GGUF: OOM\n"); exit(EXIT_FAILURE); }

    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        ctx->kv[i].key = gguf_read_string(f);
        uint32_t vtype;
        gguf_read_check(f, &vtype, 4, "kv value type");
        gguf_read_value(f, &ctx->kv[i].value, (GGUFValueType)vtype);
    }

    // tensor info
    ctx->tensors = (GGUFTensor *)calloc(ctx->n_tensors, sizeof(GGUFTensor));
    if (!ctx->tensors) { fprintf(stderr, "GGUF: OOM\n"); exit(EXIT_FAILURE); }

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        GGUFTensor *t = &ctx->tensors[i];
        t->name = gguf_read_string(f);
        gguf_read_check(f, &t->n_dims, 4, "n_dims");
        for (uint32_t d = 0; d < t->n_dims; d++) {
            gguf_read_check(f, &t->dims[d], 8, "dim");
        }
        uint32_t ttype;
        gguf_read_check(f, &ttype, 4, "tensor type");
        t->type = (GGMLType)ttype;
        gguf_read_check(f, &t->offset, 8, "offset");
    }

    // tensor data starts at next alignment boundary
    long header_end = ftell(f);
    long align_end = ((header_end + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    ctx->data_offset = (size_t)align_end;

    // compute total tensor data size
    uint64_t max_end = 0;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        GGUFTensor *t = &ctx->tensors[i];
        // compute number of elements
        uint64_t ne = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        // compute bytes
        int bsz = ggml_blk_size[t->type];
        int tsz = ggml_type_size[t->type];
        uint64_t n_blocks = (ne + bsz - 1) / bsz;
        t->data_size = (size_t)(n_blocks * tsz);
        uint64_t end = t->offset + t->data_size;
        if (end > max_end) max_end = end;
    }
    ctx->data_size = (size_t)max_end;

    // load tensor data
    ctx->data = malloc(ctx->data_size);
    if (!ctx->data && ctx->data_size > 0) {
        fprintf(stderr, "GGUF: OOM for tensor data (%zu bytes)\n", ctx->data_size);
        exit(EXIT_FAILURE);
    }
    fseek(f, (long)ctx->data_offset, SEEK_SET);
    size_t got = fread(ctx->data, 1, ctx->data_size, f);
    if (got != ctx->data_size) {
        fprintf(stderr, "GGUF: partial tensor data read (%zu / %zu)\n", got, ctx->data_size);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    // set data pointers
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        GGUFTensor *t = &ctx->tensors[i];
        t->data = (char *)ctx->data + t->offset;
    }

    return ctx;
}

// ----------------------------------------------------------------------------
// Metadata lookup helpers

static GGUFValue *gguf_find_kv(GGUFContext *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            return &ctx->kv[i].value;
        }
    }
    return NULL;
}

static const char *gguf_get_str(GGUFContext *ctx, const char *key, const char *def) {
    GGUFValue *v = gguf_find_kv(ctx, key);
    if (!v || v->type != GGUF_TYPE_STRING) return def;
    return v->str;
}

static uint64_t gguf_get_u64(GGUFContext *ctx, const char *key, uint64_t def) {
    GGUFValue *v = gguf_find_kv(ctx, key);
    if (!v) return def;
    switch (v->type) {
        case GGUF_TYPE_UINT8:  return v->u8;
        case GGUF_TYPE_INT8:   return (uint64_t)v->i8;
        case GGUF_TYPE_UINT16: return v->u16;
        case GGUF_TYPE_INT16:  return (uint64_t)v->i16;
        case GGUF_TYPE_UINT32: return v->u32;
        case GGUF_TYPE_INT32:  return (uint64_t)v->i32;
        case GGUF_TYPE_UINT64: return v->u64;
        case GGUF_TYPE_INT64:  return (uint64_t)v->i64;
        default: return def;
    }
}

static float gguf_get_f32(GGUFContext *ctx, const char *key, float def) {
    GGUFValue *v = gguf_find_kv(ctx, key);
    if (!v) return def;
    if (v->type == GGUF_TYPE_FLOAT32) return v->f32;
    if (v->type == GGUF_TYPE_FLOAT64) return (float)v->f64;
    return def;
}

static GGUFTensor *gguf_find_tensor(GGUFContext *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

// ----------------------------------------------------------------------------
// F16 -> F32 conversion

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            // denormal
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // inf or nan
        result = sign | 0x7F800000 | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ----------------------------------------------------------------------------
// Dequantize a tensor row to float

// Returns number of elements written to out.
// out must have space for `ne` floats.
static void gguf_dequantize_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0]; // elements per row (first dim is fastest)
    uint8_t *src = (uint8_t *)t->data;

    if (t->type == GGML_TYPE_F32) {
        float *p = (float *)src + row * row_size;
        memcpy(out, p, row_size * sizeof(float));
        return;
    }

    if (t->type == GGML_TYPE_F16) {
        uint16_t *p = (uint16_t *)src + row * row_size;
        for (uint64_t i = 0; i < row_size; i++) {
            out[i] = f16_to_f32(p[i]);
        }
        return;
    }

    int blk_size = ggml_blk_size[t->type];
    int blk_bytes = ggml_type_size[t->type];
    uint64_t n_blocks_per_row = (row_size + blk_size - 1) / blk_size;
    uint8_t *row_data = src + row * n_blocks_per_row * blk_bytes;

    if (t->type == GGML_TYPE_Q8_0) {
        // block = [f16 scale (2 bytes)] [32 x int8 (32 bytes)]
        for (uint64_t b = 0; b < n_blocks_per_row; b++) {
            uint8_t *blk = row_data + b * 34;
            uint16_t scale_bits;
            memcpy(&scale_bits, blk, 2);
            float scale = f16_to_f32(scale_bits);
            int8_t *quants = (int8_t *)(blk + 2);
            uint64_t base = b * 32;
            for (int k = 0; k < 32 && (base + k) < row_size; k++) {
                out[base + k] = scale * (float)quants[k];
            }
        }
        return;
    }

    if (t->type == GGML_TYPE_Q4_0) {
        // block = [f16 scale (2 bytes)] [16 bytes = 32 x 4-bit quants]
        for (uint64_t b = 0; b < n_blocks_per_row; b++) {
            uint8_t *blk = row_data + b * 18;
            uint16_t scale_bits;
            memcpy(&scale_bits, blk, 2);
            float scale = f16_to_f32(scale_bits);
            uint8_t *quants = blk + 2;
            uint64_t base = b * 32;
            for (int k = 0; k < 16 && (base + k * 2) < row_size; k++) {
                uint8_t byte = quants[k];
                int lo = (int)(byte & 0xF) - 8;
                int hi = (int)(byte >> 4)  - 8;
                out[base + k * 2]     = scale * (float)lo;
                out[base + k * 2 + 1] = scale * (float)hi;
            }
        }
        return;
    }

    fprintf(stderr, "GGUF: unsupported tensor type %d for dequantization\n", t->type);
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// Free

static void gguf_value_free(GGUFValue *v) {
    if (v->type == GGUF_TYPE_STRING) { free(v->str); }
    else if (v->type == GGUF_TYPE_ARRAY) {
        for (uint64_t i = 0; i < v->arr.count; i++) {
            gguf_value_free(&v->arr.items[i]);
        }
        free(v->arr.items);
    }
}

static void gguf_free(GGUFContext *ctx) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        free(ctx->kv[i].key);
        gguf_value_free(&ctx->kv[i].value);
    }
    free(ctx->kv);
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        free(ctx->tensors[i].name);
    }
    free(ctx->tensors);
    free(ctx->data);
    free(ctx);
}

#endif // GGUF_H
