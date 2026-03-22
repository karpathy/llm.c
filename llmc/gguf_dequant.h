/*
Extensible GGUF dequantization module.

Provides a registry-based system for dequantizing quantized tensor rows.
Supports the base types (F32, F16, Q4_0, Q4_1, Q8_0) from gguf.h plus
k-quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) and legacy Q5_0/Q5_1.

Usage:
  1. Call gguf_dequant_init() once at startup to register all built-in types.
  2. Use gguf_dequant_row() as a drop-in replacement for gguf_dequantize_row().
  3. To add custom types: implement a DequantRowFn and call gguf_dequant_register().

This header extends (not replaces) gguf.h — it depends on GGUFTensor, GGMLType,
and f16_to_f32() from that header.
*/
#ifndef GGUF_DEQUANT_H
#define GGUF_DEQUANT_H

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ============================================================================
// Extended type enum values (must match GGML canonical IDs)
//
// Base types already in gguf.h:
//   GGML_TYPE_F32  = 0
//   GGML_TYPE_F16  = 1
//   GGML_TYPE_Q4_0 = 2
//   GGML_TYPE_Q4_1 = 3
//   GGML_TYPE_Q8_0 = 8
//
// Extended types:

#ifndef GGML_TYPE_Q5_0
#define GGML_TYPE_Q5_0  6
#endif
#ifndef GGML_TYPE_Q5_1
#define GGML_TYPE_Q5_1  7
#endif
#ifndef GGML_TYPE_Q2_K
#define GGML_TYPE_Q2_K  10
#endif
#ifndef GGML_TYPE_Q3_K
#define GGML_TYPE_Q3_K  11
#endif
#ifndef GGML_TYPE_Q4_K
#define GGML_TYPE_Q4_K  12
#endif
#ifndef GGML_TYPE_Q5_K
#define GGML_TYPE_Q5_K  13
#endif
#ifndef GGML_TYPE_Q6_K
#define GGML_TYPE_Q6_K  14
#endif

// ============================================================================
// Dequantization function signature
//
// Dequantizes a single row of a tensor to float32.
//   tensor:  pointer to the GGUFTensor (for accessing raw data + dims)
//   row:     which row to dequantize
//   out:     output float buffer (must have space for dims[0] floats)

typedef void (*DequantRowFn)(GGUFTensor *tensor, int64_t row, float *out);

// ============================================================================
// Registry
//
// Maps GGMLType -> { block_size, type_size_bytes, dequant_fn }
// Max 32 types should be plenty. Increase if needed.

#define DEQUANT_MAX_TYPES 32

typedef struct {
    int         block_size;     // elements per block
    int         type_size;      // bytes per block
    DequantRowFn dequant_fn;    // dequantization function (NULL = unsupported)
    const char  *name;          // human-readable name
} DequantTypeInfo;

typedef struct {
    DequantTypeInfo types[DEQUANT_MAX_TYPES];
    int initialized;
} DequantRegistry;

// Global registry (static, single-translation-unit)
static DequantRegistry g_dequant_registry = {0};

// ============================================================================
// Registry API

static void gguf_dequant_register(int type_id, int block_size, int type_size,
                                   DequantRowFn fn, const char *name) {
    if (type_id < 0 || type_id >= DEQUANT_MAX_TYPES) {
        fprintf(stderr, "gguf_dequant: type_id %d out of range [0, %d)\n",
                type_id, DEQUANT_MAX_TYPES);
        return;
    }
    DequantTypeInfo *info = &g_dequant_registry.types[type_id];
    info->block_size = block_size;
    info->type_size  = type_size;
    info->dequant_fn = fn;
    info->name       = name;
}

// ============================================================================
// Built-in dequantization implementations

// --- F32 ---
static void dequant_f32_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    float *p = (float *)t->data + row * row_size;
    memcpy(out, p, row_size * sizeof(float));
}

// --- F16 ---
static void dequant_f16_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint16_t *p = (uint16_t *)t->data + row * row_size;
    for (uint64_t i = 0; i < row_size; i++) {
        out[i] = f16_to_f32(p[i]);
    }
}

// --- Q4_0 ---
// Block: [f16 scale (2B)] [16B = 32 × 4-bit quants], total 18B per 32 elements
static void dequant_q4_0_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 31) / 32;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 18;

    for (uint64_t b = 0; b < n_blocks; b++) {
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
            if (base + k * 2 + 1 < row_size)
                out[base + k * 2 + 1] = scale * (float)hi;
        }
    }
}

// --- Q4_1 ---
// Block: [f16 scale (2B)] [f16 min (2B)] [16B = 32 × 4-bit quants], total 20B per 32 elements
static void dequant_q4_1_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 31) / 32;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 20;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 20;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&m_bits, blk + 2, 2);
        float d = f16_to_f32(d_bits);
        float m = f16_to_f32(m_bits);
        uint8_t *quants = blk + 4;
        uint64_t base = b * 32;
        for (int k = 0; k < 16 && (base + k * 2) < row_size; k++) {
            uint8_t byte = quants[k];
            out[base + k * 2]     = d * (float)(byte & 0xF) + m;
            if (base + k * 2 + 1 < row_size)
                out[base + k * 2 + 1] = d * (float)(byte >> 4) + m;
        }
    }
}

// --- Q5_0 ---
// Block: [f16 scale (2B)] [4B high-bits mask] [16B = 32 × 4-bit low quants], total 22B per 32 elements
static void dequant_q5_0_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 31) / 32;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 22;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 22;
        uint16_t scale_bits;
        memcpy(&scale_bits, blk, 2);
        float scale = f16_to_f32(scale_bits);
        uint32_t qh;
        memcpy(&qh, blk + 2, 4);
        uint8_t *ql = blk + 6;
        uint64_t base = b * 32;
        for (int k = 0; k < 16 && (base + k * 2) < row_size; k++) {
            uint8_t byte = ql[k];
            int lo_4 = (int)(byte & 0xF);
            int hi_4 = (int)(byte >> 4);
            // 5th bit from qh
            int lo_5 = (qh >> (k * 2))     & 1;
            int hi_5 = (qh >> (k * 2 + 1)) & 1;
            int lo = lo_4 | (lo_5 << 4);
            int hi = hi_4 | (hi_5 << 4);
            out[base + k * 2]     = scale * ((float)lo - 16.0f);
            if (base + k * 2 + 1 < row_size)
                out[base + k * 2 + 1] = scale * ((float)hi - 16.0f);
        }
    }
}

// --- Q5_1 ---
// Block: [f16 d (2B)] [f16 m (2B)] [4B high-bits mask] [16B low quants], total 24B per 32 elements
static void dequant_q5_1_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 31) / 32;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 24;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 24;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&m_bits, blk + 2, 2);
        float d = f16_to_f32(d_bits);
        float m = f16_to_f32(m_bits);
        uint32_t qh;
        memcpy(&qh, blk + 4, 4);
        uint8_t *ql = blk + 8;
        uint64_t base = b * 32;
        for (int k = 0; k < 16 && (base + k * 2) < row_size; k++) {
            uint8_t byte = ql[k];
            int lo_4 = (int)(byte & 0xF);
            int hi_4 = (int)(byte >> 4);
            int lo_5 = (qh >> (k * 2))     & 1;
            int hi_5 = (qh >> (k * 2 + 1)) & 1;
            int lo = lo_4 | (lo_5 << 4);
            int hi = hi_4 | (hi_5 << 4);
            out[base + k * 2]     = d * (float)lo + m;
            if (base + k * 2 + 1 < row_size)
                out[base + k * 2 + 1] = d * (float)hi + m;
        }
    }
}

// --- Q8_0 ---
// Block: [f16 scale (2B)] [32 × int8 quants (32B)], total 34B per 32 elements
static void dequant_q8_0_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 31) / 32;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 34;

    for (uint64_t b = 0; b < n_blocks; b++) {
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
}

// ============================================================================
// K-Quant implementations
//
// K-quants use "super blocks" of 256 elements, subdivided into sub-blocks.
// Each super block has its own scale/min structure.

// --- Q4_K ---
// Super block: 256 elements
// Layout: [f16 d (2B)] [f16 dmin (2B)] [12B scales] [128B quants (256 × 4-bit)]
// Total: 144 bytes per 256 elements
//
// The 12 bytes of scales encode 8 pairs of (scale, min), 6 bits each.
// First 4 bytes: low 4 bits of scales[0..7] (nybbles)
// Next 4 bytes: low 4 bits of mins[0..7] (nybbles)
// Next 4 bytes: high 2 bits of scales[0..3] | high 2 bits of mins[0..3] |
//               high 2 bits of scales[4..7] | high 2 bits of mins[4..7]

static void dequant_q4_k_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 255) / 256;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 144;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 144;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d    = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);

        const uint8_t *scales_raw = blk + 4;  // 12 bytes of packed scales/mins
        const uint8_t *qs = blk + 16;         // 128 bytes of 4-bit quants

        // Unpack 8 scales and 8 mins (6 bits each)
        uint8_t sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i]     = (scales_raw[i] & 0xF) | (((scales_raw[8 + i] >> 0) & 3) << 4);
            sc[i + 4] = (scales_raw[i] >> 4)  | (((scales_raw[8 + i] >> 2) & 3) << 4);  // Correction: see below
            mn[i]     = (scales_raw[4 + i] & 0xF) | (((scales_raw[8 + i] >> 4) & 3) << 4);
            mn[i + 4] = (scales_raw[4 + i] >> 4)  | (((scales_raw[8 + i] >> 6) & 3) << 4);
        }

        // Wait — the packing for Q4_K is actually:
        // bytes 0..3:  low 6 bits → scales[0..3] low 4 bits in low nybble,
        //                           scales[4..7] low 4 bits in high nybble
        // Actually let me use the canonical ggml unpacking.
        //
        // The canonical layout from ggml:
        //   scales_raw[0..3]: low nybble = scale_low[0..3], high nybble = scale_low[4..7]
        //   scales_raw[4..7]: low nybble = min_low[0..3],   high nybble = min_low[4..7]
        //   scales_raw[8..11]: packed high bits
        //     byte 8:  bits 0-1 = scale_hi[0], bits 2-3 = scale_hi[1], bits 4-5 = min_hi[0], bits 6-7 = min_hi[1]
        //     byte 9:  bits 0-1 = scale_hi[2], bits 2-3 = scale_hi[3], bits 4-5 = min_hi[2], bits 6-7 = min_hi[3]
        //     byte 10: bits 0-1 = scale_hi[4], bits 2-3 = scale_hi[5], bits 4-5 = min_hi[4], bits 6-7 = min_hi[5]
        //     byte 11: bits 0-1 = scale_hi[6], bits 2-3 = scale_hi[7], bits 4-5 = min_hi[6], bits 6-7 = min_hi[7]
        //
        // Rewrite:
        for (int i = 0; i < 4; i++) {
            sc[i]     = (scales_raw[i] & 0xF);
            sc[i + 4] = (scales_raw[i] >> 4);
            mn[i]     = (scales_raw[4 + i] & 0xF);
            mn[i + 4] = (scales_raw[4 + i] >> 4);
        }
        // Add high bits from bytes 8..11
        for (int i = 0; i < 4; i++) {
            uint8_t hb = scales_raw[8 + i];
            int j = i * 2;
            sc[j]     |= ((hb >> 0) & 3) << 4;
            sc[j + 1] |= ((hb >> 2) & 3) << 4;
            mn[j]     |= ((hb >> 4) & 3) << 4;
            mn[j + 1] |= ((hb >> 6) & 3) << 4;
        }

        uint64_t base = b * 256;
        for (int sub = 0; sub < 8; sub++) {
            float sub_d   = d * (float)sc[sub];
            float sub_min = dmin * (float)mn[sub];
            const uint8_t *q = qs + sub * 16;  // 16 bytes = 32 nybbles = 32 elements
            uint64_t sb = base + (uint64_t)sub * 32;
            for (int k = 0; k < 16 && (sb + k * 2) < row_size; k++) {
                uint8_t byte = q[k];
                out[sb + k * 2]     = sub_d * (float)(byte & 0xF) - sub_min;
                if (sb + k * 2 + 1 < row_size)
                    out[sb + k * 2 + 1] = sub_d * (float)(byte >> 4) - sub_min;
            }
        }
    }
}

// --- Q6_K ---
// Super block: 256 elements
// Layout: [128B low quants (ql)] [64B high quants (qh)] [16B scales (int8)] [f16 d (2B)]
// Total: 210 bytes per 256 elements
//
// Each element is 6 bits: 4 low bits from ql, 2 high bits from qh.
// scales: 16 × int8 scales, one per 16 elements.

static void dequant_q6_k_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 255) / 256;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 210;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 210;

        const uint8_t *ql = blk;           // 128 bytes: low 4 bits, packed as nybbles
        const uint8_t *qh = blk + 128;     // 64 bytes: high 2 bits
        const int8_t  *sc = (const int8_t *)(blk + 192);  // 16 × int8 scales
        uint16_t d_bits;
        memcpy(&d_bits, blk + 208, 2);
        float d = f16_to_f32(d_bits);

        uint64_t base = b * 256;

        // Process in two halves of 128 elements each
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;   // 64 bytes → 128 nybbles → 128 elements
            const uint8_t *qh_h = qh + half * 32;   // 32 bytes → 128 × 2-bit
            const int8_t  *sc_h = sc + half * 8;     // 8 scales for this half

            for (int sub = 0; sub < 8; sub++) {
                float scale = d * (float)sc_h[sub];
                int elem_off = half * 128 + sub * 16;

                for (int k = 0; k < 16; k++) {
                    uint64_t idx = base + (uint64_t)elem_off + k;
                    if (idx >= row_size) break;

                    // Low 4 bits: ql_h has 64 bytes for 128 elements (nybble-packed)
                    int ql_idx = sub * 8 + k / 2;      // which byte in ql_h
                    int lo4;
                    if (k % 2 == 0)
                        lo4 = ql_h[ql_idx] & 0xF;
                    else
                        lo4 = ql_h[ql_idx] >> 4;

                    // High 2 bits: qh_h has 32 bytes for 128 elements (2 bits each)
                    int qh_bit_idx = sub * 16 + k;     // element index within this half
                    int qh_byte = qh_bit_idx / 4;
                    int qh_shift = (qh_bit_idx % 4) * 2;
                    int hi2 = (qh_h[qh_byte] >> qh_shift) & 3;

                    int val = lo4 | (hi2 << 4);         // 6-bit value [0, 63]
                    out[idx] = scale * ((float)val - 32.0f);
                }
            }
        }
    }
}

// --- Q2_K ---
// Super block: 256 elements
// Layout: [16B scales (4-bit packed)] [16B mins (4-bit packed)] [64B quants (2-bit)] [f16 d (2B)] [f16 dmin (2B)]
// Total: 100 bytes (actually the layout is slightly different — let me use the canonical one)
//
// Canonical ggml Q2_K block (256 elements, 84 bytes):
//   [32B scales+mins (4 bits each)] [64B quants] [f16 d (2B)] [f16 dmin (2B)]
//   Total: 32 + 64 + 2 + 2 = 100? No — the canonical size is 84.
//
// Actually from ggml source:
//   struct block_q2_K {
//       uint8_t scales[16]; // 16 bytes: 4-bit scale and 4-bit min for each sub-block (16 sub-blocks)
//       uint8_t qs[64];     // 64 bytes: 256 × 2-bit quants
//       ggml_half d;        // 2 bytes: super scale
//       ggml_half dmin;     // 2 bytes: super min
//   }; // total: 84 bytes

static void dequant_q2_k_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 255) / 256;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 84;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 84;

        const uint8_t *scales_raw = blk;        // 16 bytes
        const uint8_t *qs         = blk + 16;   // 64 bytes
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk + 80, 2);
        memcpy(&dmin_bits, blk + 82, 2);
        float d    = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);

        uint64_t base = b * 256;

        // 16 sub-blocks of 16 elements each
        for (int sub = 0; sub < 16; sub++) {
            uint8_t sb = scales_raw[sub];
            float sub_d   = d * (float)(sb & 0xF);
            float sub_min = dmin * (float)(sb >> 4);

            // Each element is 2 bits in qs
            // sub-block `sub` covers elements [sub*16 .. sub*16+15]
            for (int k = 0; k < 16; k++) {
                uint64_t idx = base + (uint64_t)sub * 16 + k;
                if (idx >= row_size) break;
                int bit_idx = sub * 16 + k;
                int byte_idx = bit_idx / 4;
                int shift = (bit_idx % 4) * 2;
                int q = (qs[byte_idx] >> shift) & 3;
                out[idx] = sub_d * (float)q - sub_min;
            }
        }
    }
}

// --- Q3_K ---
// Super block: 256 elements
// From ggml:
//   struct block_q3_K {
//       uint8_t hmask[32];   // 32 bytes: high bits mask
//       uint8_t qs[64];      // 64 bytes: low 2 bits of quants
//       uint8_t scales[12];  // 12 bytes: packed scales
//       ggml_half d;         // 2 bytes
//   }; // total: 110 bytes

static void dequant_q3_k_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 255) / 256;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 110;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 110;

        const uint8_t *hmask      = blk;           // 32 bytes
        const uint8_t *qs         = blk + 32;      // 64 bytes
        const uint8_t *scales_raw = blk + 96;      // 12 bytes
        uint16_t d_bits;
        memcpy(&d_bits, blk + 108, 2);
        float d = f16_to_f32(d_bits);

        // Unpack 16 scales from 12 bytes (like Q4_K but for 16 sub-blocks of 16 elements)
        // Actually Q3_K uses a different scale packing:
        //   scales_raw[0..3]:  low 4 bits → scales[0..7] (nybble-packed, 4 bits each)
        //   scales_raw[4..7]:  low 4 bits → scales[8..15]
        //   scales_raw[8..11]: high 2 bits packed
        //
        // The canonical unpacking from ggml:
        int8_t sc[16];
        // Low bits
        for (int i = 0; i < 8; i++) {
            int raw_idx = i / 2;
            if (i % 2 == 0)
                sc[i] = (int)(scales_raw[raw_idx] & 0xF);
            else
                sc[i] = (int)(scales_raw[raw_idx] >> 4);
        }
        for (int i = 0; i < 8; i++) {
            int raw_idx = 4 + i / 2;
            if (i % 2 == 0)
                sc[8 + i] = (int)(scales_raw[raw_idx] & 0xF);
            else
                sc[8 + i] = (int)(scales_raw[raw_idx] >> 4);
        }
        // High bits from bytes 8..11
        for (int i = 0; i < 4; i++) {
            uint8_t hb = scales_raw[8 + i];
            sc[i * 4 + 0] |= ((hb >> 0) & 3) << 4;
            sc[i * 4 + 1] |= ((hb >> 2) & 3) << 4;
            sc[i * 4 + 2] |= ((hb >> 4) & 3) << 4;
            sc[i * 4 + 3] |= ((hb >> 6) & 3) << 4;
        }
        // Scales are 6-bit signed values; subtract 32 to center
        for (int i = 0; i < 16; i++) {
            sc[i] -= 32;
        }

        uint64_t base = b * 256;

        // 16 sub-blocks of 16 elements each
        for (int sub = 0; sub < 16; sub++) {
            float scale = d * (float)sc[sub];

            for (int k = 0; k < 16; k++) {
                uint64_t idx = base + (uint64_t)sub * 16 + k;
                if (idx >= row_size) break;

                int elem = sub * 16 + k;

                // Low 2 bits from qs
                int qs_byte = elem / 4;
                int qs_shift = (elem % 4) * 2;
                int lo2 = (qs[qs_byte] >> qs_shift) & 3;

                // High bit from hmask
                int hm_byte = elem / 8;
                int hm_bit = elem % 8;
                int hi1 = (hmask[hm_byte] >> hm_bit) & 1;

                int q = lo2 | (hi1 << 2);  // 3-bit value [0, 7]
                out[idx] = scale * ((float)q - 4.0f);
            }
        }
    }
}

// --- Q5_K ---
// Super block: 256 elements
// From ggml:
//   struct block_q5_K {
//       ggml_half d;          // 2 bytes: super scale
//       ggml_half dmin;       // 2 bytes: super min
//       uint8_t scales[12];   // 12 bytes: packed scales/mins (6 bits each, like Q4_K)
//       uint8_t qh[32];       // 32 bytes: high bits (1 per element = 256 bits)
//       uint8_t qs[128];      // 128 bytes: low 4 bits (nybble packed)
//   }; // total: 176 bytes

static void dequant_q5_k_row(GGUFTensor *t, int64_t row, float *out) {
    uint64_t row_size = t->dims[0];
    uint64_t n_blocks = (row_size + 255) / 256;
    uint8_t *row_data = (uint8_t *)t->data + row * n_blocks * 176;

    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = row_data + b * 176;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d    = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);

        const uint8_t *scales_raw = blk + 4;    // 12 bytes
        const uint8_t *qh         = blk + 16;   // 32 bytes
        const uint8_t *ql         = blk + 48;   // 128 bytes

        // Unpack scales/mins (same format as Q4_K: 8 scales + 8 mins, 6 bits each)
        uint8_t sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i]     = (scales_raw[i] & 0xF);
            sc[i + 4] = (scales_raw[i] >> 4);
            mn[i]     = (scales_raw[4 + i] & 0xF);
            mn[i + 4] = (scales_raw[4 + i] >> 4);
        }
        for (int i = 0; i < 4; i++) {
            uint8_t hb = scales_raw[8 + i];
            int j = i * 2;
            sc[j]     |= ((hb >> 0) & 3) << 4;
            sc[j + 1] |= ((hb >> 2) & 3) << 4;
            mn[j]     |= ((hb >> 4) & 3) << 4;
            mn[j + 1] |= ((hb >> 6) & 3) << 4;
        }

        uint64_t base = b * 256;

        // 8 sub-blocks of 32 elements each
        for (int sub = 0; sub < 8; sub++) {
            float sub_d   = d * (float)sc[sub];
            float sub_min = dmin * (float)mn[sub];

            for (int k = 0; k < 32; k++) {
                uint64_t idx = base + (uint64_t)sub * 32 + k;
                if (idx >= row_size) break;

                int elem = sub * 32 + k;

                // Low 4 bits from ql (nybble packed: 128 bytes = 256 nybbles)
                int ql_byte = elem / 2;
                int lo4;
                if (elem % 2 == 0)
                    lo4 = ql[ql_byte] & 0xF;
                else
                    lo4 = ql[ql_byte] >> 4;

                // High bit from qh (1 bit per element: 32 bytes = 256 bits)
                int qh_byte = elem / 8;
                int qh_bit  = elem % 8;
                int hi1 = (qh[qh_byte] >> qh_bit) & 1;

                int q = lo4 | (hi1 << 4);  // 5-bit value [0, 31]
                out[idx] = sub_d * (float)q - sub_min;
            }
        }
    }
}

// ============================================================================
// Registry initialization

static void gguf_dequant_init(void) {
    if (g_dequant_registry.initialized) return;

    // Base types
    gguf_dequant_register(GGML_TYPE_F32,  1,  4, dequant_f32_row,  "F32");
    gguf_dequant_register(GGML_TYPE_F16,  1,  2, dequant_f16_row,  "F16");
    gguf_dequant_register(GGML_TYPE_Q4_0, 32, 18, dequant_q4_0_row, "Q4_0");
    gguf_dequant_register(GGML_TYPE_Q4_1, 32, 20, dequant_q4_1_row, "Q4_1");
    gguf_dequant_register(GGML_TYPE_Q5_0, 32, 22, dequant_q5_0_row, "Q5_0");
    gguf_dequant_register(GGML_TYPE_Q5_1, 32, 24, dequant_q5_1_row, "Q5_1");
    gguf_dequant_register(GGML_TYPE_Q8_0, 32, 34, dequant_q8_0_row, "Q8_0");

    // K-quant types (super blocks of 256 elements)
    gguf_dequant_register(GGML_TYPE_Q2_K, 256,  84, dequant_q2_k_row, "Q2_K");
    gguf_dequant_register(GGML_TYPE_Q3_K, 256, 110, dequant_q3_k_row, "Q3_K");
    gguf_dequant_register(GGML_TYPE_Q4_K, 256, 144, dequant_q4_k_row, "Q4_K");
    gguf_dequant_register(GGML_TYPE_Q5_K, 256, 176, dequant_q5_k_row, "Q5_K");
    gguf_dequant_register(GGML_TYPE_Q6_K, 256, 210, dequant_q6_k_row, "Q6_K");

    g_dequant_registry.initialized = 1;
}

// ============================================================================
// Main dequantization entry point (drop-in replacement for gguf_dequantize_row)

static void gguf_dequant_row(GGUFTensor *t, int64_t row, float *out) {
    if (!g_dequant_registry.initialized) {
        gguf_dequant_init();
    }

    int type_id = (int)t->type;
    if (type_id < 0 || type_id >= DEQUANT_MAX_TYPES ||
        !g_dequant_registry.types[type_id].dequant_fn) {
        fprintf(stderr, "gguf_dequant: unsupported tensor type %d for '%s'\n",
                type_id, t->name ? t->name : "?");
        fprintf(stderr, "  Register a handler with gguf_dequant_register(%d, ...)\n", type_id);
        exit(EXIT_FAILURE);
    }

    g_dequant_registry.types[type_id].dequant_fn(t, row, out);
}

// ============================================================================
// Helper: get block size for a type (needed for data size calculations)

static int gguf_dequant_block_size(int type_id) {
    if (!g_dequant_registry.initialized) gguf_dequant_init();
    if (type_id < 0 || type_id >= DEQUANT_MAX_TYPES) return 0;
    return g_dequant_registry.types[type_id].block_size;
}

static int gguf_dequant_type_size(int type_id) {
    if (!g_dequant_registry.initialized) gguf_dequant_init();
    if (type_id < 0 || type_id >= DEQUANT_MAX_TYPES) return 0;
    return g_dequant_registry.types[type_id].type_size;
}

static const char *gguf_dequant_type_name(int type_id) {
    if (!g_dequant_registry.initialized) gguf_dequant_init();
    if (type_id < 0 || type_id >= DEQUANT_MAX_TYPES) return "unknown";
    const char *n = g_dequant_registry.types[type_id].name;
    return n ? n : "unknown";
}

#endif // GGUF_DEQUANT_H
