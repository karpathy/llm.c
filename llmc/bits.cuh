/*
Utilities for manipulating at the bit level
*/
#ifndef LLMC_BITS_CUH
#define LLMC_BITS_CUH

#include "cuda_bf16.h"

// implementation of unreachable from C++23/C23 that works across compilers
[[noreturn]] __host__ __device__ inline void unreachable()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}

// ----------------------------------------------------------------------------
//  bit-fiddling to reconstruct master weights from BF16 and missing bits
// handling bf16 is _almost_ just storing the 16 bits of significant that get
// truncated when going from float -> bf16; except we do stochastic rounding.
// if we end up rounding towards zero, we could just keep the bits, but if we
// round away from zero, there is a chance that the other bits of the bf16 no
// longer correspond to the bits in the original float. So we have to reserve
// one bit to remember whether we rounded up or down, for an effective master
// weight precision of fp31. That should still be more than sufficient.

// Result of splitting a float into a stochastically-rounded bfloat16 and
// additional reconstruction bits
struct SplitFloatResult {
    nv_bfloat16 b_float;
    unsigned short bits;
};

// UB-free bit-level conversions. A C++17 version for C++20 std::bit_cast
template<class T, class S>
__host__ __device__ T bit_cast(S v) {
    T dest;
    static_assert(sizeof(v) == sizeof(dest), "Size mismatch.");
    memcpy(&dest, &v, sizeof(v));
    return dest;
}

// Splits a float into a bfloat16 and the remaining significant bits
__host__ __device__ SplitFloatResult split_float(float value, unsigned short threshold) {
    unsigned int float_bits = bit_cast<unsigned int>(value);
    // IEEE 754: float: S E(8) M (23)    bfloat: same, but significant 23-16 = 7 bits
    // ideally, we'd just store the cut-off 16 bits, but that doesn't work if rounding
    // is involved.
    unsigned int rounded_bits = float_bits & 0x0000FFFFu;
    if(rounded_bits > threshold) {
        SplitFloatResult result;
        result.b_float = __float2bfloat16_rn(bit_cast<float>(float_bits | 0xFFFFu));
        result.bits = rounded_bits & (~1u) | 1u;
        return result;
    } else {
        // truncation is easy
        SplitFloatResult result;
        result.b_float = bit_cast<__nv_bfloat16>((unsigned short)(float_bits >> 16u));
        result.bits = rounded_bits & (~1u);
        return result;
    }
}

// Reassembles a float from the bfloat16 part and the missing mantissa
__host__ __device__ float assemble_float(nv_bfloat16 b_float, unsigned short bits) {
    constexpr const unsigned short BF16_SIGN_MASK        = 0b1'00000000'0000000u;
    constexpr const unsigned short BF16_EXPONENT_MASK    = 0b0'11111111'0000000u;
    constexpr const unsigned short BF16_SIGNIFICANT_MASK = 0b0'00000000'1111111u;
    unsigned short bf = bit_cast<unsigned short>(b_float);
    if(bits & 1u) {
        // if we rounded away from zero, we need to undo these changes.
        // first, check if the significant (7 bits) of bf16 is zero
        const unsigned short significant = bf & BF16_SIGNIFICANT_MASK;
        if(significant == 0) {
            // significant overflowed, need to decrement the exponent
            const unsigned short exponent = (bf & BF16_EXPONENT_MASK) >> 7u;
            if(exponent == 0) {
                // zero, cannot be reached if we round away from zero
                unreachable();
            }
            // decrement the exponent and set significant to all-ones
            bf = (bf & BF16_SIGN_MASK) | ((exponent-1u) << 7u) | BF16_SIGNIFICANT_MASK;
        } else {
            // significant was incremented, decrement
            bf = (bf & (BF16_SIGN_MASK | BF16_EXPONENT_MASK)) | (significant - 1u);
        }
    }
    const unsigned int result = (bits & (~1u)) | (bf << 16u);
    return bit_cast<float>(result);
}

#endif  // LLMC_BITS_CUH
