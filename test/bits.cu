#include "llmc/bits.cuh"

#undef NDEBUG
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

float round_trip(float f, unsigned short threshold) {
    const SplitFloatResult split = split_float(f, threshold);
    const  float r = assemble_float(split.b_float, split.bits);
    return r;
}

bool match_floats(float f1, float f2) {
    const unsigned int u1 = bit_cast<unsigned int>(f1);
    const unsigned int u2 = bit_cast<unsigned int>(f2);
    if((u1 & (~1u)) != (u2 & (~1u))) {
        printf("MISMATCH: %0x %0x\n", u1, u2);
        return false;
    }
    return true;
}

#define ASSERT_ROUND_TRIP(f) \
    assert(match_floats(f, round_trip(f, 0))); \
    assert(match_floats(f, round_trip(f, 0xFFFF)));  \

int main() {
    ASSERT_ROUND_TRIP(1.4623f)
    ASSERT_ROUND_TRIP(-63623.9f)
    ASSERT_ROUND_TRIP(FLT_TRUE_MIN)
    ASSERT_ROUND_TRIP(NAN)
    ASSERT_ROUND_TRIP(0)
    ASSERT_ROUND_TRIP(INFINITY)
    // make sure we trigger the "rounding increases exponent" code path
    const float increment_exponent = bit_cast<float>((unsigned int)(0x40ff'fff0));
    ASSERT_ROUND_TRIP(increment_exponent)
    printf("PASS\n");
    return EXIT_SUCCESS;
}