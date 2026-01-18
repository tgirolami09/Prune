#ifndef SIMD_DEFINITIONS_HPP
#define SIMD_DEFINITIONS_HPP

#if defined(__x86_64__)
    #include <immintrin.h>

#elif (defined(__arm__) || defined(__aarch64__)) && defined(__ARM_NEON__)
    // Make sure AVX2 is undefined
    #undef __AVX2__
    #pragma message("Warning: We do not support ARM SIMD instructions yet.")
    //#include <arm_neon.h>

#endif

#define dbyte int16_t
// Manual SIMD wrapper for cross-platform compatibility
#if defined(__AVX2__)
    using simd16 = __m256i;  // 16 int16_t values
    const int nb16 = 16;
    using simdint = __m256i;
    const int nbint = 16;  // 16 int32 values total
#elif defined(__SSE2__)
    using simd16 = __m128i;  // 8 int16_t values  
    const int nb16 = 8;
    // For SSE2, we need TWO __m128i to hold 8 int32 values
    using simdint = __m128i;
    const int nbint = 8;  // 8 int32 values total
#else
    #error "This code requires at least SSE2 support"
#endif

// SIMD utility functions
simd16 simd16_zero();
simdint simdint_zero();
simd16 simd16_set1(dbyte value);
simdint simdint_add(simdint a, simdint b);
simd16 simd16_mullo(simd16 a, simd16 b);
simd16 simd16_clamp(simd16 value, simd16 min_val, simd16 max_val);
simdint simdint_mullo(simdint a, simdint b);
simdint mull_add(simd16 a, simd16 b);
int mysum(simdint x);
simd16 simd16_add(simd16 a, simd16 b);
simd16 simd16_sub(simd16 a, simd16 b);
#endif