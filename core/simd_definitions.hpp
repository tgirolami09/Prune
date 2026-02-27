#ifndef SIMD_DEFINITIONS_HPP
#define SIMD_DEFINITIONS_HPP
#include <cstdint>
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
#ifdef __AVX512F__
    using simd16 = __m512i;
    using simdint = __m512i;
    #define MM _mm512
    #define SIZE 512
#elif defined(__AVX2__)
    using simd16 = __m256i; 
    using simdint = __m256i;
    #define MM _mm256
    #define SIZE 256
#elif defined(__SSE2__)
    using simd16 = __m128i;
    using simdint = __m128i;
    #define MM _mm
    #define SIZE 128
#else
    #error "This code requires at least SSE2 support"
#endif
const int nb16 = sizeof(simd16)/sizeof(int16_t);
const int nbint = sizeof(simdint)/sizeof(int32_t);
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#define ADDMM(func_name) CONCAT(MM, _ ## func_name)
#define ADDSIZE(func_name) CONCAT(func_name, SIZE)

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
