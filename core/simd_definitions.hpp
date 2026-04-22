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
//#define __AVX2__
#define dbyte int16_t
// Manual SIMD wrapper for cross-platform compatibility
#ifdef __AVX512F__
    using simd16 = __m512i;
    using simdint = __m512i;
    using simd8 = __m512i;
    using simdhalf = __m256i;
    #define MM _mm512
    #define SIZE 512
#elif defined(__AVX2__)
    using simd16 = __m256i;
    using simd8 = __m256i;
    using simdint = __m256i;
    using simdhalf = __m128i;
    #define MM _mm256
    #define SIZE 256
#elif defined(__SSE2__)
    using simd16 = __m128i;
    using simd8 = __m128i;
    using simdint = __m128i;
    using simdhalf = int64_t;
    #define MM _mm
    #define SIZE 128
#else
    #error "This code requires at least SSE2 support"
#endif

#if defined(__AVX512VNNI__) || defined(__AVX2_VNNI__)
#define VNNI
#pragma message("cpu support vnni instructions")
#endif


const int nb16 = sizeof(simd16)/sizeof(int16_t);
const int nb8 = sizeof(simd8)/sizeof(int8_t);
const int nbint = sizeof(simdint)/sizeof(int32_t);
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#define ADDMM(func_name) CONCAT(MM, _ ## func_name)
#define ADDSIZE(func_name) CONCAT(func_name, SIZE)

// SIMD utility functions
simd16 simd16_zero();
simdint simdint_zero();
simd16 simd16_set1(dbyte value);
simdint simdint_set1(int value);
simdint simdint_add(const simdint& a, const simdint& b);
simd16 simd16_mullo(const simd16& a, const simd16& b);
simd16 simd16_mulhi(const simd16& a, const simd16& b);
simd16 simd16_min(const simd16& a, const simd16& b);
simdint simdint_min(const simdint& a, const simdint& b);
simdint simdint_max(const simdint& a, const simdint& b);
simd16 simd16_clamp(const simd16& value, const simd16& min_val, const simd16& max_val);
simd16 simd16_uclamp(const simd16& value, const simd16& min_val, const simd16& max_val);
simdint simdint_clamp(const simdint& value, const simdint& min_val, const simdint& max_val);
simdint simdint_mullo(const simdint& a, const simdint& b);
simdint mull_add(const simd16& a, const simd16& b);
simdint simdint_shr(const simdint& a, int b);
simdint simd16_shr(const simdint& a, int b);
simd16 simd16_sli(const simd16& a, int shift);
int mysum(const simdint& x);
simd16 simd16_add(const simd16& a, const simd16& b);
simd16 simd16_sub(const simd16& a, const simd16& b);
simd8 simd8_add(const simd8& a, const simd8& b);
simd8 simd8_sub(const simd8& a, const simd8& b);
simd16 simdh8_16(const simdhalf& v);
#endif
