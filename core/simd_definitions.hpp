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
inline simd16 simd16_zero() {
#if defined(__AVX2__)
    return _mm256_setzero_si256();
#else
    return _mm_setzero_si128();
#endif
}

inline simdint simdint_zero() {
    return simdint();
}

inline simd16 simd16_set1(dbyte value) {
#if defined(__AVX2__)
    return _mm256_set1_epi16(value);
#else
    return _mm_set1_epi16(value);
#endif
}

inline simdint simdint_add(simdint a, simdint b) {
#if defined(__AVX2__)
    return _mm256_add_epi32(a, b);
#else
    return _mm_add_epi32(a, b);
#endif
}

inline simd16 simd16_mullo(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_mullo_epi16(a, b);
#else
    return _mm_mullo_epi16(a, b);
#endif
}

inline simd16 simd16_clamp(simd16 value, simd16 min_val, simd16 max_val) {
#if defined(__AVX2__)
    return _mm256_min_epi16(_mm256_max_epi16(value, min_val), max_val);
#else
    return _mm_min_epi16(_mm_max_epi16(value, min_val), max_val);
#endif
}

inline simdint simdint_mullo(simdint a, simdint b) {
#if defined(__AVX2__)
    return _mm256_mullo_epi32(a, b);
#else
    return _mm_mullo_epi32(a, b);
#endif
}

inline simdint mull_add(simd16 a, simd16 b){
#ifdef __AVX2__
    return _mm256_madd_epi16(a, b);
#else
    return _mm_madd_epi16(a, b);
#endif
}

inline int mysum(simdint x){
#if defined(__AVX2__)
    const auto high128 = _mm256_extracti128_si256(x, 1);
    const auto low128 = _mm256_castsi256_si128(x);

    const auto sum128 = _mm_add_epi32(high128, low128);

    const auto high64 = _mm_unpackhi_epi64(sum128, sum128);
    const auto sum64 = _mm_add_epi32(sum128, high64);

    const auto high32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    const auto sum32 = _mm_add_epi32(sum64, high32);

    return _mm_cvtsi128_si32(sum32);
#else
    // Sum all 8 int32 values from two SSE registers
    const auto high64 = _mm_unpackhi_epi64(x, x);
    const auto sum64 = _mm_add_epi32(x, high64);

    const auto high32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    const auto sum32 = _mm_add_epi32(sum64, high32);

    return _mm_cvtsi128_si32(sum32);
#endif
}

inline simd16 simd16_add(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_add_epi16(a, b);
#else
    return _mm_add_epi16(a, b);
#endif
}

inline simd16 simd16_sub(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_sub_epi16(a, b);
#else
    return _mm_sub_epi16(a, b);
#endif
}
#endif