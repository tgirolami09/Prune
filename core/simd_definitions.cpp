#include "simd_definitions.hpp"

// SIMD utility functions
simd16 simd16_zero() {
#if defined(__AVX2__)
    return _mm256_setzero_si256();
#else
    return _mm_setzero_si128();
#endif
}

simdint simdint_zero() {
    return simdint();
}

simd16 simd16_set1(dbyte value) {
#if defined(__AVX2__)
    return _mm256_set1_epi16(value);
#else
    return _mm_set1_epi16(value);
#endif
}

simdint simdint_add(simdint a, simdint b) {
#if defined(__AVX2__)
    return _mm256_add_epi32(a, b);
#else
    return _mm_add_epi32(a, b);
#endif
}

simd16 simd16_mullo(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_mullo_epi16(a, b);
#else
    return _mm_mullo_epi16(a, b);
#endif
}

simd16 simd16_clamp(simd16 value, simd16 min_val, simd16 max_val) {
#if defined(__AVX2__)
    return _mm256_min_epi16(_mm256_max_epi16(value, min_val), max_val);
#else
    return _mm_min_epi16(_mm_max_epi16(value, min_val), max_val);
#endif
}

simdint simdint_mullo(simdint a, simdint b) {
#if defined(__AVX2__)
    return _mm256_mullo_epi32(a, b);
#else
    return _mm_mullo_epi32(a, b);
#endif
}

simdint mull_add(simd16 a, simd16 b){
#ifdef __AVX2__
    return _mm256_madd_epi16(a, b);
#else
    return _mm_madd_epi16(a, b);
#endif
}

int mysum(simdint x){
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

simd16 simd16_add(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_add_epi16(a, b);
#else
    return _mm_add_epi16(a, b);
#endif
}

simd16 simd16_sub(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_sub_epi16(a, b);
#else
    return _mm_sub_epi16(a, b);
#endif
}