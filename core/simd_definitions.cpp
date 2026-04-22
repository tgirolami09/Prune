#include "simd_definitions.hpp"

// SIMD utility functions
simd16 simd16_zero() {
    return ADDSIZE(ADDMM(setzero_si))();
}

simdint simdint_zero() {
    return simdint();
}

simd16 simd16_set1(dbyte value) {
    return ADDMM(set1_epi16)(value);
}

simdint simdint_set1(int value) {
    return ADDMM(set1_epi32)(value);
}

simdint simdint_add(const simdint& a, const simdint& b) {
    return ADDMM(add_epi32)(a, b);
}

simd16 simd16_sli(const simd16& a, int shift){
    return ADDMM(slli_epi16)(a, shift);
}

simd16 simd16_mulhi(const simd16& a, const simd16& b){
    return ADDMM(mulhi_epi16)(a, b);
}

simd16 simd16_mullo(const simd16& a, const simd16& b) {
    return ADDMM(mullo_epi16)(a, b);
}

simdint simdint_shr(const simdint& a, int b){
    return ADDMM(srli_epi32)(a, b);
}

simd16 simd16_shr(const simd16& a, int b){
    return ADDMM(srli_epi16)(a, b);
}

simd16 simd16_min(const simd16& a, const simd16& b) {
    return ADDMM(min_epi16)(a, b);
}
simdint simdint_min(const simdint& a, const simdint& b) {
    return ADDMM(min_epi32)(a, b);
}
simdint simdint_max(const simdint& a, const simdint& b) {
    return ADDMM(max_epi32)(a, b);
}

simd16 simd16_clamp(const simd16& value, const simd16& min_val, const simd16& max_val) {
    return ADDMM(min_epi16)(ADDMM(max_epi16)(value, min_val), max_val);
}
simd16 simd16_uclamp(const simd16& value, const simd16& min_val, const simd16& max_val) {
    return ADDMM(min_epu16)(ADDMM(max_epu16)(value, min_val), max_val);
}

simd16 simdint_clamp(const simdint& value, const simdint& min_val, const simdint& max_val) {
    return ADDMM(min_epi32)(ADDMM(max_epi32)(value, min_val), max_val);
}

simdint simdint_mullo(const simdint& a, const simdint& b) {
    return ADDMM(mullo_epi32)(a, b);
}

simdint mull_add(const simd16& a, const simd16& b){
    return ADDMM(madd_epi16)(a, b);
}

int mysum(const simdint& x){
#ifdef __AVX512F__
    return _mm512_reduce_add_epi32(x);
#elif defined(__AVX2__)
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

simd16 simd16_add(const simd16& a, const simd16& b) {
    return ADDMM(add_epi16)(a, b);
}

simd16 simd16_sub(const simd16& a, const simd16& b) {
    return ADDMM(sub_epi16)(a, b);
}
simd8 simd8_add(const simd8& a, const simd8& b) {
    return ADDMM(add_epi8)(a, b);
}

simd8 simd8_sub(const simd8& a, const simd8& b) {
    return ADDMM(sub_epi8)(a, b);
}
#ifndef __AVX2__
simd16 simdh8_16(const simdhalf& v){
    return ADDMM(cvtepi8_epi16)(_mm_set_epi64x(0, v));
}
#else
simd16 simdh8_16(const simdhalf& v){
    return ADDMM(cvtepi8_epi16)(v);
}
#endif