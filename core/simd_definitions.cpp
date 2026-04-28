#include "simd_definitions.hpp"

// SIMD utility functions
simd<32> simdint_add(const simd<32>& a, const simd<32>& b) {
    return ADDMM(add_epi32)(a, b);
}

simd<16> simd16_sli(const simd<16>& a, int shift){
    return ADDMM(slli_epi16)(a, shift);
}

simd<16> simd16_mulhi(const simd<16>& a, const simd<16>& b){
    return ADDMM(mulhi_epi16)(a, b);
}

simd<16> simd16_mullo(const simd<16>& a, const simd<16>& b) {
    return ADDMM(mullo_epi16)(a, b);
}

simd<32> simdint_shr(const simd<32>& a, int b){
    return ADDMM(srli_epi32)(a, b);
}

simd<16> simd16_shr(const simd<16>& a, int b){
    return ADDMM(srli_epi16)(a, b);
}

simd<16> simd16_min(const simd<16>& a, const simd<16>& b) {
    return ADDMM(min_epi16)(a, b);
}
simd<32> simdint_min(const simd<32>& a, const simd<32>& b) {
    return ADDMM(min_epi32)(a, b);
}
simd<32> simdint_max(const simd<32>& a, const simd<32>& b) {
    return ADDMM(max_epi32)(a, b);
}

simd<16> simd16_clamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val) {
    return ADDMM(min_epi16)(ADDMM(max_epi16)(value, min_val), max_val);
}
simd<16> simd16_uclamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val) {
    return ADDMM(min_epu16)(ADDMM(max_epu16)(value, min_val), max_val);
}

simd<32> simdint_clamp(const simd<32>& value, const simd<32>& min_val, const simd<32>& max_val) {
    return ADDMM(min_epi32)(ADDMM(max_epi32)(value, min_val), max_val);
}

simd<32> simdint_mullo(const simd<32>& a, const simd<32>& b) {
    return ADDMM(mullo_epi32)(a, b);
}

simd<16> mull_add(const simd<16>& a, const simd<16>& b){
    return ADDMM(madd_epi16)(a, b);
}

int mysum(const simd<32>& x){
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

simd<16> simd16_add(const simd<16>& a, const simd<16>& b) {
    return ADDMM(add_epi16)(a, b);
}

simd<16> simd16_sub(const simd<16>& a, const simd<16>& b) {
    return ADDMM(sub_epi16)(a, b);
}
simd<8> simd8_add(const simd<8>& a, const simd<8>& b) {
    return ADDMM(add_epi8)(a, b);
}

simd<8> simd8_sub(const simd<8>& a, const simd<8>& b) {
    return ADDMM(sub_epi8)(a, b);
}
#ifndef __AVX2__
simd<16> simdh8_16(const simdhalf& v){
    return ADDMM(cvtepi8_epi16)(_mm_set_epi64x(0, v));
}
#else
simd<16> simdh8_16(const simdhalf& v){
    return ADDMM(cvtepi8_epi16)(v);
}
#endif