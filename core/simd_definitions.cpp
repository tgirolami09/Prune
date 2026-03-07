#include "simd_definitions.hpp"
#include <vector>

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

simdint simdint_add(simdint a, simdint b) {
    return ADDMM(add_epi32)(a, b);
}

simd16 simd16_mullo(simd16 a, simd16 b) {
    return ADDMM(mullo_epi16)(a, b);
}

simd16 simd16_clamp(simd16 value, simd16 min_val, simd16 max_val) {
    return ADDMM(min_epi16)(ADDMM(max_epi16)(value, min_val), max_val);
}

simdint simdint_mullo(simdint a, simdint b) {
    return ADDMM(mullo_epi32)(a, b);
}

simdint mull_add(simd16 a, simd16 b){
    return ADDMM(madd_epi16)(a, b);
}

int mysum(simdint x){
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

simd16 simd16_add(simd16 a, simd16 b) {
    return ADDMM(add_epi16)(a, b);
}

simd16 simd16_sub(simd16 a, simd16 b) {
    return ADDMM(sub_epi16)(a, b);
}
simd8 simd8_add(simd8 a, simd8 b) {
    return ADDMM(add_epi8)(a, b);
}

simd8 simd8_sub(simd8 a, simd8 b) {
    return ADDMM(sub_epi8)(a, b);
}
#ifdef __AVX512F__
#elif defined(__AVX2__)
alignas(64) static const uint8_t shuffle8[nb8] = {0, 0x80, 1, 0x80, 2, 0x80, 3, 0x80, 4, 0x80, 5, 0x80, 6, 0x80, 7, 0x80, 24, 0x80, 25, 0x80, 26, 0x80, 27, 0x80, 28, 0x80, 29, 0x80, 30, 0x80, 31, 0x80};
const simd8& shuf8 = *(simd8*)&shuffle8;
#else
alignas(64) static const uint8_t shuffle8l[nb8] = {0, 0x80, 1, 0x80, 2 , 0x80, 3 , 0x80, 4 , 0x80, 5 , 0x80, 6 , 0x80, 7 , 0x80};
alignas(64) static const uint8_t shuffle8h[nb8] = {8, 0x80, 9, 0x80, 10, 0x80, 11, 0x80, 12, 0x80, 13, 0x80, 14, 0x80, 15, 0x80};
static_assert(sizeof(shuffle8h) == sizeof(simd8));
static_assert(sizeof(shuffle8l) == sizeof(simd8));
const simd8& shuf8l = *(simd8*)&shuffle8l;
const simd8& shuf8h = *(simd8*)&shuffle8l;
#endif
alignas(64) static const std::vector<int16_t> n128(nb16, 8);
static const simd8& v128=*(simd16*)&n128;

#ifdef __AVX2__
simd16 simd8_16l(simd8 v){
    return _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
}
simd16 simd8_16h(simd8 v){
    return _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
}
#else
simd16 simd8_16l(simd8 v){
    return ADDMM(shuffle_epi8)(v, shuf8l);
}

simd16 simd8_16h(simd8 v){
    return ADDMM(shuffle_epi8)(v, shuf8h);
}
#endif