#include "simd_definitions.hpp"

#ifdef __ARM_NEON__
//  ARM NEON backend

simd<32> simdint_add(const simd<32>& a, const simd<32>& b) {
    return vaddq_s32(a, b);
}

simd<16> simd16_sli(const simd<16>& a, int shift){
    return vshlq_s16(a, vdupq_n_s16(static_cast<int16_t>(shift)));
}

simd<16> simd16_mulhi(const simd<16>& a, const simd<16>& b){
    const int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    const int32x4_t hi = vmull_high_s16(a, b);
    const int16x4_t lo16 = vshrn_n_s32(lo, 16);
    return vshrn_high_n_s32(lo16, hi, 16);
}

simd<16> simd16_mullo(const simd<16>& a, const simd<16>& b) {
    return vmulq_s16(a, b);
}

simd<32> simdint_shr(const simd<32>& a, int b){
    const uint32x4_t u = vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-b));
    return vreinterpretq_s32_u32(u);
}

simd<16> simd16_shr(const simd<16>& a, int b){
    const uint16x8_t u = vshlq_u16(vreinterpretq_u16_s16(a),
                                   vdupq_n_s16(static_cast<int16_t>(-b)));
    return vreinterpretq_s16_u16(u);
}

simd<16> simd16_min(const simd<16>& a, const simd<16>& b) {
    return vminq_s16(a, b);
}
simd<32> simdint_min(const simd<32>& a, const simd<32>& b) {
    return vminq_s32(a, b);
}
simd<32> simdint_max(const simd<32>& a, const simd<32>& b) {
    return vmaxq_s32(a, b);
}

simd<16> simd16_clamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val) {
    return vminq_s16(vmaxq_s16(value, min_val), max_val);
}
simd<16> simd16_uclamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val) {
    const uint16x8_t v  = vreinterpretq_u16_s16(value);
    const uint16x8_t mn = vreinterpretq_u16_s16(min_val);
    const uint16x8_t mx = vreinterpretq_u16_s16(max_val);
    return vreinterpretq_s16_u16(vminq_u16(vmaxq_u16(v, mn), mx));
}

simd<32> simdint_clamp(const simd<32>& value, const simd<32>& min_val, const simd<32>& max_val) {
    return vminq_s32(vmaxq_s32(value, min_val), max_val);
}

simd<32> simdint_mullo(const simd<32>& a, const simd<32>& b) {
    return vmulq_s32(a, b);
}

simd<32> mull_add(const simd<16>& a, const simd<16>& b){
    const int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    const int32x4_t hi = vmull_high_s16(a, b);
    return vpaddq_s32(lo, hi);
}

int mysum(const simd<32>& x){
    return vaddvq_s32(x);
}

simd<16> simd16_add(const simd<16>& a, const simd<16>& b) {
    return vaddq_s16(a, b);
}

simd<16> simd16_sub(const simd<16>& a, const simd<16>& b) {
    return vsubq_s16(a, b);
}
simd<8> simd8_add(const simd<8>& a, const simd<8>& b) {
    return vaddq_s8(a, b);
}

simd<8> simd8_sub(const simd<8>& a, const simd<8>& b) {
    return vsubq_s8(a, b);
}

simd<16> simdh8_16(const simdhalf& v){
    return vmovl_s8(v);
}

simd<8> simd8_packus(const simd<16>& a, const simd<16>& b){
    const uint8x8_t lo = vqmovun_s16(a);
    return vreinterpretq_s8_u8(vqmovun_high_s16(lo, b));
}

simd<16> simd16_maddubs(const simd<8>& a, const simd<8>& b){
    const uint8x16_t u = vreinterpretq_u8_s8(a);
    const int16x8_t u_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u)));
    const int16x8_t u_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u)));
    const int16x8_t s_lo = vmovl_s8(vget_low_s8(b));
    const int16x8_t s_hi = vmovl_s8(vget_high_s8(b));
    const int16x8_t p_lo = vmulq_s16(u_lo, s_lo);
    const int16x8_t p_hi = vmulq_s16(u_hi, s_hi);
    const int32x4_t sum_lo = vpaddlq_s16(p_lo);
    const int32x4_t sum_hi = vpaddlq_s16(p_hi);
    return vcombine_s16(vqmovn_s32(sum_lo), vqmovn_s32(sum_hi));
}

simd<8> simd8_broadcast32(uint32_t v){
    return vreinterpretq_s8_s32(vdupq_n_s32(static_cast<int32_t>(v)));
}

simd<32> simdint_dpbusd(const simd<32>& acc, const simd<8>& u, const simd<8>& s){
    const uint8x16_t uu = vreinterpretq_u8_s8(u);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    // Single-instruction unsigned x signed dot product, should work on M2 chips and later
    return vusdotq_s32(acc, uu, s);
#elif defined(__ARM_FEATURE_DOTPROD)
    // dotprod-only chips (M1 chips) have signed sdot but no unsigned x signed
    const int8x16_t us = vreinterpretq_s8_u8(veorq_u8(uu, vdupq_n_u8(0x80)));
    const int32x4_t bias = vdotq_s32(vdupq_n_s32(0), vdupq_n_s8(1), s);
    return vaddq_s32(vdotq_s32(acc, us, s), vshlq_n_s32(bias, 7));
#else
    // Portable NEON fallback, in case none of the above exist
    const int16x8_t p_lo = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(uu))),
                                     vmovl_s8(vget_low_s8(s)));
    const int16x8_t p_hi = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(uu))),
                                     vmovl_s8(vget_high_s8(s)));
    const int32x4_t summed = vpaddq_s32(vpaddlq_s16(p_lo), vpaddlq_s16(p_hi));
    return vaddq_s32(acc, summed);
#endif
}

#else

// x86 backend (SSE2 / AVX2 / AVX512) with the MM / ADDMM macro family
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

simd<8> simd8_packus(const simd<16>& a, const simd<16>& b){
    return ADDMM(packus_epi16)(a, b);
}

simd<16> simd16_maddubs(const simd<8>& a, const simd<8>& b){
    return ADDMM(maddubs_epi16)(a, b);
}

simd<8> simd8_broadcast32(uint32_t v){
    return ADDMM(set1_epi32)(static_cast<int>(v));
}

simd<32> simdint_dpbusd(const simd<32>& acc, const simd<8>& u, const simd<8>& s){
#ifdef VNNI
    return ADDMM(dpbusd_epi32)(acc, u, s);
#else
    return ADDMM(add_epi32)(acc, ADDMM(madd_epi16)(ADDMM(maddubs_epi16)(u, s), simd16_set1(1)));
#endif
}

#endif