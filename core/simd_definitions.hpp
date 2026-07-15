#ifndef SIMD_DEFINITIONS_HPP
#define SIMD_DEFINITIONS_HPP
#include <cstdint>

#if defined(__x86_64__)
    #include <immintrin.h>

#elif defined(__ARM_NEON__)
    #include <arm_neon.h>
    // Make sure no x86 features are on
    #undef __AVX512F__
    #undef __AVX2__
    #undef __SSE2__

#endif

#define dbyte int16_t
// Manual SIMD wrapper for cross-platform compatibility
#ifdef __AVX512F__
    using _simd = __m512i;
    using simdhalf = __m256i;
    #define MM _mm512
    #define SIZE 512
#elif defined(__AVX2__)
    using _simd = __m256i;
    using simdhalf = __m128i;
    #define MM _mm256
    #define SIZE 256
#elif defined(__SSE2__)
    using _simd = __m128i;
    using simdhalf = int64_t;
    #define MM _mm
    #define SIZE 128
#elif defined(__ARM_NEON__)
    // NEON behaves a bit differently than x86 simd
    using simdhalf = int8x8_t;
    #define SIZE 128
#else
    #error "This code requires at least SSE2 (x86) or NEON (ARM) support"
#endif

#if defined(__AVX512VNNI__) || defined(__AVX2_VNNI__)
#define VNNI
#pragma message("cpu support vnni instructions")
#endif


#ifdef __ARM_NEON__
    // On NEON each width maps to its own register type so we have to do everything explicitly
    template<int size> struct simd_register;
    template<> struct simd_register<8>  { using type = int8x16_t; };
    template<> struct simd_register<16> { using type = int16x8_t; };
    template<> struct simd_register<32> { using type = int32x4_t; };
    template<int size> using simd = typename simd_register<size>::type;
#else
    // On x86 a single register type is used for every width
    template<int size>
    using simd = _simd;
#endif

template<int size>
constexpr int nb = sizeof(simd<size>)*8/size;

template<typename T1, typename T2>
constexpr int nbTypes = sizeof(T1)/sizeof(T2); 
constexpr int I8inI32 = nbTypes<int32_t, int8_t>;
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#define ADDMM(func_name) CONCAT(MM, _ ## func_name)
#define ADDSIZE(func_name) CONCAT(func_name, SIZE)

// SIMD utility functions
#ifdef __ARM_NEON__
inline simd<16> simd16_zero(){
    return vdupq_n_s16(0);
}
inline simd<32> simdint_zero(){
    return vdupq_n_s32(0);
}
inline simd<16> simd16_set1(dbyte value){
    return vdupq_n_s16(value);
}
inline simd<32> simdint_set1(int value){
    return vdupq_n_s32(value);
}
#else
inline simd<16> simd16_zero(){
    return ADDSIZE(ADDMM(setzero_si))();
}
inline simd<32> simdint_zero(){
    return ADDSIZE(ADDMM(setzero_si))();
}
inline simd<16> simd16_set1(dbyte value){
    return ADDMM(set1_epi16)(value);
}
inline simd<32> simdint_set1(int value){
    return ADDMM(set1_epi32)(value);
}
#endif

simd<32> simdint_add(const simd<32>& a, const simd<32>& b);
simd<16> simd16_mullo(const simd<16>& a, const simd<16>& b);
simd<16> simd16_mulhi(const simd<16>& a, const simd<16>& b);
simd<16> simd16_min(const simd<16>& a, const simd<16>& b);
simd<32> simdint_min(const simd<32>& a, const simd<32>& b);
simd<32> simdint_max(const simd<32>& a, const simd<32>& b);
simd<16> simd16_clamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val);
simd<16> simd16_uclamp(const simd<16>& value, const simd<16>& min_val, const simd<16>& max_val);
simd<32> simdint_clamp(const simd<32>& value, const simd<32>& min_val, const simd<32>& max_val);
simd<32> simdint_mullo(const simd<32>& a, const simd<32>& b);
simd<32> mull_add(const simd<16>& a, const simd<16>& b);
simd<32> simdint_shr(const simd<32>& a, int b);
simd<16> simd16_shr(const simd<16>& a, int b);
simd<16> simd16_sli(const simd<16>& a, int shift);
int mysum(const simd<32>& x);
simd<16> simd16_add(const simd<16>& a, const simd<16>& b);
simd<16> simd16_sub(const simd<16>& a, const simd<16>& b);
simd<8> simd8_add(const simd<8>& a, const simd<8>& b);
simd<8> simd8_sub(const simd<8>& a, const simd<8>& b);
simd<16> simdh8_16(const simdhalf& v);
simd<8> simd8_packus(const simd<16>& a, const simd<16>& b);
simd<16> simd16_maddubs(const simd<8>& a, const simd<8>& b);
simd<8> simd8_broadcast32(uint32_t v);
simd<32> simdint_dpbusd(const simd<32>& acc, const simd<8>& u, const simd<8>& s);
#endif
