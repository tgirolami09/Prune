#include "NNUE.hpp"
#include "Const.hpp"
#include <immintrin.h>  // For Intel intrinsics
#include "Functions.hpp"

using namespace std;

// Manual SIMD wrapper for cross-platform compatibility
#if defined(__AVX2__)
    simdint_type::simdint_type() : lo(_mm256_setzero_si256()), hi(_mm256_setzero_si256()) {}
    simdint_type::simdint_type(__m256i l, __m256i h) : lo(l), hi(h) {}
#elif defined(__SSE2__)
    // For SSE2, we need TWO __m128i to hold 8 int32 values
    simdint_type::simdint_type() : lo(_mm_setzero_si128()), hi(_mm_setzero_si128()) {}
    simdint_type::simdint_type(__m128i l, __m128i h) : lo(l), hi(h) {}
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

inline simdint simdint_set1(int value) {
#if defined(__AVX2__)
    return simdint(_mm256_set1_epi32(value), _mm256_set1_epi32(value));
#else
    return simdint(_mm_set1_epi32(value), _mm_set1_epi32(value));
#endif
}

inline simdint simdint_add(simdint a, simdint b) {
#if defined(__AVX2__)
    return simdint(_mm256_add_epi32(a.lo, b.lo), _mm256_add_epi32(a.hi, b.hi));
#else
    return simdint(_mm_add_epi32(a.lo, b.lo), _mm_add_epi32(a.hi, b.hi));
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

inline simdint convert16(simd16 var) {
#if defined(__AVX2__)
    // Convert all 16 int16 values to 16 int32 values using two AVX2 registers
    __m128i lo_half = _mm256_extracti128_si256(var, 0);  // Lower 8 int16 values
    __m128i hi_half = _mm256_extracti128_si256(var, 1);  // Upper 8 int16 values
    __m256i lo_32 = _mm256_cvtepi16_epi32(lo_half);      // Convert lower 8 int16 to int32
    __m256i hi_32 = _mm256_cvtepi16_epi32(hi_half);      // Convert upper 8 int16 to int32
    return simdint(lo_32, hi_32);
#else
    // Convert all 8 int16 values to 8 int32 values using two SSE registers
    __m128i lo_16 = _mm_unpacklo_epi16(var, _mm_setzero_si128()); // Convert lower 4 int16 to int32
    __m128i hi_16 = _mm_unpackhi_epi16(var, _mm_setzero_si128()); // Convert upper 4 int16 to int32
    return simdint(lo_16, hi_16);
#endif
}

inline simdint simdint_mullo(simdint a, simdint b) {
#if defined(__AVX2__)
    return simdint(_mm256_mullo_epi32(a.lo, b.lo), _mm256_mullo_epi32(a.hi, b.hi));
#else
    return simdint(_mm_mullo_epi32(a.lo, b.lo), _mm_mullo_epi32(a.hi, b.hi));
#endif
}

template<int min, int max>
simdint SCReLU(simd16 value){
    static const simd16 mini = simd16_set1(min);
    static const simd16 maxi = simd16_set1(max);
    value = simd16_clamp(value, mini, maxi);
    simdint res = convert16(value);
    return simdint_mullo(res, res);
}

simdint activation(simd16 value){
    return SCReLU<0, QA>(value);
}

int mysum(simdint x){
#if defined(__AVX2__)
    // Sum all 16 int32 values from two AVX2 registers
    __m256i sum256 = _mm256_add_epi32(x.lo, x.hi);  // Add the two halves
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_extract_epi32(sum128, 0);
#else
    // Sum all 8 int32 values from two SSE registers
    __m128i sum = _mm_add_epi32(x.lo, x.hi);  // Add the two halves
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_extract_epi32(sum, 0);
#endif
}

BINARY_ASM_INCLUDE("model.bin", baseModel);

dbyte NNUE::read_bytes(ifstream& file){
    char ret;
    file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    return ret;
}

// Helper to set individual elements in SIMD vectors
void NNUE::set_simd16_element(simd16& vec, int index, dbyte value) {
    dbyte* ptr = reinterpret_cast<dbyte*>(&vec);
    ptr[index] = value;
}

void NNUE::set_simdint_element(simdint& vec, int index, int value) {
#if defined(__AVX2__)
    if (index < 8) {
        int* ptr = reinterpret_cast<int*>(&vec.lo);
        ptr[index] = value;
    } else {
        int* ptr = reinterpret_cast<int*>(&vec.hi);
        ptr[index - 8] = value;
    }
#else
    if (index < 4) {
        int* ptr = reinterpret_cast<int*>(&vec.lo);
        ptr[index] = value;
    } else {
        int* ptr = reinterpret_cast<int*>(&vec.hi);
        ptr[index - 4] = value;
    }
#endif
}

NNUE::NNUE(string name){
    ifstream file(name);
    for(int i=0; i<INPUT_SIZE; i++) {
        for(int j=0; j<HL_SIZE/nb16; j++) {
            hlWeights[i][j] = simd16_zero();
            for(int k=0; k<nb16; k++) {
                set_simd16_element(hlWeights[i][j], k, read_bytes(file));
            }
        }
    }
    
    for(int i=0; i<HL_SIZE/nb16; i++){
        hlBiases[i] = simd16_zero();
        for(int id16=0; id16<nb16; id16++) {
            set_simd16_element(hlBiases[i], id16, read_bytes(file));
        }
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
    
    for(int i=0; i<2*HL_SIZE/nbint; i++) {
        outWeights[i] = simdint_zero();
        for(int id16=0; id16<nbint; id16++) {
            set_simdint_element(outWeights[i], id16, read_bytes(file));
        }
    }
    
    outbias = read_bytes(file);
}

NNUE::NNUE(){
    int pointer = 0;
    for(int i=0; i<INPUT_SIZE; i++) {
        for(int j=0; j<HL_SIZE/nb16; j++) {
            hlWeights[i][j] = simd16_zero();
            for(int k=0; k<nb16; k++) {
                set_simd16_element(hlWeights[i][j], k, transform(baseModel[pointer++]));
            }
        }
    }
    
    for(int i=0; i<HL_SIZE/nb16; i++){
        hlBiases[i] = simd16_zero();
        for(int id16=0; id16<nb16; id16++) {
            set_simd16_element(hlBiases[i], id16, transform(baseModel[pointer++]));
        }
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
    
    for(int i=0; i<2*HL_SIZE/nb16; i++) {
        outWeights[i] = simdint_zero();
        for(int id16=0; id16<nbint; id16++) {
            set_simdint_element(outWeights[i], id16, transform(baseModel[pointer++]));
        }
    }
    
    outbias = transform(baseModel[pointer++]);
}

void NNUE::clear(){
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
}

int NNUE::get_index(int piece, int square) const{
    return (piece<<6)|(square^56);
}


dbyte NNUE::eval(bool side) const{
    simdint res = simdint_zero();
    for(int i=0; i<HL_SIZE/nb16; i++){
        res = simdint_add(res, simdint_mullo(activation(accs[side][i]), outWeights[i]));
        res = simdint_add(res, simdint_mullo(activation(accs[side^1][i]), outWeights[i+HL_SIZE/nb16]));
    }
    int finRes = mysum(res);
    finRes /= QA;
    finRes += outbias;
    if(finRes >= 0)
        finRes = finRes*SCALE/(QA*QB);
    else 
        finRes = (finRes*SCALE-(QA*QB-1))/(QA*QB);
    return finRes;
}