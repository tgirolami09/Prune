#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "Functions.hpp"
#include <cstdint>
#include <immintrin.h>  // For Intel intrinsics
#include <fstream>
#include <string>

using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 64;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;

#define dbyte int16_t

// Manual SIMD wrapper for cross-platform compatibility
#if defined(__AVX2__)
    using simd16 = __m256i;  // 16 int16_t values
    const int nb16 = 16;
    using simdint = __m256i;
    const int nbint = 8;  // 8 int32 values total
#elif defined(__SSE2__)
    using simd16 = __m128i;  // 8 int16_t values  
    const int nb16 = 8;
    using simdint = __m128i;
    const int nbint = 4;  // 4 int32 values total
#else
    #error "This code requires at least SSE2 support"
#endif

static_assert(HL_SIZE%nb16 == 0);

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

inline simd16 simd16_add(simd16 a, simd16 b) {
#if defined(__AVX2__)
    return _mm256_add_epi16(a, b);
#else
    return _mm_add_epi16(a, b);
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

inline void convert16(simd16 var, simdint& lo, simdint& hi) {
#if defined(__AVX2__)
    // Convert all 16 int16 values to 16 int32 values using two AVX2 registers
    __m128i lo_half = _mm256_extracti128_si256(var, 0);  // Lower 8 int16 values
    __m128i hi_half = _mm256_extracti128_si256(var, 1);  // Upper 8 int16 values
    lo = _mm256_cvtepi16_epi32(lo_half);      // Convert lower 8 int16 to int32
    hi = _mm256_cvtepi16_epi32(hi_half);      // Convert upper 8 int16 to int32
#else
    // Convert all 8 int16 values to 8 int32 values using two SSE registers
    lo = _mm_unpacklo_epi16(var, _mm_setzero_si128()); // Convert lower 4 int16 to int32
    hi = _mm_unpackhi_epi16(var, _mm_setzero_si128()); // Convert upper 4 int16 to int32
#endif
}

inline simdint simdint_mullo(simdint a, simdint b) {
#if defined(__AVX2__)
    return _mm256_mullo_epi32(a, b);
#else
    return _mm_mullo_epi32(a, b);
#endif
}

template<int min, int max>
void SCReLU(simd16 value, simdint& res1, simdint& res2){
    static const simd16 mini = simd16_set1(min);
    static const simd16 maxi = simd16_set1(max);
    value = simd16_clamp(value, mini, maxi);
    convert16(value, res1, res2);
    res1 = simdint_mullo(res1, res1);
    res2 = simdint_mullo(res2, res2);
}

void activation(simd16 value, simdint& res1, simdint& res2){
    SCReLU<0, QA>(value, res1, res2);
}

int mysum(simdint x){
#if defined(__AVX2__)
    // Sum all 16 int32 values from two AVX2 registers
    __m256i sum256 = x;  // Add the two halves
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_extract_epi32(sum128, 0);
#else
    // Sum all 8 int32 values from two SSE registers
    __m128i sum = x;  // Add the two halves
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_extract_epi32(sum, 0);
#endif
}

BINARY_ASM_INCLUDE("model.bin", baseModel);

class NNUE{
public:
    simd16 hlWeights[INPUT_SIZE][HL_SIZE/nb16];
    simd16 hlBiases[HL_SIZE/nb16];
    simdint outWeights[2*HL_SIZE/nbint];
    dbyte outbias;
    simd16 accs[2][HL_SIZE/nb16];
    
    dbyte read_bytes(ifstream& file){
        char ret;
        file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
        return ret;
    }
    
    // Helper to set individual elements in SIMD vectors
    void set_simd16_element(simd16& vec, int index, dbyte value) {
        dbyte* ptr = reinterpret_cast<dbyte*>(&vec);
        ptr[index] = value;
    }
    
    void set_simdint_element(simdint& vec, int index, int value) {
        int* ptr = reinterpret_cast<int*>(&vec);
        ptr[index] = value;
    }
    
    NNUE(string name){
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
    
    NNUE(){
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
        
        for(int i=0; i<2*HL_SIZE/nbint; i++) {
            outWeights[i] = simdint_zero();
            for(int id16=0; id16<nbint; id16++) {
                set_simdint_element(outWeights[i], id16, transform(baseModel[pointer++]));
            }
        }
        
        outbias = transform(baseModel[pointer++]);
    }
    
    void clear(){
        for(int i=0; i<HL_SIZE/nb16; i++){
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
    }
    
    int get_index(int piece, int square){
        return (piece<<6)|(square^56);
    }
    
    template<int f>
    void change2(int piece, int square){
        int index = get_index(piece, square);
        int index2 = index ^ 56 ^ 64; // point of view change
        for(int i=0; i<HL_SIZE/nb16; i++){
            if constexpr (f == 1) {
                accs[WHITE][i] = simd16_add(accs[WHITE][i], hlWeights[index][i]);
                accs[BLACK][i] = simd16_add(accs[BLACK][i], hlWeights[index2][i]);
            } else {
                // For subtraction, we need to negate and add
                simd16 neg_weight1 = simd16_mullo(hlWeights[index][i], simd16_set1(-1));
                simd16 neg_weight2 = simd16_mullo(hlWeights[index2][i], simd16_set1(-1));
                accs[WHITE][i] = simd16_add(accs[WHITE][i], neg_weight1);
                accs[BLACK][i] = simd16_add(accs[BLACK][i], neg_weight2);
            }
        }
    }
    
    dbyte eval(bool side){
        simdint res = simdint_zero();
        for(int i=0; i<HL_SIZE/nb16; i++){
            simdint act1, act2;
            activation(accs[side][i], act1, act2);
            res = simdint_add(res, simdint_mullo(act1, outWeights[2*i  ]));
            res = simdint_add(res, simdint_mullo(act2, outWeights[2*i+1]));
            activation(accs[side^1][i], act1, act2);
            res = simdint_add(res, simdint_mullo(act1, outWeights[HL_SIZE/nbint+2*i  ]));
            res = simdint_add(res, simdint_mullo(act2, outWeights[HL_SIZE/nbint+2*i+1]));
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
};
#endif