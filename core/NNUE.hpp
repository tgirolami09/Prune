#ifndef NNUE_CPP
#define NNUE_CPP
#include <cstdint>
#include "Const.hpp"
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
BINARY_INCLUDE(baseModel)
// Manual SIMD wrapper for cross-platform compatibility
#if defined(__AVX2__)
    using simd16 = __m256i;  // 16 int16_t values
    const int nb16 = 16;
    // For AVX2, we need TWO __m256i to hold 16 int32 values
    struct simdint_type {
        __m256i lo, hi;
        simdint_type();
        simdint_type(__m256i l, __m256i h);
    };
    using simdint = simdint_type;
    const int nbint = 16;  // 16 int32 values total
#elif defined(__SSE2__)
    using simd16 = __m128i;  // 8 int16_t values  
    const int nb16 = 8;
    // For SSE2, we need TWO __m128i to hold 8 int32 values
    struct simdint_type {
        __m128i lo, hi;
        simdint_type();
        simdint_type(__m128i l, __m128i h);
    };
    using simdint = simdint_type;
    const int nbint = 8;  // 8 int32 values total
#else
    #error "This code requires at least SSE2 support"
#endif

static_assert(HL_SIZE%nb16 == 0);

// SIMD utility functions
inline simd16 simd16_zero();
inline simdint simdint_zero();
inline simd16 simd16_set1(dbyte value);
inline simdint simdint_set1(int value);
inline simd16 simd16_add(simd16 a, simd16 b);
inline simd16 simd16_sub(simd16 a, simd16 b);
inline simdint simdint_add(simdint a, simdint b);
inline simd16 simd16_mullo(simd16 a, simd16 b);
inline simd16 simd16_clamp(simd16 value, simd16 min_val, simd16 max_val);
inline simdint convert16(simd16 var);
inline simdint simdint_mullo(simdint a, simdint b);
template<int min, int max>
simdint SCReLU(simd16 value);
simdint activation(simd16 value);
int mysum(simdint x);

class NNUE{
public:
    simd16 hlWeights[INPUT_SIZE][HL_SIZE/nb16];
    simd16 hlBiases[HL_SIZE/nb16];
    simdint outWeights[2*HL_SIZE/nb16];
    dbyte outbias;
    simd16 accs[2][HL_SIZE/nb16];
    
    dbyte read_bytes(ifstream& file);
    // Helper to set individual elements in SIMD vectors
    void set_simd16_element(simd16& vec, int index, dbyte value);
    void set_simdint_element(simdint& vec, int index, int value);
    NNUE(string name);
    NNUE();
    void clear();
    int get_index(int piece, int square) const;
    template<int f>
    void change2(int piece, int square);
    dbyte eval(bool side) const;
};


template<int f>
void NNUE::change2(int piece, int square){
    int index = get_index(piece, square);
    int index2 = index ^ 56 ^ 64; // point of view change
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accs[WHITE][i] = simd16_add(accs[WHITE][i], hlWeights[index][i]);
            accs[BLACK][i] = simd16_add(accs[BLACK][i], hlWeights[index2][i]);
        } else {
            accs[WHITE][i] = simd16_sub(accs[WHITE][i], hlWeights[index][i]);
            accs[BLACK][i] = simd16_sub(accs[BLACK][i], hlWeights[index2][i]);
        }
    }
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