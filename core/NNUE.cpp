#include "NNUE.hpp"
#include "Const.hpp"
#include <immintrin.h>  // For Intel intrinsics
#include "Functions.hpp"
#include "embeder.hpp"

using namespace std;

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
    }
    
    for(int i=0; i<2*HL_SIZE/nb16; i++) {
        outWeights[i] = simd16_zero();
        for(int id16=0; id16<nb16; id16++) {
            set_simd16_element(outWeights[i], id16, read_bytes(file));
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
    }
    
    for(int i=0; i<2*HL_SIZE/nb16; i++) {
        outWeights[i] = simd16_zero();
        for(int id16=0; id16<nb16; id16++) {
            set_simd16_element(outWeights[i], id16, transform(baseModel[pointer++]));
        }
    }
    
    outbias = transform(baseModel[pointer++]);
}

void NNUE::initAcc(Accumulator& accs){
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
}

int NNUE::get_index(int piece, int square) const{
    return (piece<<6)|(square^56);
}

static const simd16 mini = simd16_set1(0);
static const simd16 maxi = simd16_set1(QA);
simdint doOut(simd16 a, simd16 w){
    simd16 clamped = simd16_clamp(a, mini, maxi);
    simd16 mul = simd16_mullo(clamped, w);
    simdint overall = mull_add(mul, clamped);
    return overall;
} 

dbyte NNUE::eval(const Accumulator& accs, bool side) const{
    simdint res = simdint_zero();
    for(int i=0; i<HL_SIZE/nb16; i++){
        res = simdint_add(res, doOut(accs[side^1][i], outWeights[i+HL_SIZE/nb16]));
        res = simdint_add(res, doOut(accs[side][i], outWeights[i]));
    }
    int finRes = mysum(res);
    finRes /= QA;
    finRes += outbias;
    finRes = finRes*SCALE/(QA*QB);
    return finRes;
}


template<int f>
void NNUE::change2(Accumulator& accs, int piece, int square){
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
template<int f>
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, int piece, int square){
    int index = get_index(piece, square);
    int index2 = index ^ 56 ^ 64; // point of view change
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accOut[WHITE][i] = simd16_add(accIn[WHITE][i], hlWeights[index][i]);
            accOut[BLACK][i] = simd16_add(accIn[BLACK][i], hlWeights[index2][i]);
        } else {
            accOut[WHITE][i] = simd16_sub(accIn[WHITE][i], hlWeights[index][i]);
            accOut[BLACK][i] = simd16_sub(accIn[BLACK][i], hlWeights[index2][i]);
        }
    }
}
template void NNUE::change2<-1>(Accumulator&, int, int);
template void NNUE::change2<1>(Accumulator&, int, int);
template void NNUE::change2<-1>(Accumulator&, Accumulator&, int, int);
template void NNUE::change2<1>(Accumulator&, Accumulator&, int, int);

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
NNUE globnnue = NNUE();