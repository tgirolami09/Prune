#include "NNUE.hpp"
#include "Const.hpp"
#include <cstring>
#include <fstream>
#include <immintrin.h>  // For Intel intrinsics
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

template<typename T>
dbyte NNUE::read_bytes(ifstream& file){
    T ret;
    file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    return ret;
}

dbyte read_i16(ifstream& file){
    dbyte ret;
    file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    return ret;
}

dbyte read_i16(const unsigned char* file){
    dbyte ret;
    memcpy(&ret, file, sizeof(ret));
    return ret;
}

// Helper to set individual elements in SIMD vectors
void NNUE::set_simd16_element(simd16& vec, int index, dbyte value) {
    dbyte* ptr = reinterpret_cast<dbyte*>(&vec);
    ptr[index] = value;
}

big genRandom(big& state){
    big z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

NNUE::NNUE(string name){
    if(name == "random"){
        big state = 42;
        for(int i=0; i<INPUT_SIZE; i++) {
            for(int j=0; j<HL_SIZE/nb16; j++) {
                hlWeights[i][j] = simd16_zero();
                for(int k=0; k<nb16; k++) {
                    set_simd16_element(hlWeights[i][j], k, genRandom(state)%256-128);
                }
            }
        }
        
        for(int i=0; i<HL_SIZE/nb16; i++){
            hlBiases[i] = simd16_zero();
            for(int id16=0; id16<nb16; id16++) {
                set_simd16_element(hlBiases[i], id16, genRandom(state)%256-128);
            }
        }
        
        for(int idB = 0; idB < BUCKET; idB++){
            for(int side=0; side < 2; side++){
                for(int i=0; i<HL_SIZE/nb16; i++) {
                    outWeights[idB][side][i] = simd16_zero();
                    for(int id16=0; id16<nb16; id16++) {
                        set_simd16_element(outWeights[idB][side][i], id16, genRandom(state)%256-128);
                    }
                }
            }
        }
        for(int idB = 0; idB < BUCKET; idB++)
            outbias[idB] = genRandom(state)%256-128;
    }else{
        ifstream file(name);
        file.read(reinterpret_cast<char*>(hlWeights), sizeof(hlWeights));
        file.read(reinterpret_cast<char*>(hlBiases), sizeof(hlBiases));
        file.read(reinterpret_cast<char*>(outWeights), sizeof(outWeights));
        file.read(reinterpret_cast<char*>(outbias), sizeof(outbias));
    }
}
template<typename T>
T get_int(const unsigned char* source, int length){
    T res;
    memcpy(&res, source, length);
    return res;
}

int turn(int index){
    return ((index^56)+384)%768;
}

NNUE::NNUE(){
    int pointer = 0;
    memcpy(hlWeights, baseModel, sizeof(hlWeights));
    pointer += sizeof(hlWeights);
    memcpy(hlBiases, &baseModel[pointer], sizeof(hlBiases));
    pointer += sizeof(hlBiases);
    memcpy(outWeights, &baseModel[pointer], sizeof(outWeights));
    pointer += sizeof(outWeights);
    memcpy(outbias, &baseModel[pointer], sizeof(outbias));
}

void NNUE::initAcc(Accumulator& accs){
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
}

int NNUE::get_index(int piece, int c, int square) const{
    return ((6*c+piece)<<6)|(square^7);
}

static const simd16 mini = simd16_set1(0);
static const simd16 maxi = simd16_set1(QA);
simdint doOut(simd16 a, simd16 w){
    simd16 clamped = simd16_clamp(a, mini, maxi);
    simd16 mul = simd16_mullo(clamped, w);
    simdint overall = mull_add(mul, clamped);
    return overall;
} 

dbyte NNUE::eval(const Accumulator& accs, bool side, int idB) const{
    simdint res = simdint_zero();
    for(int i=0; i<HL_SIZE/nb16; i++){
        res = simdint_add(res, doOut(accs[side^1][i], outWeights[idB][1][i]));
        res = simdint_add(res, doOut(accs[side][i], outWeights[idB][0][i]));
    }
    int finRes = mysum(res);
    finRes /= QA;
    finRes += outbias[idB];
    finRes = finRes*SCALE/(QA*QB);
    return finRes;
}


template<int f>
void NNUE::change2(Accumulator& accs, int piece, int c, int square){
    int index = get_index(piece, c, square);
    int index2 = turn(index); // point of view change
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
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, int piece, int c, int square){
    int index = get_index(piece, c, square);
    int index2 = turn(index); // point of view change
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
void NNUE::move3(Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 white = simd16_sub(hlWeights[indexto     ][i], simd16_add(hlWeights[indexfrom     ][i], hlWeights[indexcap     ][i]));
        simd16 black = simd16_sub(hlWeights[turn(indexto)][i], simd16_add(hlWeights[turn(indexfrom)][i], hlWeights[turn(indexcap)][i]));
        accOut[WHITE][i] = simd16_add(accIn[WHITE][i], white);
        accOut[BLACK][i] = simd16_add(accIn[BLACK][i], black);
    }
}
void NNUE::move2(Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 white = simd16_sub(hlWeights[indexto     ][i], hlWeights[indexfrom     ][i]);
        simd16 black = simd16_sub(hlWeights[turn(indexto)][i], hlWeights[turn(indexfrom)][i]);
        accOut[WHITE][i] = simd16_add(accIn[WHITE][i], white);
        accOut[BLACK][i] = simd16_add(accIn[BLACK][i], black);
    }
}
void NNUE::move4(Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 white = simd16_sub(simd16_add(hlWeights[indexto1     ][i], hlWeights[indexto2     ][i]), simd16_add(hlWeights[indexfrom1     ][i], hlWeights[indexfrom2     ][i]));
        simd16 black = simd16_sub(simd16_add(hlWeights[turn(indexto1)][i], hlWeights[turn(indexto2)][i]), simd16_add(hlWeights[turn(indexfrom1)][i], hlWeights[turn(indexfrom2)][i]));
        accOut[WHITE][i] = simd16_add(accIn[WHITE][i], white);
        accOut[BLACK][i] = simd16_add(accIn[BLACK][i], black);
    }

}
template void NNUE::change2<-1>(Accumulator&, int, int, int);
template void NNUE::change2<1>(Accumulator&, int, int, int);
template void NNUE::change2<-1>(Accumulator&, Accumulator&, int, int, int);
template void NNUE::change2<1>(Accumulator&, Accumulator&, int, int, int);

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
