#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include <immintrin.h>  // For Intel intrinsics
#include <fstream>

using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 128;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
const int BUCKET = 8;
const int DIVISOR=(31+BUCKET)/BUCKET;
#define dbyte int16_t
// Manual SIMD wrapper for cross-platform compatibility
#if defined(__AVX2__)
    using simd16 = __m256i;  // 16 int16_t values
    const int nb16 = 16;
    using simdint = __m256i;
    const int nbint = 16;  // 16 int32 values total
#elif defined(__SSE2__)
    using simd16 = __m128i;  // 8 int16_t values  
    const int nb16 = 8;
    // For SSE2, we need TWO __m128i to hold 8 int32 values
    using simdint = __m128i;
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
simd16 SCReLU(simd16 value);
simd16 activation(simd16 value);
int mysum(simdint x);
using Accumulator=simd16[2][HL_SIZE/nb16];

class NNUE{
public:
    simd16 hlWeights[INPUT_SIZE][HL_SIZE/nb16];
    simd16 hlBiases[HL_SIZE/nb16];
    simd16 outWeights[BUCKET][2][HL_SIZE/nb16];
    dbyte outbias[BUCKET];

    template<typename T=char>
    dbyte read_bytes(ifstream& file);
    // Helper to set individual elements in SIMD vectors
    void set_simd16_element(simd16& vec, int index, dbyte value);
    void set_simdint_element(simdint& vec, int index, int value);
    NNUE(string name);
    NNUE();
    void initAcc(Accumulator& accs);
    int get_index(int piece, int c, int square) const;
    template<int f>
    void change2(Accumulator& accIn, int piece, int c, int square);
    template<int f>
    void change2(Accumulator& accIn, Accumulator& accOut, int piece, int c, int square);
    void move3(Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap);
    void move2(Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto);
    void move4(Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2);
    dbyte eval(const Accumulator& accs, bool side, int idB) const;
};

extern NNUE globnnue;
#endif