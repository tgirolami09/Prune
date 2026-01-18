#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "simd_definitions.hpp"
#include <fstream>

using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 128;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
const int BUCKET = 8;
const int DIVISOR=(31+BUCKET)/BUCKET;

static_assert(HL_SIZE%nb16 == 0);

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