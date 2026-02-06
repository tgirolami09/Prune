#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "simd_definitions.hpp"
#include <fstream>

using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 512;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
const int BUCKET = 8;
const int DIVISOR=(31+BUCKET)/BUCKET;

static_assert(HL_SIZE%nb16 == 0);

class updateBuffer{
public:
    int add1[2], add2[2];
    int sub1[2], sub2[2]; //each pieces provoque a change in black and white pov
    bool dirty;
    int type;
    updateBuffer();
    updateBuffer(int _add1, int _add2, int _sub1, int _sub2);
    updateBuffer(int _add1, int _sub1, int _sub2);
    updateBuffer(int _add1, int _sub1);
};

class Accumulator{
public:
    simd16 accs[2][HL_SIZE/nb16];
    updateBuffer update;
    Accumulator(){}
    Accumulator(int add1, int add2, int sub1, int sub2);
    Accumulator(int add1, int sub1, int sub2);
    Accumulator(int add1, int sub1);
    const simd16* operator[](int idx) const{
        return accs[idx];
    }
    simd16* operator[](int idx){
        return accs[idx];
    }
    void updateSelf(Accumulator& accIn);
};

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
    void move3(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap);
    void move2(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto);
    void move4(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2);
    void updateStack(Accumulator stack[maxDepth], int stackIndex);
    dbyte eval(const Accumulator& accs, bool side, int idB) const;
};

extern NNUE globnnue;
#endif