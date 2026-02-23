#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "simd_definitions.hpp"
#include "GameState.hpp"
#include <fstream>
#include "embeder.hpp"
using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 1024;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
const int BUCKET = 8;
const int DIVISOR=(31+BUCKET)/BUCKET;

static_assert(HL_SIZE%nb16 == 0);

class Index{
public:
    int square;
    int piece;
    bool color;
    Index();
    Index(int square, int piece, bool color);
    void smirror(bool needmirror);
    Index mirror(bool needmirror);
    Index changepov();
    Index changepov(bool needs);
    void schangepov();
    operator int();
    bool isnull();
};

int mirrorSquare(int square, bool mirror);

class updateBuffer{
public:
    Index add1[2], add2[2];
    Index sub1[2], sub2[2]; //each pieces provoque a change in black and white pov
    bool dirty;
    int type;
    updateBuffer();
    updateBuffer(Index sub1, Index add1, Index sub2, Index add2);
    void print();
};

class Accumulator{
public:
    simd16 accs[2][HL_SIZE/nb16];
    bool Kside[2];
    bool side;
    bool mustmirror;
    big bitboards[2][6];
    updateBuffer update;
    Accumulator(){}
    void reinit(const GameState* state, Accumulator& prevAcc, bool side, bool mirror, Index sub1, Index add1, Index sub2=Index(), Index add2=Index());
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
    void initAcc(Accumulator& accs) const;
    void initAcc(Accumulator& accs, bool color) const;
    int get_index(int piece, int c, int square) const;
    template<int f>
    void change1(Accumulator& accIn, bool pov, int index) const;
    template<int f>
    void change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index) const;
    void move3(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap) const;
    void move2(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto) const;
    void move4(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2) const;
    void updateStack(Accumulator* stack, int stackIndex) const;
    dbyte eval(const Accumulator& accs, bool side, int idB) const;
};

inline const NNUE& globnnue = *reinterpret_cast<const NNUE*>(baseModel);
#endif
