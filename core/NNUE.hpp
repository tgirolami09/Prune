#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "simd_definitions.hpp"
#include <fstream>
#include "Move.hpp"
#include "embeder.hpp"
#include "GameState.hpp"

using namespace std;
#ifdef DEBUG_MACRO
#include "stats_helpers.hpp"
extern StatVar<sbig, 64, 0> TIupdateRemStat;
extern StatVar<sbig, 64, 0> TIupdateAddStat;
extern StatVar<sbig, 64, 0> TIupdateTotStat;
extern StatVar<sbig, 128, -128> TIupdateDiffStat;
#endif
const int maxThreatUpdates=80;

const int INPUT_SIZE = 12*64;
const int THREAT_SIZE = 60144;

const int QA = 255;
const int QB = 64;

const int BUCKET = 8;
const int nbInputBuckets = 4;

const int HL_SIZE = 384;
const int L2=16;
const int L3=32;

const int SCALE = 400;
const int inputBuckets[32] = {
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
};
const int DIVISOR=(31+BUCKET)/BUCKET;

static_assert(HL_SIZE%nb16 == 0);

int getInputBucket(int Kpos, bool side, bool mirror);

class Index{
public:
    int square;
    int piece;
    bool color;
    Index();
    Index(int square, int piece, bool color);
    void smirror(bool needmirror);
    Index mirror(bool needmirror) const;
    Index changepov() const;
    Index changepov(bool needs) const;
    int fullpiece() const;
    void schangepov();
    void schangepov(bool needs);
    operator int();
    bool operator==(const Index a) const;
    bool isnull();
    void print() const;
};

int mirrorSquare(int square, bool mirror);
class ThreatIndex{
public:
    Index from;
    Index to;
    ThreatIndex(Index _from, Index _to);
    ThreatIndex(int fromsquare, int frompiece, int fromcolor, int tosquare, int topiece, int tocolor);
    ThreatIndex();
    bool isexcluded() const;
    bool issemiexcluded() const;
    void swap();
    ThreatIndex rswap() const;
    operator int() const;
    ThreatIndex changepov(bool needs) const;
    ThreatIndex mirror(bool needs) const;
    void print() const;
    ThreatIndex swapSemiExcluded() const{
        return issemiexcluded()?ThreatIndex(to, from):ThreatIndex(from, to);
    }
    ThreatIndex swapExcluded() const{
        return isexcluded()?ThreatIndex(to, from):ThreatIndex(from, to);
    }
};
using oneAccumulator=simd16[HL_SIZE/nb16];
class FinnytableNormal{
public:
    big bitboards[8];
    oneAccumulator accs;
};

class FinnyTables{
public:
    FinnytableNormal normals[nbInputBuckets*4];
    void init();
};

class updateBuffer{
public:
    Index add1[2], add2[2];
    Index sub1[2], sub2[2]; //each pieces provoque a change in black and white pov
    int nbThreats[2];
    ThreatIndex threatUpdates[2][32];
    bool dirty;
    int type;
    updateBuffer();
    void reset(Index sub1, Index add1, Index sub2, Index add2);
    void addThreat(const ThreatIndex& threat, const bool remove);
    void print();
};


static const simdint mini = simdint_set1(0);
static const simdint maxiA = simdint_set1(QA);
static const simdint maxiB = simdint_set1(QB);

template<bool squared>
inline simdint doOut(simdint a, simdint w, const simdint maxi){
    simdint clamped = simdint_clamp(a, mini, maxi);
    simdint mul = simdint_mullo(clamped, w);
    if constexpr(squared)
        mul = simdint_mullo(mul, clamped);
    return mul;
}

template<int input, int output, bool isSquared, int clamp>
struct Layer{
    simdint weights[output][input/nbint];
    int biases[output];
    void forward(simdint x[input/nbint], int y[output]) const{
        for(int i=0; i<output; i++){
            simdint partial = simdint_set1(0);
            for(int j=0; j<input/nb16; j++){
                partial = simdint_add(partial, doOut<isSquared>(x[j], weights[i][j], simdint_set1(clamp)));
            }
            y[i] = biases[i]+mysum(partial);
        }
    }
};

inline simdint matrix_mul(simdint output, simd8 inputs, simd8 weights){
#ifdef VNNI
    return ADDMM(dpbusd_epi32(output, inputs, weights));
#else
    return ADDMM(add_epi32)(output, ADDMM(madd_epi16)(ADDMM(maddubs_epi16)(inputs, weights), simd16_set1(1)));
#endif
}

template<int input, int output>
struct Layer1{
    static const int full=input/nb16;
    static const int half=full/2;
    alignas(64) simd8 weights[input*output/nb8];
    alignas(64) simdint biases[output/nbint];
    static const int frame = 4;
    void forward(uint32_t x[input/frame], simdint y[output/nbint]) const{
        for(int i=0; i<output/nbint; i++){
            y[i] = biases[i];
        }
        for(int i=0; i<input/frame; i++){
            simd8 inp = ADDMM(set1_epi32)(x[i]);
            const int offset = i*output*frame;
            for(int j=0; j<output; j += nbint){
                y[j/nbint] = matrix_mul(y[j/nbint], inp, weights[(offset+j*frame)/nb8]);
            }
        }
    }
};

struct Layers{
    Layer1<HL_SIZE, L2> l1;
    Layer<L2, L3, true, QB> l2;
    Layer<L3, 1, false, QB*QB*QB> l3;
};

class Accumulator{
    void defstaterelated(const PositionState& state);
    void updatePieceOutComing(const int8_t mailbox[64], int piece, bool colorpiece, int square, bool remove, int removepos, const big sliders[3]);
    void updatePieceIncoming(const int8_t mailbox[64], int piece, bool colorpiece, int square, bool remove, int removepos, const big sliders[3]);
    void updatePiece(const int8_t mailbox[64], int piece, bool colorpiece, int square, bool remove, int removepos);
    template<bool enPassant=false, bool tworemove=false>
    void updateXrays(const int8_t mailbox[64], int square, bool remove, int removepos, int removepos2=-1);
    void getThreatUpdates(const PositionState& state1, const PositionState& state2, const Move& move);
    void applythreatsUpdates(Accumulator& accIn, bool side);
public:
    simd16 accs[4][HL_SIZE/nb16];
    bool Kside[2];
    bool side;
    bool pstrefresh;
    bool threatrefresh;
    big occupied;
    int idInputBucket[2];
    PositionState board;
    updateBuffer update;
    Accumulator(){}
    void reinit(const Move& move, const PositionState& state1, const PositionState& state2, Accumulator& prevAcc, bool side, bool mirror, Index sub1, Index add1, Index sub2=Index(), Index add2=Index());
    const simd16* operator[](int idx) const{
        return accs[idx];
    }
    simd16* operator[](int idx){
        return accs[idx];
    }
    void updateSelf(Accumulator& accIn, FinnyTables& finny);
};

class NNUE{
public:
    alignas(64) simd16 hlWeights[nbInputBuckets][INPUT_SIZE][HL_SIZE/nb16];
    alignas(64) simd8 threatWeights[THREAT_SIZE][HL_SIZE/nb8];
    alignas(64) simd16 hlBiases[HL_SIZE/nb16];
    Layers laterLayers[BUCKET];

    template<typename T=char>
    dbyte read_bytes(ifstream& file);
    // Helper to set individual elements in SIMD vectors
    void set_simd16_element(simd16& vec, int index, dbyte value);
    void set_simdint_element(simdint& vec, int index, int value);
    NNUE(string name);
    NNUE();
    void initAcc(Accumulator& accs) const;
    void init1Acc(oneAccumulator& accs) const;
    void initAcc(Accumulator& accs, bool color) const;
    int get_index(int piece, int c, int square) const;
    template<int f>
    void change1(Accumulator& accIn, bool pov, int index, int idInputBucket) const;
    template<int f>
    void change1acc(oneAccumulator& accIn, int index, int idInputBucket) const;
    template<int f>
    void addThreat(Accumulator& accIn, bool pov, int index) const;
    template<int f>
    void addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, int index) const;
    template<int f, int N>
    void addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, uint16_t* index) const;
    template<int N>
    void Threataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, uint16_t indexadds[N], uint16_t indexrems[N]) const;
    template<int f>
    void change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index, int idInputBucket) const;
    void move3(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap, int idInputBucket) const;
    void move2(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int idInputBucket) const;
    void move2In(oneAccumulator& accOut, int indexfrom, int indexto, int idInputBucket) const;
    void move4(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2, int idInputBucket) const;
    void updateStack(Accumulator* stack, int stackIndex, FinnyTables& finny) const;
    void calcThreats(Accumulator& accs, bool color, const PositionState& state) const;
    dbyte eval(Accumulator& accs, bool side, int idB) const;
};

inline const NNUE& globnnue = *reinterpret_cast<const NNUE*>(baseModel);
inline void updateBuffer::addThreat(const ThreatIndex& threat, const bool remove){
    threatUpdates[remove][nbThreats[remove]++] = threat;
}
#endif
