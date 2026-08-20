#ifndef NNUE_CPP
#define NNUE_CPP
#include <cstdint>
#include <fstream>
#include "Const.hpp"
#include "GameState.hpp"
#include "Move.hpp"
#include "embeder.hpp"
#include "simd_definitions.hpp"
#include <cstdint>
#include <fstream>
#include <array>
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

constexpr inline int ilog2c(int n) {
    return (31 ^ __builtin_clz(n)) + !!(n & (n - 1));
}

constexpr inline int _abs(int x) {
    return x < 0 ? -x : x;
}

const int maxThreatUpdates = 80;

const int INPUT_SIZE = 11 * 64;  // merged king planes
const int THREAT_SIZE = 60144;

constexpr int QA = 255;
constexpr int QB = 128;
constexpr int QC = 64;
constexpr int FT_BITS = 9;
constexpr int FT_LSHIFT = 16 - FT_BITS;

constexpr int QA_bits = ilog2c(QA);
constexpr int QB_bits = ilog2c(QB);
constexpr int QC_bits = ilog2c(QC);
constexpr int L1shift = _abs(16 + QC_bits - FT_LSHIFT - QA_bits * 2 - QB_bits);

const int BUCKET = 8;
const int nbInputBuckets = 16;

const int L1 = 640;
const int L2 = 16;
const int L3 = 32;

const int SCALE = 283;
const int inputBuckets[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9,  9,  10, 10, 11, 11,
    12, 12, 13, 13, 12, 12, 13, 13, 14, 14, 15, 15, 14, 14, 15, 15,

};
const int DIVISOR=32/BUCKET;
#ifdef DEBUG_MACRO
extern StatVar<sbig, L1/4, 0> nnzCount;
#endif
static_assert(L1%nb<16> == 0, "L1 size needs to be a multiple of nb<16>");

int getInputBucket(int Kpos, bool side, bool mirror);
class NNUE;
#ifdef __ARM_NEON__
using simdsmol=uint16x8_t;
#else
using simdsmol=__m128i;
#endif
static inline simdsmol loadsimd(const uint16_t* pointer){
#ifdef __ARM_NEON__
    return vld1q_u16(pointer);
#else
    return _mm_load_si128(reinterpret_cast<const __m128i*>(pointer));
#endif
}
static inline void storesimd(uint16_t* pointer, simdsmol vec){
#ifdef __ARM_NEON__
    vst1q_u16(pointer, vec);
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pointer), vec);
#endif
}
static inline simdsmol simdsmoladd(simdsmol a, simdsmol b){
#ifdef __ARM_NEON__
    return vpaddlq_u16(pointer);
#else
    return _mm_add_epi16(a, b);
#endif
}

static inline simdsmol simdset(uint16_t x){
#ifdef __ARM_NEON__
    return vdupq_n_u16(x);
#else
    return _mm_set1_epi16(x);
#endif
}

// code from https://rmeguro.com/blogs/sparse-nnue.html
struct SparseIterator{
    alignas(16) uint16_t indices[L1/4] = {0};
    int count_ = 0;
    __m128i offset = _mm_setzero_si128();
    alignas(16) static constexpr array<uint16_t, 256*8> nonzero_idx = [] {
        array<uint16_t, 256*8> idx{};

        for (int32_t i = 0; i < 256; i++) {
            int32_t nnz = 0;
            for (uint8_t mask = i; mask != 0; mask &= mask - 1)
                idx[i*8+(nnz++)] = __countr_zero(mask);
        }

        return idx;
    }();
public:
    int count() const{
        return count_;
    };
    int index(int nnzidx) const{
        return indices[nnzidx];
    }
    void add_nonzero(simd<8> ft_out0, simd<8> ft_out1) {
        // VecU8 = __m256i on AVX2, __m512i on AVX512
        // ALIGNMENT = 32 on AVX2 or 64 on AVX512
        constexpr int32_t regw32 = SIZE / 8 / sizeof(int32_t);
        constexpr int32_t n_mask_bytes = 2 * regw32 / 8;

        uint32_t full_mask = (nonzero_mask(ft_out1) << regw32) | nonzero_mask(ft_out0);
        for (int32_t i = 0; i < n_mask_bytes; i++) {
            // get offset of up to 8 nonzero blocks
            const uint8_t mask = full_mask & 0xFF;
            full_mask >>= 8;
            const auto idxs = simdsmoladd(
                offset,
                loadsimd(&nonzero_idx[mask*8])
            );

            storesimd(&indices[count_], idxs);
            offset = simdsmoladd(offset, simdset(8));
            count_ += __popcount(mask);
        }
    }
};

class Index {
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
class ThreatIndex {
   public:
    Index from;
    Index to;
    ThreatIndex(Index _from, Index _to);
    ThreatIndex(int fromsquare, int frompiece, int fromcolor, int tosquare, int topiece,
                int tocolor);
    ThreatIndex();
    bool isexcluded() const;
    bool issemiexcluded() const;
    void swap();
    ThreatIndex rswap() const;
    operator int() const;
    ThreatIndex changepov(bool needs) const;
    ThreatIndex mirror(bool needs) const;
    void print() const;
    ThreatIndex swapSemiExcluded() const {
        return issemiexcluded() ? ThreatIndex(to, from) : ThreatIndex(from, to);
    }
    ThreatIndex swapExcluded() const {
        return isexcluded() ? ThreatIndex(to, from) : ThreatIndex(from, to);
    }
};
using oneAccumulator = simd<16>[L1 / nb<16>];
class FinnytableNormal {
   public:
    big bitboards[8];
    oneAccumulator accs;
};

class FinnyTables {
   public:
    FinnytableNormal normals[nbInputBuckets * 4];
    void init(const NNUE& nnue);
};

class updateBuffer {
   public:
    Index add1[2], add2[2];
    Index sub1[2],
        sub2[2];  // each pieces provoque a change in black and white pov
    int nbThreats[2];
    ThreatIndex threatUpdates[2][32];
    bool dirty;
    Move deferredMove;
    int type;
    updateBuffer();
    void reset(Index sub1, Index add1, Index sub2, Index add2);
    void addThreat(const ThreatIndex& threat, const bool remove);
    void print();
};

static const inline simd<16> zero_16 = simd16_zero();
static const inline simd<32> zero_32 = simdint_zero();
static const inline simd<16> A_16 = simd16_set1(QA);

template <int input, int output, int _clamp>
struct midLayer {
    simd<32> weights[input][output / nb<32>];
    simd<32> biases[output / nb<32>];
    void forward(const int x[input], simd<32> y[output / nb<32>]) const;
};

template <int input, int output>
struct lastLayer {
    simd<32> weights[output][input / nb<32>];
    int biases[output];
    void forward(const simd<32> x[input / nb<32>], int y[output]) const;
};

template<int input, int output>
struct Layer1{
    alignas(64) simd<8> weights[input*output/nb<8>];
    alignas(64) simd<32> biases[output/nb<32>];
    void forward(const uint32_t x[input/I8inI32], simd<32> y[output/nb<32>], const SparseIterator& si) const;
};

struct Layers {
    Layer1<L1, L2> l1;
    midLayer<L2, L3, QC * QC * QC> l2;
    lastLayer<L3, 1> l3;
};

class Accumulator {
    void defstaterelated(const PositionState& state);
    void updatePieceOutComing(const PositionState& state, int piece, bool colorpiece, int square,
                              bool remove, int removepos, const big sliders[3]);
    void updatePieceIncoming(const PositionState& state, int piece, bool colorpiece, int square,
                             bool remove, int removepos, const big sliders[3]);
    void updatePiece(const PositionState& state, int piece, bool colorpiece, int square,
                     bool remove, int removepos);
    template <bool enPassant = false, bool tworemove = false>
    void updateXrays(const PositionState& state, int square, bool remove, int removepos,
                     int removepos2 = -1);
    void getThreatUpdates(const PositionState& state1, const PositionState& state2,
                          const Move& move);
    void applythreatsUpdates(Accumulator& accIn, bool side, const NNUE& nnue);

   public:
    simd<16> accs[4][L1 / nb<16>];
    bool Kside[2];
    bool side;
    bool pstrefresh;
    bool threatrefresh;
    big occupied;
    int idInputBucket[2];
    PositionState board;
    updateBuffer update;
    Accumulator() {}
    void reinit(const Move& move, const PositionState& state1, const PositionState& state2,
                Accumulator& prevAcc, bool side, bool mirror, Index sub1, Index add1,
                Index sub2 = Index(), Index add2 = Index());
    const simd<16>* operator[](int idx) const { return accs[idx]; }
    simd<16>* operator[](int idx) { return accs[idx]; }
    void updateSelf(Accumulator& accIn, FinnyTables& finny, const NNUE& nnue);
};

class NNUE {
   public:
    alignas(64) simd<16> hlWeights[nbInputBuckets][INPUT_SIZE][L1 / nb<16>];
    alignas(64) simd<8> threatWeights[THREAT_SIZE][L1 / nb<8>];
    alignas(64) simd<16> hlBiases[L1 / nb<16>];
    Layers laterLayers[BUCKET];

    template <typename T = char>
    dbyte read_bytes(ifstream& file);
    // Helper to set individual elements in SIMD vectors
    void set_simd16_element(simd<16>& vec, int index, dbyte value);
    void set_simdint_element(simd<32>& vec, int index, int value);
    NNUE(string name);
    NNUE();
    void initAcc(Accumulator& accs) const;
    void init1Acc(oneAccumulator& accs) const;
    void initAcc(Accumulator& accs, bool color) const;
    int get_index(int piece, int c, int square) const;
    template <int f>
    void change1(Accumulator& accIn, bool pov, int index, int idInputBucket) const;
    template <int f>
    void change1acc(oneAccumulator& accIn, int index, int idInputBucket) const;
    template <int f>
    void addThreat(Accumulator& accIn, bool pov, int index) const;
    template <int f>
    void addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, int index) const;
    template <int f, int N>
    void addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, uint16_t* index) const;
    template <int N>
    void Threataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, uint16_t indexadds[N],
                      uint16_t indexrems[N]) const;
    template <int f>
    void change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index,
                 int idInputBucket) const;
    void move3(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto,
               int indexcap, int idInputBucket) const;
    void move2(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto,
               int idInputBucket) const;
    void move2In(oneAccumulator& accOut, int indexfrom, int indexto, int idInputBucket) const;
    void move4(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom1,
               int indexto1, int indexfrom2, int indexto2, int idInputBucket) const;
    void updateStack(Accumulator* stack, int stackIndex, FinnyTables& finny) const;
    void calcThreats(Accumulator& accs, bool color, const PositionState& state) const;
    dbyte eval(Accumulator& accs, bool side, int idB) const;
};

inline const NNUE& globnnue = *reinterpret_cast<const NNUE*>(baseModel);
inline void updateBuffer::addThreat(const ThreatIndex& threat, const bool remove) {
    threatUpdates[remove][nbThreats[remove]++] = threat;
}
#endif
