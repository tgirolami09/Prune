#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "GameState.hpp"
#include <climits>
#include <cstdint>
#include <vector>
const int maxAge = 0b11111;
using resHash=uint16_t;
class __attribute__((packed)) infoScore{
public:
    int16_t score,
            raw_eval;
    ubyte flag;
    int16_t bestMoveInfo;
    ubyte depth;
    resHash hash;
    int typeNode() const;
    int age() const;
    void setFlag(int typeNode, int age, bool pv);
    bool tt_pv() const;
};
static_assert(sizeof(infoScore) == 10, "size of infoScore should be 10");
const int clusterByte=32;
const int clusterSize=clusterByte/sizeof(infoScore);
class Cluster{
public:
    infoScore entries[clusterSize];
    ubyte padding[clusterByte-clusterSize*sizeof(infoScore)];
    infoScore& probe(resHash hash, bool& ttHit);
    void push(infoScore& entry, int curAge);
};
static_assert(sizeof(Cluster) == clusterByte, "size of cluster should be 32");

const int INVALID = INT_MAX;
class transpositionTable{
public:
    Cluster* table;
    big modulo;
    int rewrite=0;
    int place=0;
    int age;
    transpositionTable(size_t count);

    inline int storedScore(int alpha, int beta, int depth, const infoScore& entry) const;

    int get_eval(const infoScore& entry, int alpha, int beta, ubyte depth) const;
    infoScore& getEntry(const GameState& state, bool& ttHit);

    int16_t getMove(const infoScore& entry) const;

    void push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth, int16_t raw_eval, bool is_pv);
    void clearRange(big start, big end);
    void prefetch(const GameState& state);
    void clear();
    void reinit(size_t count);
    void aging();
};

class perftMem{
public:
    big hash;
    big leefs;
    ubyte depth;
};
class TTperft{
public:
    vector<perftMem> mem;
    int modulo;
    TTperft(int alloted_mem);
    void push(perftMem eval);
    int get_eval(big hash, int depth);
    void clear();
    void reinit(int count);
    void clearMem();
};

#endif