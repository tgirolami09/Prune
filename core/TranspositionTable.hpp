#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "GameState.hpp"
#include <climits>
#include <vector>

class __attribute__((packed)) infoScore{
public:
    int16_t score,
            raw_eval;
    ubyte typeNode;
    int16_t bestMoveInfo;
    ubyte depth;
    uint32_t hash;
};
static_assert(sizeof(infoScore) == 12, "size of infoScore should be 12");

const int INVALID = INT_MAX;
class transpositionTable{
public:
    infoScore* table;
    big modulo;
    int rewrite=0;
    int place=0;
    transpositionTable(size_t count);

    inline int storedScore(int alpha, int beta, int depth, const infoScore& entry) const;

    int get_eval(const infoScore& entry, int alpha, int beta, ubyte depth) const;
    infoScore& getEntry(const GameState& state, bool& ttHit);

    int16_t getMove(const infoScore& entry) const;

    void push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth, int16_t raw_eval);
    void clearRange(big start, big end);
    void prefetch(const GameState& state);
    void clear();
    void reinit(size_t count);
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