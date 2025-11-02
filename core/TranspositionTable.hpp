#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "GameState.hpp"
#include <climits>
#include <vector>

class __attribute__((packed)) infoScore{
public:
    int score;
    ubyte typeNode;
    int16_t bestMoveInfo;
    ubyte depth;
    big hash;
};
const int INVALID = INT_MAX;
class transpositionTable{
public:
    vector<infoScore> byDepth;
    vector<infoScore> always;
    int modulo;
    int rewrite=0;
    int place=0;
    transpositionTable(size_t count);

    inline int storedScore(int alpha, int beta, int depth, const infoScore& entry);

    int get_eval(const GameState& state, int alpha, int beta, ubyte depth, int16_t& best);

    int16_t getMove(const GameState& state);

    void push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth);
    void clear();
    void reinit(int count);
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