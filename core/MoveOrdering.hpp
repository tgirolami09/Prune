#ifndef MOVEORDERING_HPP
#define MOVEORDERING_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "Move.hpp"
#include "simd_definitions.hpp"
#include "tunables.hpp"

#ifdef DEBUG_MACRO
extern int quiethistSum;
extern double quiethistSquare;
extern int nbquietHist;
extern int capthistSum;
extern double capthistSquare;
extern int nbCaptHist;
#endif

class HelpOrdering{
    Move killers[maxDepth][2];
    int history[2][64][64];
    int captHist[2][nbPieces+4][6][64];
    int& getIndex(Move move, bool c);
    bool fastEq(Move a, Move b) const;
public:
    tunables parameters;
    void init(tunables& parameters);
    void addKiller(Move move, int depth, int relDepth, bool c);
    bool isKiller(Move move, int relDepth) const;
    int getHistoryScore(Move move, bool c) const;
    void updateHistory(int bonus, int& hist);
    void negUpdate(Move[maxMoves], int upto, bool c, int depth);

    int getMoveScore(Move move, bool c, int relDepths) const;
};

class Order{
public:
    Move moves[maxMoves];
    int nbMoves;
    #if defined(__AVX2__)
    // +8 pour avoir la place de rajouter 8 valeurs de padding de simd
    // 32-byte aligned memory
    alignas(32) int scores[maxMoves + 8];
    #else
    int scores[maxMoves];
    #endif
    int nbPriority;
    int pointer;
    big dangerPositions;
    bool sorted = false;
    Order();
    void swap(int idMove1, int idMove2);
    void init(bool c, int16_t moveInfoPriority, const HelpOrdering& history, ubyte relDepth, const GameState& state);
    void reinit(int16_t priorityMove);
    bool compareMove(int idMove1, int idMove2);
    Move pop_max(int& flag);
};
#endif