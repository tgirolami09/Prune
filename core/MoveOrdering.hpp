#ifndef MOVEORDERING_HPP
#define MOVEORDERING_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Move.hpp"
//#define COUNTER

class HelpOrdering{
    Move killers[maxDepth][2];
    int history[2][64][64];
#ifdef COUNTER
    int16_t counterMove[64*64];
#endif
    int& getIndex(Move move, bool c);
    bool fastEq(Move a, Move b) const;
public:
    void init();
    void addKiller(Move move, int depth, int relDepth, bool c, Move lastMove);
    bool isKiller(Move move, int relDepth) const;
    int getHistoryScore(Move move, bool c) const;
    void updateHistory(Move move, bool c, int bonus);
    void negUpdate(Move[maxMoves], int upto, bool c, int depth);
#ifdef COUNTER
    bool isCounter(Move move, Move lastMove) const;
#endif

    int getMoveScore(Move move, bool c, int relDepth, Move lastMove) const;
};

class Order{
public:
    Move moves[maxMoves];
    int nbMoves;
    int scores[maxMoves];
    int nbPriority;
    int pointer;
    ubyte flags[maxMoves]; // winning captures, non-losing quiet move, losing captures, losing quiet moves
    big dangerPositions;
    bool sorted = false;
    Order();
    void swap(int idMove1, int idMove2);
    void init(bool c, int16_t moveInfoPriority, int16_t PVMove, const HelpOrdering& history, ubyte relDepth, GameState& state, LegalMoveGenerator& generator, bool useSEE=true);
    void reinit(int16_t priorityMove);
    bool compareMove(int idMove1, int idMove2);
    Move pop_max();
};
#endif