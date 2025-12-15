#ifndef MOVEORDERING_HPP
#define MOVEORDERING_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Move.hpp"

class HelpOrdering{
    Move killers[maxDepth][2];
    int history[2][64][64];
    int& getIndex(Move move, bool c);
    bool fastEq(Move a, Move b) const;
public:
    void init();
    void addKiller(Move move, int depth, int relDepth, bool c);
    bool isKiller(Move move, int relDepth) const;
    int getHistoryScore(Move move, bool c) const;
    void updateHistory(Move move, bool c, int bonus);
    void negUpdate(Move[maxMoves], int upto, bool c, int depth);

    int getMoveScore(Move move, bool c, int relDepths) const;
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
    void init(bool c, int16_t moveInfoPriority, const HelpOrdering& history, ubyte relDepth, const GameState& state, bool useSEE=true);
    void reinit(int16_t priorityMove);
    bool compareMove(int idMove1, int idMove2);
    Move pop_max(int& flag);
};
#endif