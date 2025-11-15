#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include "Const.hpp"
#include "TranspositionTable.hpp"
#include "TimeManagement.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include "LegalMoveGenerator.hpp"
#include "MoveOrdering.hpp"
#include "loadpolyglot.hpp"
#include <chrono>
#include <atomic>
#include <string>
#include <vector>
#define MoveScore pair<int, Move>
#define bestMoveResponse tuple<Move, Move, int, vector<depthInfo>>

int compScoreMove(const void* a, const void*b);

int fromTT(int score, int rootDist);

int absoluteScore(int score, int rootDist);

class LINE{
public:
    int cmove;
    int16_t argMoves[maxDepth];
};

struct StackCase{
    Order order;
    int static_score;
};

string scoreToStr(int score);

//Class to find the best in a situation
class BestMoveFinder{
    unordered_map<uint64_t,PolyglotEntry> book;

    //Returns the best move given a position and time to use
    transpositionTable transposition;
    HelpOrdering history;
    StackCase stack[maxDepth];
public:
    std::atomic<bool> running;
    IncrementalEvaluator eval;
    BestMoveFinder(int memory, bool mute=false);

    sbig hardBound;
    using timeMesure=chrono::high_resolution_clock;
    timeMesure::time_point startSearch;
    chrono::milliseconds hardBoundTime;
    void stop();

private:
    chrono::nanoseconds getElapsedTime();

    LegalMoveGenerator generator;
    sbig nodes;
    int nbCutoff;
    int nbFirstCutoff;
    int seldepth;
    int bestMoveNodes;
    template<int limitWay, bool isPV>
    int quiescenceSearch(GameState& state, int alpha, int beta, int relDepth);
    int startRelDepth;
    corrhists correctionHistory;
    //PV making
    LINE PVlines[maxDepth]; //store only the move info, because it only need that to print the pv

    string PVprint(LINE pvLine);
    LINE lastPV;
    void transferLastPV();
    void transfer(int relDepth, Move move);
    void beginLine(int relDepth);
    void beginLineMove(int relDepth, Move move);
    void resetLines();
    int16_t getPVMove(int relDepth);
    enum{PVNode=0, CutNode=1, AllNode=-1};
    template<int nodeType, int limitWay, bool mateSearch>
    inline int Evaluate(GameState& state, int alpha, int beta, int relDepth);
    Move rootBestMove;
    bool verbose;
    template <int nodeType, int limitWay, bool mateSearch, bool isRoot=false>
    int negamax(const int depth, GameState& state, int alpha, const int beta, const int relDepth);
public:
    template <int limitWay=0>
    bestMoveResponse bestMove(GameState& state, TM tm, vector<Move> movesFromRoot, bool verbose=true, bool mateHardBound=true);
    int testQuiescenceSearch(GameState& state);
    void clear();
    void reinit(size_t count);
};


class Perft{
public:
    Move stack[100][maxMoves];
    LegalMoveGenerator generator;
    Perft();
    big visitedNodes;
    big _perft(GameState& state, ubyte depth);
    big perft(GameState& state, ubyte depth);
    void reinit(size_t count);
};
#endif