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
#include <vector>
#include "tunables.hpp"
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <string>
#include <vector>
#include <thread>
#ifdef NUMA_BUILD
#include "helper_numa.hpp"
#endif
#define MoveScore pair<int, Move>
#define bestMoveResponse tuple<Move, Move, int, vector<depthInfo>>

#ifdef DEBUG_MACRO
extern int
    nmpVerifAllNode,
    nmpVerifCutNode,
    nmpVerifPassCutNode,
    nmpVerifPassAllNode;
#endif

class BestMoveFinder;

class usefull{
private:
    class LINE{
    public:
        int cmove;
        int16_t argMoves[maxDepth];
    };
    struct StackCase{
        Order order;
        int static_score;
        int raw_eval;
    };
public:
    LegalMoveGenerator generator;
    StackCase stack[maxDepth];
    LINE PVlines[maxDepth];
    IncrementalEvaluator eval;
    atomic<sbig> nodes;
    atomic<sbig> bestMoveNodes;
    atomic<int> seldepth;
    atomic<sbig> nbCutoff, nbFirstCutoff;
    Move rootBest;
    bool mainThread;
    HelpOrdering history;
    corrhists correctionHistory;
    int min_nmp_ply=0;
    usefull(const GameState& state, tunables& parameters);
    usefull();
    void reinit(const GameState& state);
    string PVprint(LINE pvLine);
    void transfer(int relDepth, Move move);
    void beginLine(int relDepth);
    void beginLineMove(int relDepth, Move move);
    void resetLines();
};

struct Record{
    sbig nodes;
    sbig nbFirstCutoff;
    sbig nbCutoff;
};

class HelperThread{
public:
    usefull local;
    GameState localState;
    thread t;
    atomic<bool> running;
    mutex mtx;
    condition_variable cv;
    int ans;
    int relDepth, limitWay;
    void launch(int relDepth, int limitWay);
    void wait_thread();
};

class threadGroup{
public:
    vector<HelperThread> helperThreads;
    threadGroup(int nb);
    threadGroup();
    int size() const;
    void clear_helpers();
    void reset(int nT, BestMoveFinder* addr);
    HelperThread& operator[](int idx);
};

#ifdef NUMA_BUILD
class numaNode{
public:
    threadGroup helperThreads;
    numaNode();
    numaNode(int x);
    int size();
    void reset(int nT, BestMoveFinder* addr);
    void clear_helpers();
    void clear();
    HelperThread& operator[](int idx);
};

class numaGroup{
public:
    numa numaHelper;
    int numnode;
    vector<int> nbTperN;
    vector<numaNode> nodes;
    numaGroup(int nb);
    void reset(int nT, BestMoveFinder* addr);
    int size();
    void clear_helpers();
    void clear();
    int idnode(int idThread);
    HelperThread& operator[](int idx);
};
#endif

using timeMesure=chrono::high_resolution_clock;
//Class to find the best in a situation
class BestMoveFinder{
    unordered_map<uint64_t,PolyglotEntry> book;

    //Returns the best move given a position and time to use
    transpositionTable transposition;
public:
    std::atomic<bool> running;
    BestMoveFinder(int memory, bool mute=false);
    BestMoveFinder();
    sbig hardBound;
    timeMesure::time_point startSearch;
    chrono::milliseconds hardBoundTime;
    ~BestMoveFinder();
    void stop();
    tunables parameters;
private:
    usefull localSS;
#ifdef NUMA_BUILD
    numaGroup helperThreads;
#else
    threadGroup helperThreads;
#endif
    atomic<bool> smp_abort, smp_end;
    void clear_helpers();
    chrono::nanoseconds getElapsedTime();
    template<int limitWay, bool isPV, bool isCalc>
    int quiescenceSearch(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    int startRelDepth;
    template<bool isPV, int limitWay>
    inline int Evaluate(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    bool verbose;
    template <bool isPV, int limitWay, bool isRoot=false>
    int negamax(usefull& ss, const int depth, GameState& state, int alpha, const int beta, const int relDepth, bool cutnode, const int16_t excludedMove=nullMove.moveInfo);
    void updatemainSS(usefull& ss, Record& oldss);
public:
    void launchSMP(int idThread);
    template<int limitWay>
    bestMoveResponse iterativeDeepening(usefull& ss, GameState& state, TM tm, int actDepth);
    template <int limitWay=0>
    bestMoveResponse bestMove(GameState& state, TM tm, vector<Move> movesFromRoot, bool verbose=true);
    template <int limitWay=0>
    bestMoveResponse goState(GameState& state, TM tm, bool verbose, int actDepth);
    int testQuiescenceSearch(GameState& state);
    void clear();
    void reinit(size_t count);
    void setThreads(int nbThreads);
    void aging();
};


class Perft{
public:
    Move stack[100][maxMoves];
    LegalMoveGenerator generator;
    Perft();
    big visitedNodes;
    big _perft(GameState& state, ubyte depth);
    big perft(GameState& state, ubyte depth, bool verbose=true);
    void reinit(size_t count);
};
#endif