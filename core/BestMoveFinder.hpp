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
#include "tunables.hpp"
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <string>
#include <vector>
#include <thread>
#define MoveScore pair<int, Move>
#define bestMoveResponse tuple<Move, Move, int, vector<depthInfo>>

#ifdef DEBUG_MACRO
extern int
    nmpVerifAllNode,
    nmpVerifCutNode,
    nmpVerifPassCutNode,
    nmpVerifPassAllNode;
#endif

using timeMesure=chrono::high_resolution_clock;
//Class to find the best in a situation
class BestMoveFinder{
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
        sbig nodes;
        sbig bestMoveNodes;
        int seldepth;
        int nbCutoff, nbFirstCutoff;
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

    class HelperThread{
    public:
        usefull local;
        GameState localState;
        thread t;
        bool running;
        mutex mtx;
        condition_variable cv;
        int ans;
        int depth, alpha, beta, relDepth, limitWay;
        void launch(int depth, int alpha, int beta, int relDepth, int limitWay);
        void wait_thread();
    };
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
    HelperThread* helperThreads;
    atomic<bool> smp_abort, smp_end;
    void clear_helpers();
    chrono::nanoseconds getElapsedTime();
    template<int limitWay, bool isPV, bool isCalc>
    int quiescenceSearch(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    int startRelDepth;
    enum{PVNode=0, CutNode=1, AllNode=-1};
    template<int nodeType, int limitWay, bool mateSearch>
    inline int Evaluate(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    bool verbose;
    template <int nodeType, int limitWay, bool mateSearch, bool isRoot=false>
    int negamax(usefull& ss, const int depth, GameState& state, int alpha, const int beta, const int relDepth, const int16_t excludedMove=nullMove.moveInfo);
    template<bool mateSearch>
    int launchSearch(int limitWay, HelperThread& ss);
    void launchSMP(int idThread);
public:
    template <int limitWay=0>
    bestMoveResponse bestMove(GameState& state, TM tm, vector<Move> movesFromRoot, bool verbose=true);
    template <int limitWay=0>
    bestMoveResponse goState(GameState& state, TM tm, bool verbose, int actDepth);
    int testQuiescenceSearch(GameState& state);
    void clear();
    void reinit(size_t count);
    void setThreads(int nbThreads);
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