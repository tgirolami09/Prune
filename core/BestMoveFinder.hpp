#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>
#include "Const.hpp"
#include "Evaluator.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Move.hpp"
#include "MoveOrdering.hpp"
#include "TablebaseProbe.hpp"
#include "TimeManagement.hpp"
#include "TranspositionTable.hpp"
#include "numa.hpp"
#include "tunables.hpp"
#define MoveScore pair<int, Move>
#define bestMoveResponse tuple<Move, Move, int, vector<depthInfo>>
#ifdef DEBUG_MACRO
#include "stats_helpers.hpp"
extern int nmpVerifAllNode, nmpVerifCutNode, nmpVerifPassCutNode, nmpVerifPassAllNode;
extern StatVar<sbig, maxHistory * 2, -maxHistory * 2> quiethistPostStat;
extern StatVar<sbig, maxHistory, -maxHistory> capthistPostStat;
#endif

// Class to find the best in a situation
class BestMoveFinder {
    class usefull {
       private:
        class LINE {
           public:
            int cmove;
            int16_t argMoves[maxDepth];
        };
        struct StackCase {
            Order order;
            LegalMoveGenerator generator;
            Move searchedMoves[maxMoves];
            int static_score;
            int raw_eval;
            PositionSnapshot snap;
        };

       public:
        StackCase stack[maxDepth + 1];
        LINE PVlines[maxDepth];
        IncrementalEvaluator eval;
        atomic<sbig> nodes;
        atomic<sbig> bestMoveNodes;
        atomic<int> seldepth;
        bool let_run;
        sbig tbHits;
        int idThread;
        Move rootBest;
        bool mainThread;
        HelpOrdering history;
        int searchedMoves = 0;
        int min_nmp_ply = 0;
        usefull(const GameState& state, const tunables& parameters, const NNUE& nnue);
        usefull();
        void reinit(const GameState& state, const NNUE& nnue);
        string PVprint(LINE pvLine);
        void transfer(int relDepth, Move move);
        void beginLine(int relDepth);
        void beginLineMove(int relDepth, Move move);
        void resetLines();
        inline bool stop(bool stop_flags) { return !let_run && stop_flags; }
    };

    struct Record {
        sbig nodes;
        sbig tbHits;
    };

    class HelperThread {
       public:
        usefull local;
        GameState localState;
        thread t;
        bool running;
        mutex mtx;
        condition_variable cv;
        atomic<bool> isready;
        int ans;
        int relDepth;
        void launch(int relDepth);
        void wait_thread();
    };

    struct Shared {
        corrhists correctionHistory;
    };
    vector<Shared> shareds;
    // Returns the best move given a position and time to use
    transpositionTable transposition;
    int thread0;

   public:
    std::atomic<int> stop_flag;
    bool minimal = false;
    BestMoveFinder(int memory, int baseThread = -1);
    BestMoveFinder();
    sbig hardBound;
    ~BestMoveFinder();
#ifdef TUNE
    tunables parameters;
#else
    static constexpr tunables parameters{};
#endif
   private:
    usefull localSS;
    vector<HelperThread> helperThreads;
    atomic<bool> smp_abort, smp_end;
    void clear_helpers();
    timeMesure::time_point startSearch;
    TM globtm;
    chrono::nanoseconds getElapsedTime();
    Move wdlFilterMoveInfos[maxMoves];
    int wdlFilterNb;
    template <bool isPV, bool isCalc>
    int quiescenceSearch(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    int startRelDepth;
    template <bool isPV>
    inline int Evaluate(usefull& ss, GameState& state, int alpha, int beta, int relDepth);
    bool verbose;
    template <bool isPV, bool isRoot = false>
    int negamax(usefull& ss, const int depth, GameState& state, int alpha, const int beta,
                const int relDepth, bool cutnode, const Move excludedMove = nullMove);
    void launchSMP(int idThread);
    void updatemainSS(usefull& ss, Record& oldss);

   public:
    bestMoveResponse iterativeDeepening(usefull& ss, GameState& state, TM tm, int actDepth);
    bestMoveResponse bestMove(GameState& state, TM tm, vector<Move> movesFromRoot,
                              bool verbose = true);
    template <bool set = false>
    bestMoveResponse goState(GameState& state, TM tm, bool verbose, int actDepth);
    int testQuiescenceSearch(GameState& state);
    void clear();
    void reinit(size_t count);
    void setThreads(int nbThreads);
    void aging();
};

class Perft {
   public:
    Move stack[100][maxMoves];
    LegalMoveGenerator generator;
    Perft();
    big visitedNodes;
    template <bool bulk>
    big _perft(GameState& state, ubyte depth);
    template <bool bulk>
    big perft(GameState& state, ubyte depth, bool verbose = true);
    void reinit(size_t count);
};
#endif