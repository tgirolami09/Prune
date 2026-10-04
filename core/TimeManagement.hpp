#ifndef TIME_MANAGEMENT_HPP
#define TIME_MANAGEMENT_HPP
#include <chrono>
#include <climits>
#include "Const.hpp"
#include "Move.hpp"
#include "tunables.hpp"
using timeMesure = chrono::high_resolution_clock;
class TM {
   public:
    int moveOverhead;
    bool colorstm;
    int wtime, winc, btime, binc;
    bool enabledtm;
    int movetime;
    u64 hardnodes, softnodes;
    bool enablednodes;
    bool enabledtime;
    int maxdepth;
    i64 hardtime;
    i64 softtime;
    i64 originsofttime;
    Move lastbestMove;
    int nbInARow;
    TM(int moveOverhead = 0, bool color = White, int wtime = INT_MAX, int winc = INT_MAX,
       int btime = INT_MAX, int binc = INT_MAX, int movetime = INT_MAX, u64 hardnodes = MAX_BIG,
       u64 softnodes = MAX_BIG, int maxdepth = maxDepth);
    void init();
    bool shouldstop_hard(u64 nodes, timeMesure::time_point start);
    bool shouldstop_soft(u64 nodes, timeMesure::time_point start, int depth, u64 bestMoveNodes,
                         u64 lastUsedNodes, int evaldiff, Move bestmove, const tunables& parameters,
                         bool verbose);
    i64 updateSoft(int depth, u64 bestMoveNodes, u64 totalNodes, int evaldiff, Move bestmove,
                   const tunables& parameters, bool verbose);
};

#endif