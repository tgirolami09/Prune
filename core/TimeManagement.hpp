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
    big hardnodes, softnodes;
    bool enablednodes;
    bool enabledtime;
    int maxdepth;
    sbig hardtime;
    sbig softtime;
    sbig originsofttime;
    Move lastbestMove;
    int nbInARow;
    TM(int moveOverhead = 0, bool color = WHITE, int wtime = INT_MAX, int winc = INT_MAX,
       int btime = INT_MAX, int binc = INT_MAX, int movetime = INT_MAX, big hardnodes = MAX_BIG,
       big softnodes = MAX_BIG, int maxdepth = maxDepth);
    void init();
    bool shouldstop_hard(big nodes, timeMesure::time_point start);
    bool shouldstop_soft(big nodes, timeMesure::time_point start, int depth, big bestMoveNodes,
                         big lastUsedNodes, int evaldiff, Move bestmove, const tunables& parameters,
                         bool verbose);
    sbig updateSoft(int depth, big bestMoveNodes, big totalNodes, int evaldiff, Move bestmove,
                    const tunables& parameters, bool verbose);
};

#endif