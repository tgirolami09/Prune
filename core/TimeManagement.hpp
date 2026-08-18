#ifndef TIME_MANAGEMENT_HPP
#define TIME_MANAGEMENT_HPP
#include "Const.hpp"
#include "tunables.hpp"
#include "Move.hpp"
#include <climits>
#include <chrono>
using timeMesure=chrono::high_resolution_clock;
class TM{
public:
    int wtime, winc, btime, binc;
    sbig hardnodes, softnodes;
    int movetime;
    int maxdepth;
    bool colorstm;
    int moveOverhead;
    sbig hardtime;
    sbig softtime;
    sbig originsofttime;
    Move lastbestMove;
    int nbInARow;
    TM(
        int moveOverhead=0, bool color=WHITE,
        int wtime=INT_MAX, int winc=INT_MAX, int btime=INT_MAX, int binc=INT_MAX,
        int movetime=INT_MAX,
        sbig hardnodes=MAX_BIG, sbig softnodes=MAX_BIG,
        int maxdepth=maxDepth
    );
    void init();
    bool shouldstop_hard(sbig nodes, timeMesure::time_point start);
    bool shouldstop_soft(sbig nodes, timeMesure::time_point start, int depth, sbig bestMoveNodes, sbig lastUsedNodes, int evaldiff, Move bestmove, const tunables& parameters, bool verbose);
    sbig updateSoft(int depth, sbig bestMoveNodes, sbig totalNodes, int evaldiff, Move bestmove, const tunables& parameters, bool verbose);
};

#endif