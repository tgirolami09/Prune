#include "TimeManagement.hpp"
#include <algorithm>
#include <cstdio>
#include "Move.hpp"
using namespace std;
// thank to heimdall for the node tm discovering
// I took some starting TM constants from his repo :
// https://github.com/nocturn9x/heimdall/blob/master/src/heimdall/util/limits.nim#L84-L98

const float bestMoveStabScaling[] = {2.50, 1.20, 0.90, 0.80, 0.75};
TM::TM(int moveOverhead, bool color, int wtime, int winc, int btime, int binc, int movetime,
       big hardnodes, big softnodes, int maxdepth)
    : moveOverhead(moveOverhead),
      colorstm(color),
      wtime(wtime),
      winc(winc),
      btime(btime),
      binc(binc),
      enabledtm(false),
      movetime(movetime),
      hardnodes(hardnodes),
      softnodes(softnodes),
      enablednodes(false),
      enabledtime(false),
      maxdepth(maxdepth) {}
void TM::init() {
    softnodes = min(softnodes, hardnodes);
    if (softnodes != MAX_BIG)
        enablednodes = true;
    if (movetime != INT_MAX) {
        hardtime = movetime;
        softtime = hardtime;
        enabledtime = true;
    }
    if (wtime != INT_MAX || btime != INT_MAX) {
        enabledtm = true;
        enabledtime = true;
        int time = (colorstm == WHITE) ? wtime : btime;
        int inc = (colorstm == WHITE) ? winc : binc;
        hardtime = min<sbig>(hardtime, max(min(time / 4 + inc * 2 / 3, time - moveOverhead), 10));
        originsofttime = softtime = min(time / 30 + inc * 2 / 3, movetime);
    }
}

bool TM::shouldstop_hard(big nodes, timeMesure::time_point start) {
    if (enablednodes && nodes >= hardnodes)
        return true;
    if (enabledtime && (nodes & 1023) == 0) {
        auto timenow = timeMesure::now() - start;
        if (timenow >= chrono::milliseconds{hardtime})
            return true;
    }
    return false;
}
bool TM::shouldstop_soft(big nodes, timeMesure::time_point start, int depth, big bestMoveNodes,
                         big lastUsedNodes, int evaldiff, Move bestmove, const tunables& parameters,
                         bool verbose) {
    if (enablednodes && nodes >= softnodes)
        return true;
    if (enabledtime) {
        auto timenow = timeMesure::now() - start;
        updateSoft(depth, bestMoveNodes, lastUsedNodes, evaldiff, bestmove, parameters, verbose);
        if (timenow >= chrono::milliseconds{softtime})
            return true;
    }
    return false;
}

sbig TM::updateSoft(int depth, big bestMoveNodes, big totalNodes, int evaldiff, Move bestmove,
                    const tunables& parameters, bool verbose) {
    if (!enabledtm)
        return softtime;
    if (lastbestMove == bestmove)
        nbInARow++;
    else {
        lastbestMove = bestmove;
        nbInARow = 0;
    }
    double frac = ((double)bestMoveNodes) / totalNodes;
    double scalenode = parameters.nodetm_base - parameters.nodetm_mul * frac;
    double scalebm = bestMoveStabScaling[min(4, nbInARow)];
    double scalecomplexity = 0.8 + clamp<double>(evaldiff / 200.0, 0, 1) * 0.4;
    if (depth < 6)
        scalecomplexity = 1.;
    sbig newSoft = originsofttime * scalebm * scalenode * scalecomplexity;
    if (verbose)
        printf("info string newSoft %" PRId64 " hard %" PRId64
               " frac %.2f scalenode %.2f scaletm %.2f scalecomplexity %.2f\n",
               newSoft, hardtime, frac, scalenode, scalebm, scalecomplexity);
    return softtime = min(hardtime, newSoft);
}