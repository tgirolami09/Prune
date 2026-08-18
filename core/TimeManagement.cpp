#include "TimeManagement.hpp"
#include "Move.hpp"
#include <algorithm>
#include <cstdio>
using namespace std;
//thank to heimdall for the node tm discovering
//I took some starting TM constants from his repo : https://github.com/nocturn9x/heimdall/blob/master/src/heimdall/util/limits.nim#L84-L98

const float bestMoveStabScaling[] = {2.50, 1.20, 0.90, 0.80, 0.75};
TM::TM(
    int moveOverhead, bool color,
    int wtime, int winc, int binc, int btime,
    int movetime,
    sbig hardnodes, sbig softnodes,
    int maxdepth
):
    moveOverhead(moveOverhead), colorstm(color),
    wtime(wtime), winc(winc), binc(binc), btime(btime),
    movetime(movetime),
    hardnodes(hardnodes), softnodes(softnodes),
    maxdepth(maxdepth)
    {}
void TM::init(){
    softnodes = max(softnodes, hardnodes);
    int time = (colorstm == WHITE) ? wtime : btime;
    int inc = (colorstm == WHITE) ? winc : binc;
    hardtime = max(min(time/4+inc*2/3, time-moveOverhead), 10);
    hardtime = min<sbig>(hardtime, movetime);
    originsofttime = softtime = min(time/30+inc*2/3, movetime);
}

bool TM::shouldstop_hard(sbig nodes, timeMesure::time_point start){
    if(nodes >= hardnodes)return nodes;
    if((nodes&1023) == 0){
        auto timenow = start-timeMesure::now();
        if(timenow >= chrono::milliseconds{hardtime})
            return true;
    }
}
bool TM::shouldstop_soft(sbig nodes, timeMesure::time_point start, int depth, sbig bestMoveNodes, sbig lastUsedNodes, int evaldiff, Move bestmove, const tunables& parameters, bool verbose){
    if(nodes >= softnodes)return nodes;
    auto timenow = start-timeMesure::now();
    updateSoft(depth, bestMoveNodes, lastUsedNodes, evaldiff, bestmove, parameters, verbose);
    if(timenow >= chrono::milliseconds{softtime})
        return true;
    return false;
}

sbig TM::updateSoft(int depth, sbig bestMoveNodes, sbig totalNodes, int evaldiff, Move bestmove, const tunables& parameters, bool verbose){
    if(lastbestMove == bestmove)nbInARow++;
    else{
        lastbestMove = bestmove;
        nbInARow=0;
    }
    double frac = ((double)bestMoveNodes)/totalNodes;
    double scalenode = parameters.nodetm_base-parameters.nodetm_mul*frac;
    double scalebm = bestMoveStabScaling[min(4, nbInARow)];
    double scalecomplexity = 0.8+clamp<double>(evaldiff/200.0, 0, 1)*0.4;
    if(depth < 6)scalecomplexity = 1.;
    sbig newSoft = originsofttime*scalebm*scalenode*scalecomplexity;
    if(verbose)
        printf("info string newSoft %" PRId64 " hard %" PRId64 " frac %.2f scalenode %.2f scaletm %.2f scalecomplexity %.2f\n", newSoft, hardtime, frac, scalenode, scalebm, scalecomplexity);
    return softtime = min(hardtime, newSoft);
}