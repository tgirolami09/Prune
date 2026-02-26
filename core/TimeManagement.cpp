#include "TimeManagement.hpp"
#include "Move.hpp"
#include <cstdio>
//thank to heimdall for the node tm discovering
//I took some starting TM constants from his repo : https://github.com/nocturn9x/heimdall/blob/master/src/heimdall/util/limits.nim#L84-L98

const float bestMoveStabScaling[] = {2.50, 1.20, 0.90, 0.80, 0.75};
TM::TM(int _softBound, int _hardBound):softBound(_softBound), hardBound(_hardBound), enableUpdate(false), lastbestMove(nullMove.moveInfo), nbInARow(0){}
TM::TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color):enableUpdate(true), lastbestMove(nullMove.moveInfo), nbInARow(0){
    int time = (color == WHITE) ? wtime : btime;
    int inc = (color == WHITE) ? winc : binc;
    hardBound = max(min(time/8+inc*2/3, time-moveOverhead), 10);
    originLowerBound = softBound = time/30+inc*2/3;
}

sbig TM::updateSoft(sbig bestMoveNodes, sbig totalNodes, int16_t bestmove, tunables& parameters, bool verbose){
    if(!enableUpdate)return softBound;
    if(lastbestMove == bestmove)nbInARow++;
    else{
        lastbestMove = bestmove;
        nbInARow=0;
    }
    double frac = ((double)bestMoveNodes)/totalNodes;
    double scalenode = parameters.nodetm_base-parameters.nodetm_mul*frac;
    double scalebm = bestMoveStabScaling[min(4, nbInARow)];
    sbig newSoft = originLowerBound*scalebm*scalenode;
    if(verbose)
        printf("info string newSoft %" PRId64 " hard %" PRId64 " frac %.2f scalenode %.2f scaletm %.2f\n", newSoft, hardBound, frac, scalenode, scalebm);
    return softBound = min(hardBound, newSoft);
}