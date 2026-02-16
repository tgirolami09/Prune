#include "TimeManagement.hpp"
#include "Move.hpp"
#include <cstdio>
//thank to heimdall for the node tm discovering
//I took some starting TM constants from his repo : https://github.com/nocturn9x/heimdall/blob/master/src/heimdall/util/limits.nim#L84-L98

const float bestMoveStabScaling[] = {2.50, 1.20, 0.90, 0.80, 0.75};
const float evalStabScaling[] = {1.60, 1.15, 0.90, 0.80, 0.75};
TM::TM(int _softBound, int _hardBound):softBound(_softBound), hardBound(_hardBound), enableUpdate(false), lastbestMove(nullMove.moveInfo), nbInARow_bm(0){}
TM::TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color):enableUpdate(true), lastbestMove(nullMove.moveInfo), nbInARow_bm(0){
    int time = (color == WHITE) ? wtime : btime;
    int inc = (color == WHITE) ? winc : binc;
    hardBound = time/8+inc*2/3-moveOverhead;
    originLowerBound = softBound = time/30+inc*2/3;
}

sbig TM::updateSoft(sbig bestMoveNodes, sbig totalNodes, int16_t bestmove, int eval, tunables& parameters, bool verbose){
    if(!enableUpdate)return softBound;
    if(lastbestMove == bestmove)nbInARow_bm++;
    else{
        lastbestMove = bestmove;
        nbInARow_bm=0;
    }
    if(abs(last_eval-eval) < 20)nbInARow_eval++;
    else nbInARow_eval=0;
    last_eval = eval;
    double frac = ((double)bestMoveNodes)/totalNodes;
    double scalenode = parameters.nodetm_base-parameters.nodetm_mul*frac;
    double scalebm = bestMoveStabScaling[min(4, nbInARow_bm)];
    double scaleeval = evalStabScaling[min(4, nbInARow_eval)];
    sbig newSoft = originLowerBound*scalebm*scalenode*scaleeval;
    if(verbose)
        printf("info string newSoft %" PRId64 " hard %" PRId64 " frac %.2f scalenode %.2f scaletm %.2f scaleeval %.2f\n", newSoft, hardBound, frac, scalenode, scalebm, scaleeval);
    return softBound = min(hardBound, newSoft);
}