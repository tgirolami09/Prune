#include "TimeManagement.hpp"
#include <cstdio>
//thank to heimdall for the scale discovering
//I took some starting TM constants from his repo : https://github.com/nocturn9x/heimdall/blob/master/src/heimdall/util/limits.nim#L84-L98

TM::TM(int _softBound, int _hardBound):softBound(_softBound), hardBound(_hardBound), enableUpdate(false){}
TM::TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color):enableUpdate(true){
    int time = (color == WHITE) ? wtime : btime;
    int inc = (color == WHITE) ? winc : binc;
    hardBound = time/10+inc*2/3-moveOverhead;
    originLowerBound = softBound = hardBound/3;
}

sbig TM::updateSoft(sbig bestMoveNodes, sbig totalNodes, tunables& parameters, bool verbose){
    if(!enableUpdate)return softBound;
    double frac = ((double)bestMoveNodes)/totalNodes;
    double scale = parameters.nodetm_base-parameters.nodetm_mul*frac;
    sbig newSoft = originLowerBound*scale;
    if(verbose)
        printf("info string newSoft %" PRId64 " hard %" PRId64 " frac %.2f scale %.2f\n", newSoft, hardBound, frac, scale);
    return softBound = min(hardBound, newSoft);
}