#include "TimeManagement.hpp"
#include <cstdio>

TM::TM(int hardBound, int softBound):softBound(softBound), hardBound(hardBound), enableUpdate(false){}
TM::TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color, bool worthMoreTime):enableUpdate(true){
    int time = (color == WHITE) ? wtime : btime;
    int inc = (color == WHITE) ? winc : binc;
    hardBound = time/10+inc*2/3-moveOverhead;
    originLowerBound = softBound = hardBound/3;
}

int TM::updateSoft(big bestMoveNodes, big totalNodes){
    if(!enableUpdate)return softBound;
    double frac = ((double)bestMoveNodes)/totalNodes;
    double scale = 2.00-1.600*frac;
    int newSoft = originLowerBound*scale;
    printf("info string newSoft %d hard %d frac %.2f scale %.2f\n", newSoft, hardBound, frac, scale);
    return softBound = min(hardBound, newSoft);
}