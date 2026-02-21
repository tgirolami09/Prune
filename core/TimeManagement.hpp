#ifndef TIME_MANAGEMENT_HPP
#define TIME_MANAGEMENT_HPP
#include "Const.hpp"
#include "tunables.hpp"

class TM{
public:
    sbig softBound, hardBound, originLowerBound;
    bool enableUpdate;
    int16_t lastbestMove;
    int nbInARow;
    TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color);
    TM(int softBound, int hardBound);
    sbig updateSoft(sbig bestMoveNodes, sbig totalNodes, int16_t bestmove, tunables& parameters, bool verbose);
};

#endif