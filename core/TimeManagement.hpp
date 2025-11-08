#ifndef TIME_MANAGEMENT_HPP
#define TIME_MANAGEMENT_HPP
#include "Const.hpp"

class TM{
public:
    sbig softBound, hardBound, originLowerBound;
    bool enableUpdate;
    TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color);
    TM(int softBound, int hardBound);
    sbig updateSoft(sbig bestMoveNodes, sbig totalNodes);
};

#endif