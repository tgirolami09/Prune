#ifndef TIME_MANAGEMENT_HPP
#define TIME_MANAGEMENT_HPP
#include "Const.hpp"

class TM{
public:
    int softBound, hardBound, originLowerBound;
    TM(int moveOverhead, int wtime, int btime, int binc, int winc, bool color, bool worthMoreTime);
    TM(int softBound, int hardBound);
    int updateSoft(big bestMoveNodes, big totalNodes);
};

#endif