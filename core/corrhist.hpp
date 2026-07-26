#ifndef CORR_HIST_HPP
#define CORR_HIST_HPP
#include "Const.hpp"
#include "GameState.hpp"
#ifdef DEBUG_MACRO
#include "stats_helpers.hpp"
extern StatVar<sbig, 64*4, -64*4> diffsStat;
#endif
template<int size, int maxCorrHist>
class corrhist{
public:
    corrhist();
    int table[2][size];
    void reset();
    void update(big, bool, int, int);
    int probe(big, bool) const;
};

class corrhists{
    corrhist<16384, 1024> pawns;
    corrhist<16384, 1024> prevMove;
    corrhist<16384, 1024> cont;
    corrhist<16384, 1024> minor;
public:
    corrhists();
    void update(const GameState&, int, int);
    int probe(const GameState& state) const;
    void reset();
};
#endif