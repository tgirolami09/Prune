#ifndef CORR_HIST_HPP
#define CORR_HIST_HPP
#include "Const.hpp"
#include "GameState.hpp"
#ifdef DEBUG_MACRO
extern int max_diff;
extern int min_diff;
extern int nb_diffs;
extern int sum_diffs;
#endif
const int corrhistGrain=64;
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
    corrhist<16384, 64*corrhistGrain> pawns;
    corrhist<16384, 64*corrhistGrain> prevMove;
    corrhist<16384, 64*corrhistGrain> cont;
    corrhist<16384, 64*corrhistGrain> minor;
    corrhist<16384, 64*corrhistGrain> major;
public:
    corrhists();
    void update(const GameState&, int, int);
    int probe(const GameState& state) const;
    void reset();
};
#endif