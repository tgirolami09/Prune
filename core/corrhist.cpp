#include "corrhist.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>

#ifdef DEBUG_MACRO
StatVar<sbig, 64*4, -64*4> diffsStat;
#endif

//used 256 / max(depth+1, 16) from https://github.com/mcthouacbb/Sirius

template<int size, int maxCorrHist>
void corrhist<size, maxCorrHist>::reset(){
    memset(table, 0, sizeof(table));
}

template<int size, int maxCorrHist>
int corrhist<size, maxCorrHist>::probe(big key, bool c) const{
    return table[c][key%size];
}

template<int size, int maxCorrHist>
corrhist<size, maxCorrHist>::corrhist(){
    reset();
}

template<int size, int maxCorrHist>
void corrhist<size, maxCorrHist>::update(big key, bool c, int diff, int weight){
    int& cur = table[c][key%size];
    cur = ((256-weight)*cur+diff*weight)/256;
    cur = clamp(cur, -maxCorrHist, maxCorrHist);
}

void corrhists::update(const GameState& state, int diff, int depth){
    int bonus = diff*corrhistGrain;
    int weight = max(depth+1, 16);
    pawns.update(state.pawnZobrist, state.friendlyColor(), bonus, weight);
    prevMove.update(state.getLastMove().moveInfo+(1<<15), state.friendlyColor(), bonus, weight);
    cont.update(state.getContMove().moveInfo+(1<<15), state.friendlyColor(), bonus, weight);
    minor.update(state.minorZobrist, state.friendlyColor(), bonus, weight);
}

int corrhists::probe(const GameState& state) const{
    int diff = (
        pawns.probe(state.pawnZobrist, state.friendlyColor()) +
        cont.probe(state.getContMove().moveInfo+(1<<15), state.friendlyColor()) +
        prevMove.probe(state.getLastMove().moveInfo+(1<<15), state.friendlyColor()) +
        minor.probe(state.minorZobrist, state.friendlyColor())
    )/corrhistGrain;
#ifdef DEBUG_MACRO
    diffsStat.update(diff);
#endif
    return diff;
}

corrhists::corrhists(){
    reset();
}

void corrhists::reset(){
    pawns.reset();
    cont.reset();
    prevMove.reset();
}