#include "corrhist.hpp"
#include "Const.hpp"
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
void corrhist<size, maxCorrHist>::update(big key, bool c, int diff, int depth){
    int& cur = table[c][key%size];
    int update = clamp(diff*(depth/8)/fracDepth, -maxCorrHist, maxCorrHist);
    cur += update - cur*abs(update)/maxCorrHist;
}

void corrhists::update(const GameState& state, int diff, int depth){
    int bonus = diff*corrhistGrain;
    int lastmoveid = state.getLastMove().move.moveInfo;
    int contmoveid = state.getContMove().move.moveInfo;
    pawns.update(state.pawnZobrist, state.friendlyColor(), bonus, depth);
    prevMove.update(lastmoveid, state.friendlyColor(), bonus, depth);
    cont.update(contmoveid^((uint32_t)lastmoveid*0xa28fU&((1U << 16)-1)), state.friendlyColor(), bonus, depth);
    minor.update(state.minorZobrist, state.friendlyColor(), bonus, depth);
}

int corrhists::probe(const GameState& state) const{
    int lastmoveid = state.getLastMove().move.moveInfo;
    int contmoveid = state.getContMove().move.moveInfo;
    int diff = (
        pawns.probe(state.pawnZobrist, state.friendlyColor()) +
        cont.probe(contmoveid^((uint32_t)lastmoveid*0xa28fU&((1U << 16)-1)), state.friendlyColor()) +
        prevMove.probe(lastmoveid, state.friendlyColor()) +
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
    minor.reset();
}