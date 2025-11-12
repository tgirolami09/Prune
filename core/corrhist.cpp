#include "corrhist.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>

#ifdef DEBUG
int sum_diffs = 0;
int nb_diffs = 0;
int max_diff = 0;
int min_diff = 0;
#endif

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
    cont.update(state.getLastMove().moveInfo+(1<<15), state.friendlyColor(), bonus, weight);
}

int corrhists::probe(const GameState& state) const{
    int diff = (
        pawns.probe(state.pawnZobrist, state.friendlyColor()) +
        cont.probe(state.getLastMove().moveInfo+(1<<15), state.friendlyColor())
    )/corrhistGrain;
#ifdef DEBUG
    if(diff > max_diff)max_diff = diff;
    else if(diff < min_diff)min_diff = diff;
    sum_diffs += diff;
    nb_diffs++;
#endif
    return diff;
}

corrhists::corrhists(){
    reset();
}

void corrhists::reset(){
    pawns.reset();
    cont.reset();
}