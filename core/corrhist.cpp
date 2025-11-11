#include "corrhist.hpp"
#include <cstdlib>
#include <cstring>
#include <cassert>

int sum_diffs = 0;
int nb_diffs = 0;
int max_diff = 0;
int min_diff = 0;

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
void corrhist<size, maxCorrHist>::update(big key, bool c, int diff){
    diff = min(max(diff, -maxCorrHist), maxCorrHist);
    table[c][key%size] += diff - table[c][key%size]*abs(diff)/maxCorrHist;
}

void corrhists::update(const GameState& state, int diff, int depth){
    int bonus = diff*corrhistGrain;
    pawns.update(state.pawnZobrist, state.friendlyColor(), bonus);
}

int corrhists::probe(const GameState& state) const{
    int diff = pawns.probe(state.pawnZobrist, state.friendlyColor())/corrhistGrain;
    if(diff > max_diff)max_diff = diff;
    else if(diff < min_diff)min_diff = diff;
    sum_diffs += diff;
    nb_diffs++;
    return diff;
}

corrhists::corrhists(){
    reset();
}

void corrhists::reset(){
    pawns.reset();
}