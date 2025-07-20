#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "Const.hpp"
#include "GameState.hpp"

class infoScore{
public:
    int score;
    int beta;
    int alpha;
    Move bestMove;
    int depth;
    big hash;
};

class transpositionTable{
public:
    vector<infoScore> table;
    int modulo;
    int rewrite=0;
    int place=0;
    transpositionTable(int count){
        table = vector<infoScore>(count);
        modulo=count;
    }
    int get_eval(const GameState& state, int alpha, int beta, bool& isok, int depth, Move& best){
        int index=state.zobristHash%modulo;
        if(table[index].hash == state.zobristHash){
            isok=true;
            if(depth <= table[index].depth){//if we have evaluated it with more depth remaining, we can just return this evaluation since it's a better evaluation
                if(table[index].score > beta)
                    return table[index].score;
                if(table[index].score < table[index].beta)
                    return table[index].score;
            }
            best = table[index].bestMove;
            isok=false;
        }
        return 0;
    }
    void push(GameState& state, infoScore info){
        info.hash = state.zobristHash;
        int index = info.hash%modulo;
        if(table[index].hash != 0){
            if(table[index].hash == info.hash && table[index].depth > info.depth)
                return;//already evaluated with a better depth
            rewrite++;
        }
        else place++;
        table[index] = info;
    }
    void clear(){
        table.clear();
    }
    void reinit(int count){
        table.resize(count);
    }
};

#endif