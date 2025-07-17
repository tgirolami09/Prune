#ifndef COMPRESSOR
#define COMPRESSOR
#include "Const.hpp"
#include "GameState.hpp"

const int nb64=10, nb8=1;
class Compressed{
public:
    big values64[nb64];
    ubyte values8[nb8];
    Compressed(GameState state){
        int index = 0;
        ubyte king = 0;
        for(int c=0; c<2; c++)
            for(int i=0; i<nbPieces; i++){
                if(i == KING){
                    king |= __builtin_ctzll(state.boardRepresentation[c][i]) << (c*4);
                }else{
                    values64[index] = state.boardRepresentation[c][i];
                    index++;
                }
            }
        values8[0] = king;
    }
    bool operator<(const Compressed& o){
        for(int i=0; i<nb64; i++){
            if(values64[i] != o.values64[i])return values64[i] < o.values64[i];
        }
        for(int i=0; i<nb8; i++){
            if(values8[i] != o.values8[i])return values8[i] < o.values8[i];
        }
        return false;
    }
};

class infoScore{
public:
    int score;
    int beta;
    int alpha;
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
    int get_eval(GameState& state, int alpha, int beta, bool& isok){
        int index=state.zobristHash%modulo;
        if(table[index].hash == state.zobristHash){
            isok=true;
            if(table[index].score > beta)
                return table[index].score;
            if(table[index].score < table[index].beta)
                return table[index].score;
            isok=false;
        }
        return 0;
    }
    void push(GameState& state, infoScore info){
        info.hash = state.zobristHash;
        if(table[info.hash%modulo].hash != 0)
            rewrite++;
        else place++;
        table[info.hash%modulo] = info;
    }
    void clear(){
        table.clear();
    }
    void reinit(int count){
        table.resize(count);
    }
};

#endif