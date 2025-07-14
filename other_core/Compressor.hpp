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
                    king |= (ffsll(state.boardRepresentation[c][i])-1) << (c*4);
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

#endif