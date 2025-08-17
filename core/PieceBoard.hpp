#ifndef PIECEBOARD_HPP
#define PIECEBOARD_HPP

#include "Const.hpp"

class PieceBoard{
    uint8_t squares[64];

    const int blackAdd = 100;

    void clear(){
        for (int i = 0; i < 64; ++i){
            squares[i] = SPACE;
        }
    }

    public:

    PieceBoard(){
        clear();
    }

    void setSquare(uint8_t pieceId, uint8_t pos, bool c){
        squares[pos] = pieceId + c * blackAdd;
    }

    void clearSquare(uint8_t pos){
        squares[pos] = SPACE;
    }

    uint8_t getPiece(uint8_t pos, bool c){
        if (((squares[pos] >= (blackAdd)) && c) || ((!c) && (squares[pos] < (blackAdd))) ){
            return squares[pos] - (c * blackAdd);
        }
        else{
            return SPACE;
        }
    }
};

#endif