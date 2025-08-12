#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
const int16_t clearTo = 0x3f;
const int16_t clearFrom = 0x3f<<6;
const int16_t clearPromot = 0xf << 12;
//Represents a move
class Move{
public :
    int8_t piece;
    //-1 by capturing by en passant
    int8_t capture = -2;

    //Stores end_pos / start_pos / promoteTo
    int16_t moveInfo = -4096;

    int from() const{
        return (moveInfo >> 6) & 0x3f;
    }

    int to() const{
        return (moveInfo) & 0x3f;
    }

    int8_t promotion() const{
        //Interestingly there is no need for the '& 7' because we also need negative numbers
        return (moveInfo >> 12) ;//& 0x7;
    }

    //Swaps from/to values
    void swapMove(){
        int from_square = from();
        int to_square = to();
        moveInfo &= ~(clearTo | clearFrom);

        swap(from_square,to_square);
        moveInfo |= (int16_t)( from_square << 6 );
        moveInfo |= (int16_t)( to_square );
    }

    void updateFrom(int from_square){
        moveInfo |= (int16_t)( from_square << 6 );
    }

    void updateTo(int to_square){
        moveInfo |= (int16_t)( to_square );
    }

    void updatePromotion(int promotionPiece){
        int oldPromotion = promotion();
        moveInfo &= ~clearPromot;
        moveInfo |= (int16_t)( promotionPiece << 12 );    
    }

    void from_uci(string move){
        moveInfo |= (int16_t)(from_str(move.substr(2, 2)));
        moveInfo |= (int16_t)(from_str(move.substr(0, 2)) << 6);
        if(move.size() == 5){
            updatePromotion(piece_to_id.at(move[4]));
        }
    }

    string to_str(){
        string newRes = to_uci(from())+to_uci(to());
        if (promotion() != -1){
            newRes += id_to_piece[promotion()];
        }
        return newRes;
    }
    bool operator==(Move o){
        //if capture is not the same, I think we can also considere that there are the same
        return o.moveInfo == moveInfo && o.piece == piece;
    }
};
// const Move nullMove={0, 0, 0, -1, -2, -4096};
const Move nullMove = {0, -2, -4096};
#endif