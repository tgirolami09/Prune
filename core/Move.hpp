#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
//Represents a move
class Move{
    public :
    // int8_t start_pos;
    // int8_t end_pos;
    int8_t piece;
    //Type of piece to promote to
    // int8_t promoteTo = -1;
    //-1 by capturing by en passant
    int8_t capture = -2;

    //Sotres end_pos / start_pos / promoteTo
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
        //Should clear info for from
        moveInfo ^= (int16_t)( from_square << 6 );

        //Should clear info for to
        moveInfo ^= (int16_t)( to_square );

        swap(from_square,to_square);
        moveInfo |= (int16_t)( from_square << 6 );
        moveInfo |= (int16_t)( to_square );
    }

    void updateFrom(int from_square){
        // start_pos = from;
        moveInfo |= (int16_t)( from_square << 6 );
    }

    void updateTo(int to_square){
        // end_pos = to;
        moveInfo |= (int16_t)( to_square );
    }

    void updatePromotion(int promotionPiece){
        // promoteTo = promotionPiece;
        int oldPromotion = promotion();
        //Should remove info
        moveInfo ^= (int16_t)( oldPromotion << 12 );

        moveInfo |= (int16_t)( promotionPiece << 12 );    
    }

    void from_uci(string move){
        // start_pos = from_str(move.substr(0, 2));
        // end_pos = from_str(move.substr(2, 2));
        moveInfo |= (int16_t)(from_str(move.substr(2, 2)));
        moveInfo |= (int16_t)(from_str(move.substr(0, 2)) << 6);
        if(move.size() == 5){
            // promoteTo = piece_to_id.at(move[4]);

            updatePromotion(piece_to_id.at(move[4]));
        }
    }

    string to_str(){
        // string res = to_uci(start_pos)+to_uci(end_pos);
        string newRes = to_uci(from())+to_uci(to());
        // if(promoteTo != -1){
            // res += id_to_piece[promoteTo];
        // }
        if (promotion() != -1){
            newRes += id_to_piece[promotion()];
        }
        // if (res != newRes){
            // printf("Normal : '%s' new : '%s' , moveInfo : %d\n",res.c_str(),newRes.c_str(),moveInfo);
        // }
        // return res;
        return newRes;
    }
    bool operator==(Move o){
        // return o.start_pos == start_pos && o.end_pos == end_pos && o.piece == piece && o.promoteTo == promoteTo;//if capture is not the same, I think we can also considere that there are the same
        return o.moveInfo == moveInfo && o.piece == piece;
    }
};
// const Move nullMove={0, 0, 0, -1, -2, -4096};
const Move nullMove = {0, -2, -4096};
#endif