#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
//Represents a move
class Move{
    public :
    //Stores end_pos / start_pos / promoteTo (in this order)
    int16_t moveInfo;
    // int8_t start_pos;
    // int8_t end_pos;
    int8_t piece;
    //Type of piece to promote to
    // int8_t promoteTo = -1;
    //-1 by capturing by en passant
    int8_t capture = -2;

    int start_pos(){
        int from_square = (moveInfo >> 6) & 0x3f;
        // from_square = row(from_square) * 8 + (7 - col(from_square));
        return from_square;
    }

    int end_pos(){
        int to_square = moveInfo & 0x3f;
        // to_square = row(to_square) * 8 + (7 - col(to_square));
        return to_square;
    }

    int promoteTo(){
        return (moveInfo >> 12) & 0x7;
    }

    void from_uci(string move){
        // start_pos = from_str(move.substr(0, 2));
        // end_pos = from_str(move.substr(2, 2));
        moveInfo |= (int16_t)(from_str(move.substr(0, 2)) << 6);
        moveInfo |= (int16_t)(from_str(move.substr(2, 2)));
        if(move.size() == 5){
            // promoteTo = piece_to_id.at(move[4]);
            moveInfo |= (int16_t)(piece_to_id.at(move[4]) << 12);
        }
        printf("Read move '%s' from uci\n",move.c_str());
        printf("Translated move : '%s'\n",to_str());
    }

    string to_str(){
        string res = to_uci(start_pos())+to_uci(end_pos());
        if(promoteTo() != -1){
            res += id_to_piece[promoteTo()];
        }
        return res;
    }
    bool operator==(Move o){
        return o.start_pos() == start_pos() && o.end_pos() == end_pos() && o.piece == piece && o.promoteTo() == promoteTo();//if capture is not the same, I think we can also considere that there are the same
    }
};
const Move nullMove={0, 0, 0, -1, -2};
#endif