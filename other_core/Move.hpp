#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
//Represents a move
class Move{
    public :
    int8_t start_pos;
    int8_t end_pos;
    int8_t piece;
    //Type of piece to promote to
    int8_t promoteTo = -1;
    //-1 by capturing by en passant
    int8_t capture = -2;

    void from_uci(string move){
        start_pos = from_str(move.substr(0, 2));
        end_pos = from_str(move.substr(2, 2));
        if(move.size() == 5){
            promoteTo = piece_to_id.at(move[4]);
        }
    }

    string to_str(){
        string res = to_uci(start_pos)+to_uci(end_pos);
        if(promoteTo == -1)
            res += id_to_piece[promoteTo];
        return res;
    }
};
#endif