#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
//Represents a move
class Move{
    public :
    int start_pos;
    int end_pos;
    int piece;
    //Type of piece to promote to
    int promoteTo = -1;
    //-1 by capturing by en passant
    int capture = -2;

    void from_uci(string move){
        start_pos = from_str(move.substr(0, 2));
        end_pos = from_str(move.substr(2, 2));
        if(move.size() == 5){
            promoteTo = piece_to_id[move[4]];
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