#ifndef MOVE_HPP
#define MOVE_HPP
//Represents a move
class Move{
    public :
    int start_pos;
    int end_pos;

    //Type of piece to promote to
    int promoteTo = -1;
    //-1 by capturing by en passant
    int capture = -2;
};
#endif