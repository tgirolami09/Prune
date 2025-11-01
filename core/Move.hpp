#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
const int16_t clearTo = 0x3f;
const int16_t clearFrom = 0x3f<<6;
const int16_t clearPromot = -4096; // 0xf << 12
//Represents a move
class Move{
public :
    int8_t piece;
    //-1 by capturing by en passant
    int8_t capture = -2;
    //Stores end_pos / start_pos / promoteTo
    int16_t moveInfo = -4096;
    int from() const;
    int to() const;
    int8_t promotion() const;
    //Swaps from/to values
    void swapMove();
    void updateFrom(int from_square);
    void updateTo(int to_square);
    void updatePromotion(int promotionPiece);
    void from_uci(string move);
    string to_str();
    bool operator==(Move o);
    bool isTactical() const;
    bool isChanger() const;
    int getMovePart() const;
};
// const Move nullMove={0, 0, 0, -1, -2, -4096};
const Move nullMove = {0, -2, -4096};
#endif