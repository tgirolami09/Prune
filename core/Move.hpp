#ifndef MOVE_HPP
#define MOVE_HPP
#include <string>
#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
const uint16_t clearTo = 0x3f;
const uint16_t clearFrom = 0x3f<<6;
const uint16_t clearPromot = 0x3 << 14; // 0xf << 12
//Represents a move
class Move{
public :
    enum{fnormal=0, fcastle=1, fpromo=2, fep=3};
    //Stores promotion|flag|from|to
    // if flag == 1 => Castling
    // if flag == 2 => Promotion
    // if flag == 3 => En Passanr
    // promotion = 2 bits
    // flag = 2 bit
    // from = 6 bits
    // to = 6 bits
    uint16_t moveInfo = 0;
    int getFlag() const;
    int from() const;
    int to() const;
    int toMover() const;
    int8_t promotion() const;
    void setFlag(int flag);
    //Swaps from/to values
    void swapMove();
    void updateFrom(int from_square);
    void updateTo(int to_square);
    void updatePromotion(int promotionPiece);
    void from_uci(string move);
    string to_str() const;
    bool operator==(Move o) const;
    int getMovePart() const;
};

class ExpendedMove{
public:
    Move move;
    int piece;
    int capture;
};

const Move nullMove = {0};
const ExpendedMove EnullMove = {{0}, 0, -2};
#endif