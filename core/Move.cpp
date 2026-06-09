#include "Move.hpp"
using namespace std;

int Move::from() const{
    return (moveInfo >> 6) & 0x3f;
}

int Move::to() const{
    return (moveInfo) & 0x3f;
}

int Move::toMover() const{
    return getFlag() == fcastle ? (to()&56)|kingposCastle[from() < to()] : to();
}

int8_t Move::promotion() const{
    return (moveInfo >> 14)+(getFlag() == fpromo);
}

int Move::getFlag() const{
    return (moveInfo >> 12) & 0b11;
}

void Move::setFlag(int flag){
    moveInfo &= ~(0b11U << 12);
    moveInfo |= flag << 12;
}
//Swaps from/to values
void Move::swapMove(){
    int from_square = from();
    int to_square = to();
    moveInfo &= ~(clearTo | clearFrom);

    swap(from_square,to_square);
    moveInfo |= (uint16_t)( from_square << 6 );
    moveInfo |= (uint16_t)( to_square );
}

void Move::updateFrom(int from_square){
    moveInfo |= (uint16_t)( from_square << 6 );
}

void Move::updateTo(int to_square){
    moveInfo |= (uint16_t)( to_square );
}

void Move::updatePromotion(int promotionPiece){
    moveInfo &= ~clearPromot;
    setFlag(fpromo);
    moveInfo |= (uint16_t)((promotionPiece-1) << 14);
}

void Move::from_uci(string move){
    moveInfo = 0;
    moveInfo |= (uint16_t)(from_str(move.substr(2, 2)));
    moveInfo |= (uint16_t)(from_str(move.substr(0, 2)) << 6);
    if(move.size() == 5){
        updatePromotion(piece_to_id.at(move[4]));
    }
}

string Move::to_str() const{
    string newRes = to_uci(from())+to_uci(to());
    if (getFlag() == fpromo){
        newRes += id_to_piece[promotion()+1];
    }
    return newRes;
}
bool Move::operator==(Move o) const{
    //if capture is not the same, I think we can also considere that there are the same
    return o.moveInfo == moveInfo;
}

int Move::getMovePart() const{
    return moveInfo&(clearTo|clearFrom);
}