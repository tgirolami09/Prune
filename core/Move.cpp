#include "Move.hpp"
using namespace std;

int Move::from() const{
    return (moveInfo >> 6) & 0x3f;
}

int Move::to() const{
    return (moveInfo) & 0x3f;
}

int8_t Move::promotion() const{
    //Interestingly there is no need for the '& 7' because we also need negative numbers
    return (moveInfo >> 12) ;//& 0x7;
}

//Swaps from/to values
void Move::swapMove(){
    int from_square = from();
    int to_square = to();
    moveInfo &= ~(clearTo | clearFrom);

    swap(from_square,to_square);
    moveInfo |= (int16_t)( from_square << 6 );
    moveInfo |= (int16_t)( to_square );
}

void Move::updateFrom(int from_square){
    moveInfo |= (int16_t)( from_square << 6 );
}

void Move::updateTo(int to_square){
    moveInfo |= (int16_t)( to_square );
}

void Move::updatePromotion(int promotionPiece){
    moveInfo &= ~clearPromot;
    moveInfo |= (int16_t)( promotionPiece << 12 );    
}

void Move::from_uci(string move){
    moveInfo |= (int16_t)(from_str(move.substr(2, 2)));
    moveInfo |= (int16_t)(from_str(move.substr(0, 2)) << 6);
    if(move.size() == 5){
        updatePromotion(piece_to_id.at(move[4]));
    }
}

string Move::to_str() const{
    string newRes = to_uci(from())+to_uci(to());
    if (promotion() != -1){
        newRes += id_to_piece[promotion()];
    }
    return newRes;
}
bool Move::operator==(Move o) const{
    //if capture is not the same, I think we can also considere that there are the same
    return o.moveInfo == moveInfo && o.piece == piece;
}

bool Move::isTactical() const{
    return moveInfo > 0 || capture != -2; // moveInfo > 0 is an equivalent of promotion() != -1
}

bool Move::isChanger() const{
    return piece == PAWN || capture != -2;
}
int Move::getMovePart() const{
    return moveInfo&~(clearTo|clearFrom);
}