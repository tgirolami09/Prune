#include "Functions.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include <cstdint>
#include <vector>
#include "viriformatUtil.hpp"
#include <cassert>

/* Important note:
    my squares are H1=0, A1=7 A8=63 H8=56,
    which are different from the expected A1=0, H1=7, A8=56, H8=63
    so to convert, I use:
        newSquare = square ^ 0x7
    also why I use reverse_col : expected to mirror verticaly the board, to get the goo occupied-piece bitboard (comes from Functions.cpp)
 */

template<typename T> 
void fastWrite(T data, FILE* file){
    fwrite(reinterpret_cast<const char*>(&data), sizeof(data), 1, file);
}

MoveInfo::MoveInfo(){
    move = nullMove;
    score = 0;
}
const uint16_t coltofield=7<<6;
void MoveInfo::dump(FILE* datafile){
    int to = move.to()^0x07, from=move.from()^0x07;
    uint16_t mv = to << 6 | from;
    int type = 0;
    if(move.capture == -1) // en passant (move.catpure == -1 is for en passant, move.capture == -2 for no capture (ik it's strange))
        type = 1;
    else if(move.piece == KING && abs(move.from()-move.to()) == 2){ //is a castling
        type = 2;
        // king takes rook notation
        if(from > to)
            mv &= ~coltofield;
        else
            mv |= coltofield;
    }
    if(move.promotion() != -1){ // for promotion
        mv |= (move.promotion()-1) << 12;
        type = 3;
    }
    mv |= type << 14;
    fastWrite(mv, datafile);
    fastWrite<int16_t>(score, datafile);
}
void GamePlayed::dump(FILE* datafile){
    big occupied = 0; // calculate the occupied bitboard
    for(int j=0; j<2; j++)
        for(int i=0; i<6; i++)
            occupied |= startPos.boardRepresentation[j][i];
    fastWrite(reverse_col(occupied), datafile);
    int8_t entry = 0x00;
    bool isSec = false;
    int nbEntry = 0;
    big castle = startPos.castlingMask();
    for(int i=0; i<64; i++){
        int index = i ^ 0x07;
        big mask = 1ULL << index;
        if(mask & occupied){//if there is a piece there
            int8_t piece = startPos.getfullPiece(index);
            int _c = color(piece);
            piece = type(piece);
            if(piece == ROOK && (mask&castle)) // rook that can castle
                piece = 6;
            uint8_t full = (_c << 3) | piece;
            if(isSec){ //if it's the second piece of the byte, we write it
                fastWrite<uint8_t>(entry|(full << 4), datafile);
            }else {
                entry = full;
            }
            isSec ^= 1;
            nbEntry += 1;
        }
    }
    for(int i=nbEntry; i<32; i++){
        if(isSec)
            fastWrite<uint8_t>(entry, datafile);
        else
            entry = 0;
        isSec ^= 1;
    }
    uint8_t info = startPos.lastDoublePawnPush == -1 ? 64 : startPos.lastDoublePawnPush^0x07; //en passant square
    info |= startPos.friendlyColor() << 7;
    fastWrite(info, datafile);
    fastWrite<uint8_t>(0, datafile);  // halfmove clock (for 50 move rule)
    fastWrite<uint16_t>(0, datafile); // full move
    fastWrite<uint16_t>(0, datafile); //score of the position
    fastWrite<uint8_t>(result, datafile); // result
    fastWrite<uint8_t>(0, datafile); //unused extra byte
    for(MoveInfo moves:game){//write all the stored moves
        moves.dump(datafile);
    }
    fastWrite<uint32_t>(0, datafile); //ending 4 bytes
}
void GamePlayed::clear(){
    game.clear();
}