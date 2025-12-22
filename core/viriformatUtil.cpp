#include "Functions.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include <vector>
#include "viriformatUtil.hpp"
#include <cassert>

template<typename T> 
void fastWrite(T data, FILE* file){
    fwrite(reinterpret_cast<const char*>(&data), sizeof(data), 1, file);
}

MoveInfo::MoveInfo(){
    move = nullMove;
    score = 0;
}
void MoveInfo::dump(FILE* datafile){
    uint16_t mv = move.to() << 6 | move.from();
    int type = 0;
    if(move.capture == -1) // en passant
        type = 1;
    else if(move.piece == KING && abs(move.from()-move.to()) == 2){
        type = 2;
        // king takes rook notation
        if(move.to() > move.from())
            mv |= 7 << 6;
        else
            mv &= ~((uint16_t)7 << 6);
        //printf("%s => %d %d\n", move.to_str().c_str(), mv >> 6, mv&0x3f);
    }
    mv ^= 0x7 << 6 | 0x7; // mirror verticaly the move
    if(move.promotion() != -1){
        mv |= (move.promotion()-1) << 12;
        type = 3;
    }
    mv |= type << 14;
    fastWrite(mv, datafile);
    fastWrite<int16_t>(score, datafile);
}
void GamePlayed::dump(FILE* datafile){
    big occupied = 0;
    for(int i=0; i<6; i++)
        for(int j=0; j<2; j++)
            occupied |= startPos.boardRepresentation[j][i];
    fastWrite(reverse_col(occupied), datafile);
    int8_t entry = 0x00;
    bool isSec = false;
    int nbEntry = 0;
    big castle = startPos.castlingMask();
    for(int i=0; i<64; i++){
        int index = i ^ 0x07;
        big mask = 1ULL << index;
        if(mask & occupied){
            int8_t piece = startPos.getfullPiece(index);
            int _c = color(piece);
            piece = type(piece);
            if(piece == ROOK && (mask&castle))
                piece = 6;
            entry = (entry << 4) | (_c << 3) | piece;
            if(isSec)
                fastWrite(entry, datafile);
            isSec ^= 1;
            nbEntry += 1;
        }
    }
    for(int i=nbEntry; i<32; i++){
        entry <<= 4;
        if(isSec)
            fastWrite(entry, datafile);
        isSec ^= 1;
    }
    uint8_t info = startPos.lastDoublePawnPush == -1 ? 64 : startPos.lastDoublePawnPush^0x07;
    info |= startPos.friendlyColor() << 7;
    fastWrite(info, datafile);
    fastWrite<uint8_t>(0, datafile);  // halfmove clock (for 50 move rule)
    fastWrite<uint16_t>(0, datafile); // full move
    fastWrite<uint16_t>(0, datafile); //score of the position
    fastWrite<uint8_t>(result, datafile); // result
    fastWrite<uint8_t>(0, datafile); //unused extra byte
    for(MoveInfo moves:game){
        moves.dump(datafile);
    }
    fastWrite<uint32_t>(0, datafile);
}
void GamePlayed::clear(){
    game.clear();
}