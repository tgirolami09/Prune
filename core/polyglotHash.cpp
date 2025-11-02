#include "polyglotHash.hpp"

uint64_t key_hash(uint64_t piece, uint64_t castle, uint64_t enpassant, uint64_t turn){
    uint64_t key = piece^castle^enpassant^turn;
    return key;
};

uint64_t piece_hash(const GameState& state){
    uint64_t piece = 0;
    const uint64_t* whitePieces = state.friendlyColor() == WHITE ? state.friendlyPieces() : state.enemyPieces();
    const uint64_t* blackPieces = state.friendlyColor() == BLACK ? state.friendlyPieces() : state.enemyPieces();
    for (int id = 0; id < 6; ++ id){
        ubyte positions[11];
        int nbWhitePieces = places(whitePieces[id],positions);
        for (int i = 0; i < nbWhitePieces; ++ i){
            int pos = positions[i];
            //+1 because white
            //Not sure that we have the right orientation with columns
            int offset_piece = (64 * (2 * id + 1)) + (8 * (row(pos))) + (7-col(pos));
            piece ^= Random64[offset_piece];
        }

        int nbBlackPieces = places(blackPieces[id],positions);
        for (int i = 0; i < nbBlackPieces; ++ i){
            int pos = positions[i];
            //Not sure that we have the right orientation with columns
            int offset_piece = (64 * (2 * id)) + (8 * (row(pos))) + (7-col(pos));
            piece ^= Random64[offset_piece];
        }
    }
    return piece;
}

uint64_t castle_hash(const GameState& state){
    uint64_t castle = 0;
    //White king side
    if (state.castlingRights[0][1]){
        castle ^= Random64[768];
    }
    //White queen side
    if (state.castlingRights[0][0]){
        castle ^= Random64[768 + 1];
    }
    //Black king side
    if (state.castlingRights[1][1]){
        castle ^= Random64[768 + 2];
    }
    //Black queen side
    if (state.castlingRights[1][0]){
        castle ^= Random64[768 + 3];
    }
    
    return castle;
}

uint64_t enPassant_hash(const GameState& state){
    if (state.lastDoublePawnPush != -1){
        //Should only bet set if there is a pawn that can capture
        uint64_t enPassant = 0;
        int baseOffset = 772;
        int enPassant_offset = col(state.lastDoublePawnPush);
        int moveFactor = state.friendlyColor() == WHITE ? 1 : -1;
        //Reverse because opposite colour pawn did double push
        // int actualPawnLine = row(state.lastDoublePawnPush) + (-8 * moveFactor);
        big possibleCapturePawns = 0;
        if (enPassant_offset != 0){
            possibleCapturePawns |= (1ull << (state.lastDoublePawnPush + (-8 * moveFactor) + (-1)));
        }
        if(enPassant_offset != 7){
            possibleCapturePawns |= (1ull << (state.lastDoublePawnPush + (-8 * moveFactor) + (1)));
        }
        if (possibleCapturePawns & state.friendlyPieces()[PAWN]){
            enPassant = Random64[baseOffset+7-enPassant_offset];
        }

        return enPassant;
    }
    return 0;
}

uint64_t turn_hash(const GameState& state){
    uint64_t turn = 0;
    int baseOffset = 780;
    if (state.friendlyColor() == WHITE){
        turn = Random64[baseOffset];
    }

    return turn;
}

uint64_t polyglotHash(const GameState& state){
    uint64_t piece = piece_hash(state);
    uint64_t castle = castle_hash(state);
    uint64_t enPassant = enPassant_hash(state);
    uint64_t turn = turn_hash(state);
    // printf("Board hash %llu\n",piece);
    // printf("castling hash %llu\n",castle);
    // printf("en passant hash %llu\n",enPassant);
    // printf("Turn hash %llu\n",turn);
    uint64_t key = key_hash(piece,castle,enPassant,turn);
    // printf("Full hash %llu\n",key);

    return key;
}