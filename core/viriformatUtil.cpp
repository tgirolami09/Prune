#include "viriformatUtil.hpp"
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "Move.hpp"

/* Important note:
    my squares are H1=0, A1=7 A8=63 H8=56 :
    63 62 61 60 59 58 57 56
    55 54 53 52 51 50 49 48
    47 46 45 44 43 42 41 40
    39 38 37 36 35 34 33 32
    31 30 29 28 27 26 25 24
    23 22 21 20 19 18 17 16
    15 14 13 12 11 10  9  8
     7  6  5  4  3  2  1  0
    which are different from the expected A1=0, H1=7, A8=56, H8=63
    so to convert, I use:
        newSquare = square ^ 0x7
    also why I use reverse_col : expected to mirror verticaly the board, to get
   the goo occupied-piece bitboard (comes from Functions.cpp)
 */

template <typename T>
void fastWrite(T data, FILE* file) {
    fwrite(reinterpret_cast<const char*>(&data), sizeof(data), 1, file);
}

template <typename T>
uint32_t fastRead(T& data, FILE* file) {
    _unused int x = fread(reinterpret_cast<char*>(&data), sizeof(data), 1, file);
    assert(x);
    return data;
}

MoveInfo::MoveInfo() {
    move = nullMove;
    score = 0;
}
void MoveInfo::dump(FILE* datafile) {
    if (move) {
        static constexpr int transfo[4] = {0, 2, 3, 1};
        int to = move.to() ^ 0x07, from = move.from() ^ 0x07;
        uint16_t mv = to << 6 | from;
        mv |= (move.promotion() - (move.getFlag() == Move::fpromo)) << 12;
        int type = transfo[move.getFlag()];
        mv |= type << 14;
        fastWrite(mv, datafile);
    } else {
        fastWrite<uint16_t>(0, datafile);
    }
    fastWrite<int16_t>(score, datafile);
}
void GamePlayed::dump(FILE* datafile) {
    big occupied = startPos.board.colors[WHITE] |
                   startPos.board.colors[BLACK];  // calculate the occupied bitboard
    fastWrite(reverse_col(occupied), datafile);
    uint8_t entry = 0x00;
    bool isSec = false;
    int nbEntry = 0;
    big castle = startPos.castlingMask;
    for (int i = 0; i < 64; i++) {
        int index = i ^ 0x07;
        big mask = 1ULL << index;
        if (mask & occupied) {  // if there is a piece there
            int8_t piece = startPos.getfullPiece(index);
            int _c = color(piece);
            piece = type(piece);
            if (piece == ROOK && (mask & castle))  // rook that can castle
                piece = 6;
            uint8_t full = (_c << 3) | piece;
            if (isSec) {  // if it's the second piece of the byte, we write it
                fastWrite<uint8_t>(entry | (full << 4), datafile);
            } else {
                entry = full;
            }
            isSec ^= 1;
            nbEntry += 1;
        }
    }
    for (int i = nbEntry; i < 32; i++) {
        if (isSec)
            fastWrite<uint8_t>(entry, datafile);
        else
            entry = 0;
        isSec ^= 1;
    }
    uint8_t info = startPos.lastDoublePawnPush == 64
                       ? 64
                       : startPos.lastDoublePawnPush ^ 0x07;  // en passant square
    info |= startPos.friendlyColor() << 7;
    fastWrite(info, datafile);
    fastWrite<uint8_t>(0, datafile);       // halfmove clock (for 50 move rule)
    fastWrite<uint16_t>(0, datafile);      // full move
    fastWrite<uint16_t>(0, datafile);      // score of the position
    fastWrite<uint8_t>(result, datafile);  // result
    fastWrite<uint8_t>(0, datafile);       // unused extra byte
    for (MoveInfo moves : game) {          // write all the stored moves
        moves.dump(datafile);
    }
    fastWrite<uint32_t>(0, datafile);  // ending 4 bytes
}
void GamePlayed::clear() {
    game.clear();
}

big chunkedToMask(__m256i chunk1, __m256i chunk2, ubyte piece) {
    big occupied =
        (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk1, _mm256_set1_epi8(piece)));
    occupied |= (big)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk2, _mm256_set1_epi8(piece))) << 32;
    return occupied;
}

big invpext(big x, big mask) {
    big res = 0;
    while (mask) {
        res |= (x & 1) << __builtin_ctzll(mask);
        mask &= mask - 1;
        x >>= 1;
    }
    return res;
}

void dumpPosition(__m256i position, const ubyte flags, FILE* datafile, int16_t score, ubyte bound,
                  Move move, int depth, int count50) {
    __m256i chunk1 = _mm256_and_si256(position >> 4, _mm256_set1_epi8(0b1111));
    __m256i chunk2 = _mm256_and_si256(position, _mm256_set1_epi8(0b1111));
    alignas(32) ubyte mailbox[64];
    _mm256_store_si256(reinterpret_cast<__m256i*>(mailbox), chunk1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(mailbox) + 1, chunk2);
    big occupied = ~chunkedToMask(chunk1, chunk2, SPACE * 2);

    fastWrite(reverse_col(occupied), datafile);  // 8B written
    bool stm = flags & 1;

    __uint128_t compressedMB = 0;
    int kingpos1 = 0;
    int kingpos2 = 0;
    int idPiece = 0;
    for (int i = 0; i < 64; i++) {
        int index = i ^ 0x07;
        big mask = 1ULL << index;
        if (mask & occupied) {  // if there is a piece there
            int8_t piece = mailbox[index];
            if (type(piece) != KING) {
                if (piece == SPACE * 2 + 1)
                    piece = 10;
                compressedMB = compressedMB * 11 + piece;
            } else {
                if (!color(piece))
                    kingpos1 = idPiece;
                else
                    kingpos2 = idPiece;
            }
            idPiece++;
        }
    }
    compressedMB = (compressedMB * 32 + kingpos1) * 31 + (kingpos2 - (kingpos1 < kingpos2));
    compressedMB = compressedMB * 2 + stm;
    compressedMB = compressedMB * 100 + count50;
    compressedMB = compressedMB * 3 + bound;
    compressedMB = compressedMB * 32 + min(depth / fracDepth, 31);
    fastWrite(compressedMB, datafile);
    // 8B + 16B = 24B written
    MoveInfo mi;
    mi.move = move;
    mi.score = score;
    mi.dump(datafile);
    // 24B + 4B = 28B written
}

GamePlayed readGame(FILE* file) {
    GamePlayed game;
    big occupied = 0;
    fastRead(occupied, file);
    occupied = reverse_col(occupied);
    uint8_t entry = 0;
    big castle = 0;
    bool isSec = false;
    int nbEntry = 0;
    for (int i = 0; i < 64; i++) {
        int index = i ^ 0x07;
        big mask = 1ULL << index;
        if (mask & occupied) {  // if there sould be a piece there
            if (!isSec)
                fastRead(entry, file);
            int8_t full = entry & 0b1111;
            entry >>= 4;
            int piece = full & 0b111;
            int _c = full >> 3;
            if (piece == 6) {
                castle |= mask;
                piece = ROOK;
            }
            game.startPos.board.addPiece(index, piece, _c);
            game.startPos.updateZobrists(piece, _c, i);
            isSec ^= 1;
            nbEntry++;
        }
    }
    for (int i = nbEntry; i < 32; i++) {
        if (!isSec)
            fastRead(entry, file);
        isSec ^= 1;
    }
    ubyte info;
    uint64_t infoGame;
    fastRead(infoGame, file);
    info = infoGame;
    infoGame >>= 8;
    game.startPos.turnNumber = (info >> 7) == WHITE ? 1 : 0;
    info &= 0b1111111;
    game.startPos.lastDoublePawnPush = info == 64 ? 64 : info ^ 0x07;
    infoGame >>= 8;   // halfmove = infoGame;
    infoGame >>= 16;  // fullmove = infoGame;
    infoGame >>= 16;  // score = infoGame;
    game.result = infoGame;
    infoGame >>= 8;
    game.startPos.castlingFromMask(castle);
    uint32_t moveInfo;
    while (fastRead(moveInfo, file) != 0) {
        // printf("%8x:", moveInfo);
        MoveInfo move;
        uint16_t mv = moveInfo;
        move.score = (int16_t)(moveInfo >> 16);
        int from = mv & 0x3f;
        int to = (mv >> 6) & 0x3f;
        to ^= 0x07;
        from ^= 0x07;
        int type = mv >> 14;
        int promo = (mv >> 12) & 0b11;
        if (type == 1)
            move.move.setFlag(Move::fep);
        else if (type == 2) {
            move.move.setFlag(Move::fcastle);
        } else if (type == 3)
            move.move.updatePromotion(promo + 1);
        // printf("%d %d %d %d\n", from, to, type, move.score);
        move.move.updateFrom(from);
        move.move.updateTo(to);
        game.game.push_back(move);
    }
    // printf("\n");
    return game;
}