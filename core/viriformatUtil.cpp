#include "viriformatUtil.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Functions.hpp"
#include "GameState.hpp"
#include "Move.hpp"

template <typename T>
void fastWrite(T data, FILE* file) {
    fwrite(reinterpret_cast<const char*>(&data), sizeof(data), 1, file);
}

template <typename T>
uint32_t fastRead(T& data, FILE* file) {
    assert(fread(reinterpret_cast<char*>(&data), sizeof(data), 1, file));
    return data;
}

MoveInfo::MoveInfo() {
    move = nullMove;
    score = 0;
}
MoveInfo::MoveInfo(Move mv, int sc) {
    move = mv;
    score = sc;
}
void MoveInfo::dump(FILE* datafile) const {
    static constexpr int transfo[4] = {0, 2, 3, 1};
    int to = move.to(), from = move.from();
    uint16_t mv = to << 6 | from;
    mv |= (move.promotion() - (move.getFlag() == Move::fpromo)) << 12;
    int type = transfo[move.getFlag()];
    mv |= type << 14;
    fastWrite(mv, datafile);
    fastWrite<int16_t>(score, datafile);
}

void dumpmove(Move move, vector<uint8_t>& buffer) {
    static constexpr int transfo[4] = {0, 2, 3, 1};
    int to = move.to(), from = move.from();
    uint16_t mv = to << 6 | from;
    mv |= (move.promotion() - (move.getFlag() == Move::fpromo)) << 12;
    int type = transfo[move.getFlag()];
    mv |= type << 14;
    size_t base = buffer.size();
    buffer.resize(base + 2, 0);
    memcpy(&buffer[base], &mv, 2);
}

void MoveInfo::dump(vector<uint8_t>& datafile) const {
    static constexpr int transfo[4] = {0, 2, 3, 1};
    int to = move.to(), from = move.from();
    uint16_t mv = to << 6 | from;
    mv |= (move.promotion() - (move.getFlag() == Move::fpromo)) << 12;
    int type = transfo[move.getFlag()];
    mv |= type << 14;
    size_t base = datafile.size();
    datafile.resize(base + 4, 0);
    int16_t scorei16 = score;
    memcpy(&datafile[base + 0], &mv, 2);
    memcpy(&datafile[base + 2], &scorei16, 2);
}

void dumpposition(vector<uint8_t>& buffer, const GameState& startPos) {
    u64 occupied = startPos.board.colors[White] |
                   startPos.board.colors[Black];  // calculate the occupied bitboard
    size_t base = buffer.size();
    buffer.resize(base + 32);
    memcpy(&buffer[base], &occupied, 8);
    uint8_t entry = 0x00;
    bool isSec = false;
    int nbEntry = 0;
    u64 castle = startPos.castlingMask;
    for (int i = 0; i < 64; i++) {
        int index = i;
        u64 mask = 1ULL << index;
        if (mask & occupied) {  // if there is a piece there
            int8_t piece = startPos.getfullPiece(index);
            int _c = color(piece);
            piece = type(piece);
            if (piece == Rook && (mask & castle))  // rook that can castle
                piece = 6;
            uint8_t full = (_c << 3) | piece;
            if (isSec) {  // if it's the second piece of the byte, we write it
                uint8_t w = entry | (full << 4);
                buffer[base + 8 + nbEntry / 2] = w;
            } else {
                entry = full;
            }
            isSec ^= 1;
            nbEntry += 1;
        }
    }
    uint8_t info = startPos.lastDoublePawnPush;  // en passant square
    info |= startPos.friendlyColor() << 7;
    buffer[base + 24] = info;
    buffer[base + 25] = startPos.rule50_count();  // halfmove clock (for 50 move rule)
    // full move
    // score of the position
    // unused extra byte
}

void GamePlayed::dump(FILE* datafile) {
    u64 occupied = startPos.board.colors[White] |
                   startPos.board.colors[Black];  // calculate the occupied bitboard
    fastWrite(occupied, datafile);
    uint8_t entry = 0x00;
    bool isSec = false;
    int nbEntry = 0;
    u64 castle = startPos.castlingMask;
    for (int i = 0; i < 64; i++) {
        int index = i;
        u64 mask = 1ULL << index;
        if (mask & occupied) {  // if there is a piece there
            int8_t piece = startPos.getfullPiece(index);
            int _c = color(piece);
            piece = type(piece);
            if (piece == Rook && (mask & castle))  // rook that can castle
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
    uint8_t info = startPos.lastDoublePawnPush;  // en passant square
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

GamePlayed readGame(FILE* file) {
    GamePlayed game;
    u64 occupied = 0;
    fastRead(occupied, file);
    uint8_t entry = 0;
    u64 castle = 0;
    bool isSec = false;
    int nbEntry = 0;
    for (int i = 0; i < 64; i++) {
        int index = i;
        u64 mask = 1ULL << index;
        if (mask & occupied) {  // if there sould be a piece there
            if (!isSec)
                fastRead(entry, file);
            int8_t full = entry & 0b1111;
            entry >>= 4;
            int piece = full & 0b111;
            int _c = full >> 3;
            if (piece == 6) {
                castle |= mask;
                piece = Rook;
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
    u8 info;
    u64 infoGame;
    fastRead(infoGame, file);
    info = infoGame;
    infoGame >>= 8;
    game.startPos.turnNumber = (info >> 7) == White ? 1 : 0;
    info &= 0b1111111;
    game.startPos.lastDoublePawnPush = info;
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