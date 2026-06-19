#ifndef GAMESATE_HPP
#define GAMESATE_HPP
#include "Move.hpp"
#include "Const.hpp"
#include <cstring>
#include <cassert>
using namespace std;
const int maxPly = 8848*2+2;
const int zobrCastle=64*2*6;
const int zobrPassant=zobrCastle+65;
const int zobrTurn=zobrPassant+8;
const int nbZobrist=zobrTurn+1;
const int sizeThreeFold=8192;
extern big zobrist[nbZobrist];
struct PositionState{
    big pieces[6];
    big colors[2];
    int8_t mailbox[64];
    forceinline void remPiece(int position, int piecetype, bool color){
        pieces[piecetype] ^= 1ULL << position;
        colors[color] ^= 1ULL << position;
        mailbox[position] = SPACE*2;
    }
    forceinline void addPiece(int position, int piecetype, bool color){
        pieces[piecetype] ^= 1ULL << position;
        colors[color] ^= 1ULL << position;
        mailbox[position] = (piecetype << 1) | color;
    }

    forceinline void remPiece(int position, int fullpiece){
        pieces[type(fullpiece)] ^= 1ULL << position;
        colors[color(fullpiece)] ^= 1ULL << position;
        mailbox[position] = SPACE*2;
    }
    forceinline void addPiece(int position, int fullpiece){
        pieces[type(fullpiece)] ^= 1ULL << position;
        colors[color(fullpiece)] ^= 1ULL << position;
        mailbox[position] = fullpiece;
    }
    forceinline void reset(){
        memset(pieces, 0, sizeof(pieces));
        memset(colors, 0, sizeof(colors));
        memset(mailbox, SPACE*2, sizeof(mailbox));
    }
    forceinline big getMask(int piece, bool color) const{
        return pieces[piece]&colors[color];
    }
    forceinline big getMask(int piece) const{
        return pieces[type(piece)]&colors[color(piece)];
    }
    forceinline big occupancy() const{
        return colors[WHITE] | colors[BLACK];
    }
    forceinline bool isChanger(const Move& move) const{
        return  type(mailbox[move.from()]) == PAWN ||                                       // mover == PAWN (takes care of ep+promo)
                (type(mailbox[move.to()]) != SPACE && move.getFlag() != Move::fcastle);     // capture and not castling
    }
    forceinline bool isCastling(const Move& move) const{
        return move.getFlag() == Move::fcastle;
    }
    forceinline bool isTactical(const Move& move) const{
        return move.getFlag() > Move::fcastle ||                            //promotion+ep
                (!move.getFlag() && type(mailbox[move.to()]) != SPACE);     //!castling + capture
    }
    forceinline int getCapture(const Move& move) const{
        return type(mailbox[move.to()])*                            //normal capture
                (move.getFlag() != Move::fep)+                      //ep => x0 => capture=0=PAWN
                (SPACE-ROOK)*(move.getFlag() == Move::fcastle);     //castle => previous=ROOK => ROOK+SPACE-ROOK = SPACE => no capture
    }
};

struct PositionSnapshot;

//Represents a state in the game
class GameState{
    // (not necessary if we create new states for exploration)
    ExpendedMove movesSinceBeginning[maxPly]; // maximum number of moves https://www.reddit.com/r/chess/comments/168qmk6/longest_possible_chess_game_88485_moves/
    big repHist[maxPly];
    int rule50[maxPly];


    //Contains a bitboard of the white pieces, then a bitboard of the black pieces
    void testPawnZobr();

    friend struct PositionSnapshot;

public : 
    void updateZobrists(int piece, bool color, int square);
    //To determine whose turn it is to play
    int turnNumber;
    big zobristHash;
    big pawnZobrist;
    big minorZobrist;
    PositionState board;
    //End of last double pawn push, (-1) if last move was not a double pawn push
    int lastDoublePawnPush;
    big castlingMask;
    GameState();
    void setDFRC(int idWhite, int idBlack);
    void fromFen(string fen);
    string toFen() const;
    int friendlyColor() const;
    int enemyColor() const;
    bool isEnPassantPossibility(const int piece, const Move& move);
    int rule50_count() const;
    bool twofold() const;
    bool threefold() const;
    void playNullMove();
    ExpendedMove getLastMove() const;
    ExpendedMove getContMove() const;
    Move playPartialMove(Move move);
    int getPiece(int square) const;
    int getfullPiece(int square) const;
    big getFriendlyMask(int piece) const;
    big getEnemyMask(int piece) const;
    void print() const;
    void initMove(Move& move);

    // Forward-only move application (no undo support needed)
    ExpendedMove playMove(Move move);
    // Optimized repetition detection: step by 2 (zobrist includes turn bit)
    bool twofoldFast();
    bool threefoldFast();
    void castlingFromMask(big mask);
    int material();
};

const string startpos="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Lightweight snapshot of mutable position state (~132 bytes)
// Used for copy/make: save before playMoveForward, restore after search
// No nbMoves, posRook, or deathRook — not needed in forward-only mode
struct PositionSnapshot {
    PositionState board;
    big zobristHash;
    big pawnZobrist;
    big minorZobrist;
    int lastDoublePawnPush;
    big castlingMask;
    int turnNumber;

    inline void save(const GameState& s) {
        memcpy(&board, &s.board, sizeof(board));
        zobristHash = s.zobristHash;
        pawnZobrist = s.pawnZobrist;
        minorZobrist = s.minorZobrist;
        lastDoublePawnPush = s.lastDoublePawnPush;
        castlingMask = s.castlingMask;
        turnNumber = s.turnNumber;
    }

    inline void restore(GameState& s) const {
        memcpy(&s.board, &board, sizeof(board));
        s.zobristHash = zobristHash;
        s.pawnZobrist = pawnZobrist;
        s.minorZobrist = minorZobrist;
        s.lastDoublePawnPush = lastDoublePawnPush;
        s.castlingMask = castlingMask;
        s.turnNumber = turnNumber;
    }
};

#endif
