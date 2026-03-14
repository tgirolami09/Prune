#ifndef GAMESATE_HPP
#define GAMESATE_HPP
#include "Move.hpp"
#include "Const.hpp"
#include <cstring>
using namespace std;
const int maxPly = 8848*2+2;
const int zobrCastle=64*2*6;
const int zobrPassant=zobrCastle+4;
const int zobrTurn=zobrPassant+8;
const int nbZobrist=zobrTurn+1;
const int sizeThreeFold=8192;
extern big zobrist[nbZobrist];

struct PositionSnapshot;

//Represents a state in the game
class GameState{
    // (not necessary if we create new states for exploration)
    Move movesSinceBeginning[maxPly]; // maximum number of moves https://www.reddit.com/r/chess/comments/168qmk6/longest_possible_chess_game_88485_moves/
    big repHist[maxPly];
    int rule50[maxPly];

    //To determine whose turn it is to play AND rules that involve turn count
    int turnNumber;

    //Contains a bitboard of the white pieces, then a bitboard of the black pieces
    short posRook[2][2];
    void updateZobrists(int piece, bool color, int square);
    void testPawnZobr();

    friend struct PositionSnapshot;

public : 
    big zobristHash;
    big pawnZobrist;
    big minorZobrist;
    big boardRepresentation[2][6];
    //End of last double pawn push, (-1) if last move was not a double pawn push
    int lastDoublePawnPush;
    bool castlingRights[2][2];

    GameState();
    void fromFen(string fen);
    string toFen() const;
    int friendlyColor() const;
    int enemyColor() const;
    //Returns the 6 bitboards of the FRIENDLY pieces on the board
    const big* friendlyPieces() const;
    //Returns the 6 bitboards of the ENEMY pieces on the board
    const big* enemyPieces() const;
    template<bool back>
    bool isEnPassantPossibility(const Move& move);
    int rule50_count() const;
    bool twofold();
    bool threefold();
    int playMove(Move move);
    void playNullMove();
    Move getLastMove() const;
    Move getContMove() const;
    Move playPartialMove(Move move);
    int getPiece(int square, int c);
    int getfullPiece(int square) const;
    pawnStruct getPawnStruct();
    void print() const;
    void initMove(Move& move);
    big castlingMask();

    // Forward-only move application (no undo support needed)
    void playMoveForward(Move move);
    void playNullMoveForward();
    Move playPartialMoveForward(Move move);
    // Optimized repetition detection: step by 2 (zobrist includes turn bit)
    bool twofoldFast();
    bool threefoldFast();
};

const string startpos="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Lightweight snapshot of mutable position state (~132 bytes)
// Used for copy/make: save before playMoveForward, restore after search
// No nbMoves, posRook, or deathRook — not needed in forward-only mode
struct PositionSnapshot {
    big boardRepresentation[2][6];
    big zobristHash;
    big pawnZobrist;
    big minorZobrist;
    int lastDoublePawnPush;
    bool castlingRights[2][2];
    int turnNumber;

    inline void save(const GameState& s) {
        memcpy(boardRepresentation, s.boardRepresentation, sizeof(boardRepresentation));
        zobristHash = s.zobristHash;
        pawnZobrist = s.pawnZobrist;
        minorZobrist = s.minorZobrist;
        lastDoublePawnPush = s.lastDoublePawnPush;
        memcpy(castlingRights, s.castlingRights, sizeof(castlingRights));
        turnNumber = s.turnNumber;
    }

    inline void restore(GameState& s) const {
        memcpy(s.boardRepresentation, boardRepresentation, sizeof(boardRepresentation));
        s.zobristHash = zobristHash;
        s.pawnZobrist = pawnZobrist;
        s.minorZobrist = minorZobrist;
        s.lastDoublePawnPush = lastDoublePawnPush;
        memcpy(s.castlingRights, castlingRights, sizeof(castlingRights));
        s.turnNumber = turnNumber;
    }
};

#endif
