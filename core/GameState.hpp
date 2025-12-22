#ifndef GAMESATE_HPP
#define GAMESATE_HPP
#include "Move.hpp"
#include "Const.hpp"
using namespace std;
const int maxPly = 8848*2+2;
const int zobrCastle=64*2*6;
const int zobrPassant=zobrCastle+4;
const int zobrTurn=zobrPassant+8;
const int nbZobrist=zobrTurn+1;
const int sizeThreeFold=8192;
extern big zobrist[nbZobrist];
void init_zobrs();
//Represents a state in the game
class GameState{
    // (not necessary if we create new states for exploration)
    Move movesSinceBeginning[maxPly]; // maximum number of moves https://www.reddit.com/r/chess/comments/168qmk6/longest_possible_chess_game_88485_moves/
    big repHist[maxPly];
    int rule50[maxPly];

    //To determine whose turn it is to play AND rules that involve turn count
    int turnNumber;

    //Contains a bitboard of the white pieces, then a bitboard of the black pieces
    short nbMoves[2][3];
    short posRook[2][2];
    short deathRook[2][2];
    int startEnPassant;
    void updateZobrists(int piece, bool color, int square);
    void testPawnZobr();
public : 
    big zobristHash;
    big pawnZobrist;
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
    template<bool enable, int side>
    void changeCastlingRights(int c);
    template<bool back, int side>
    void updateCastlingRights(int c, int pos);
    template<bool back>
    void moveKing(int c);
    void captureRook(int pos, int c);
    void uncaptureRook(int pos, int c);
    template<bool back>
    bool isEnPassantPossibility(const Move& move);
    int rule50_count() const;
    bool twofold();
    bool threefold();
    template<bool back=false>
    int playMove(Move move);
    void playNullMove();
    void undoNullMove();
    Move getLastMove() const;
    Move getContMove() const;
    void undoLastMove();
    Move playPartialMove(Move move);
    int getPiece(int square, int c);
    int getfullPiece(int square) const;
    pawnStruct getPawnStruct();
    void print() const;
    void initMove(Move& move);
};

const string startpos="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#endif
