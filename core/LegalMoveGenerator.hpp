#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "embeder.hpp"
using namespace std;

class  __attribute__((packed)) constTable{
public:
    int bits;
    big magic;
};

big parseInt(int& pointer);
extern big KnightMoves[64]; //Knight moves for each position of the board
extern big pieceCastlingMasks[2][2];
extern big attackCastlingMasks[2][2];
extern big normalKingMoves[64];
extern big attackPawns[128];
void PrecomputeKnightMoveData();
void load_table();
void clear_table();
void precomputeCastlingMasks();
void precomputeNormlaKingMoves();
void precomputePawnsAttack();
big moves_table(int index, big mask_pieces, big mask);

class LegalMoveGenerator{
private:

    const int doubleCheckFromSameType = -100;

    //Pin ray bitboards: union of all pin rays of each type
    big pinHV;   // horizontal/vertical pin rays (includes pinner + ray + pinned piece)
    big pinD12;  // diagonal pin rays

    template<bool isPawn>
    void maskToMoves(int start, big mask, Move* moves, int& nbMoves, int8_t piece, bool promotQueen=false);
    big pseudoLegalBishopMoves(int bishopPosition, big allPieces);
    big pseudoLegalRookMoves(int rookPosition, big allPieces);

    big pseudoLegalKnightMoves(int knightPosition);
    template<bool IsWhite, bool canCapture, bool canQuiet>
    big pseudoLegalPawnMoves(int pawnPosition, big allPieces, int friendKingPos, big moveMask = -1, big captureMask = -1, big enemyPieces = -1, int enPassant = -1, big enemyRooks = 0);
    big pseudoLegalKingMoves(int kingPosition);
    template<bool IsWhite>
    int dealWithEnemyPawns(big enemyPawnPositions, int friendKingPos);
    int dealWithEnemyKnights(big enemyKnightPositions, int friendKingPos);
    int dealWithEnemyBishops(big enemyBishopPositions, big Pieces, int friendKingPos);
    int dealWithEnemyRooks(big enemyRookPositions, big allPieces, int friendKingPos);
    void dealWithEnemyKing(int enemyKingPos);
    template<bool IsWhite>
    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves, big allPieces, big captureMask = -1);
    template<bool IsWhite>
    void legalPawnMoves(big pawnMask, int lastDoublePawnPush, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves, big allPieces, big enemyRooks, bool promotQueen=false);
    void legalKnightMoves(big knightMask, big moveMask, big captureMask, Move* knightMoves, int& nbMoves);
    void legalSlidingMoves(big moveMask, big captureMask, Move* slidingMoves, int& nbMoves, big allPieces);
    template<bool IsWhite>
    bool initDangersImpl(const GameState& state);
    template<bool IsWhite, bool InCheck>
    int generateLegalMovesImpl(const GameState& state, bool& inCheck, Move* legalMoves, big& dangerPositions, bool onlyCapture);
    template<bool IsWhite>
    Move getLVAImpl(int posCapture, GameState& state);

    big friendlyPieces[6];
    big enemyPieces[6];
    big allFriends;
    big allEnemies;
    big allPieces;

    int friendlyKingPosition;
    int enemyKingPosition;
    big allDangerSquares;
    int nbCheckers;
    int checkerPos;

public : 
    bool isCheck() const;
    bool initDangers(const GameState& state);
    int generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves,big& dangerPositions, bool onlyCapture=false);
    Move getLVA(int posCapture, GameState& state);
};
#endif