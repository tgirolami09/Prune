#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "embeder.hpp"
using namespace std;

class  __attribute__((packed)) constTable{
public:
    big magic;
    int decR;
    int bits;
};

big parseInt(int& pointer);
static constexpr int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
extern big KnightMoves[64]; //Knight moves for each position of the board
extern big pieceCastlingMasks[2][2];
extern big attackCastlingMasks[2][2];
extern big normalKingMoves[64];
extern big attackPawns[128];
extern big directions[64][64];
extern big fullDir[64][64];
void PrecomputeKnightMoveData();
void precomputeDirections();
void load_table();
void clear_table();
void precomputeCastlingMasks();
void precomputeNormlaKingMoves();
void precomputePawnsAttack();
big moves_table(int index, big mask_pieces);

class LegalMoveGenerator{
private:

    const int doubleCheckFromSameType = -100;

    //A mask for allowed squares given pins for every piece on the board
    big pinnedMasks[64];

    template<bool isPawn>
    void maskToMoves(int start, big mask, Move* moves, int& nbMoves, int8_t piece, bool promotQueen=false);
    big pseudoLegalBishopMoves(int bishopPosition, big allPieces);
    big pseudoLegalRookMoves(int rookPosition, big allPieces);
    big pseudoLegalQueenMoves(int queenPositions, big allPieces);
    big pseudoLegalKnightMoves(int knightPosition);
    big pseudoLegalPawnMoves(int pawnPosition, bool color, big allPieces, int friendKingPos, big moveMask = -1, big captureMask = -1, big enemyPieces = -1, int enPassant = -1, big enemyRooks = 0);
    big pseudoLegalKingMoves(int kingPosition,const big Pieces, bool color, bool kingCastling, bool queenCastling);
    int dealWithEnemyPawns(big enemyPawnPositions, int friendKingPos, int enemyColor);
    int dealWithEnemyKnights(big enemyKnightPositions, int friendKingPos);
    int dealWithEnemyBishops(big enemyBishopPositions, big Pieces, int friendKingPos);
    int dealWithEnemyRooks(big enemyRookPositions, big allPieces, int friendKingPos);
    void dealWithEnemyKing(int enemyKingPos);
    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves, big allPieces, big captureMask = -1);
    void legalPawnMoves(big pawnMask, bool friendlyColor, int lastDoublePawnPush, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves, big allPieces, big enemyRooks, bool promotQueen=false);
    void legalKnightMoves(big knightMask, big moveMask, big captureMask, Move* knightMoves, int& nbMoves);
    void legalSlidingMoves(big moveMask, big captureMask, Move* slidingMoves, int& nbMoves, big allPieces);

    const big* friendlyPieces;
    const big* enemyPieces;
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