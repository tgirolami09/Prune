#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "embeder.hpp"
using namespace std;

class __attribute__((packed)) constTable {
   public:
    int bits;
    u64 magic;
};

u64 parseInt(int& pointer);
extern u64 KnightMoves[64];  // Knight moves for each position of the board
extern u64 pieceCastlingMasks[2][2];
extern u64 attackCastlingMasks[2][2];
extern u64 normalKingMoves[64];
extern u64 attackPawns[128];
void PrecomputeKnightMoveData();
void load_table();
void clear_table();
void precomputeCastlingMasks();
void precomputeNormlaKingMoves();
void precomputePawnsAttack();
u64 moves_table(int index, u64 mask_pieces, u64 mask);

static constexpr int doubleCheckFromSameType = -100;
class LegalMoveGenerator {
   private:
    // Pin ray bitboards: union of all pin rays of each type
    u64 pinHV;   // horizontal/vertical pin rays (includes pinner + ray + pinned
                 // piece)
    u64 pinD12;  // diagonal pin rays

    template <bool isPawn>
    void maskToMoves(int start, u64 mask, Move* moves, int& nbMoves, int8_t piece,
                     bool promotQueen = false);
    u64 pseudoLegalBishopMoves(int bishopPosition, u64 allPieces);
    u64 pseudoLegalRookMoves(int rookPosition, u64 allPieces);

    u64 pseudoLegalKnightMoves(int knightPosition);
    template <bool IsWhite, bool canCapture, bool canQuiet>
    u64 pseudoLegalPawnMoves(int pawnPosition, u64 allPieces, int friendKingPos, u64 moveMask = -1,
                             u64 captureMask = -1, u64 enemyPieces = -1, int enPassant = -1,
                             u64 enemyRooks = 0);
    u64 pseudoLegalKingMoves(int kingPosition);
    template <bool IsWhite>
    int dealWithEnemyPawns(u64 enemyPawnPositions, int friendKingPos);
    int dealWithEnemyKnights(u64 enemyKnightPositions, int friendKingPos);
    int dealWithEnemyBishops(u64 enemyBishopPositions, u64 Pieces, int friendKingPos);
    int dealWithEnemyRooks(u64 enemyRookPositions, u64 allPieces, int friendKingPos);
    void dealWithEnemyKing(int enemyKingPos);
    template <bool IsWhite>
    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves, u64 allPieces,
                        u64 captureMask = -1);
    template <bool IsWhite>
    void legalPawnMoves(u64 pawnMask, int lastDoublePawnPush, u64 moveMask, u64 captureMask,
                        Move* pawnMoves, int& nbMoves, u64 allPieces, u64 enemyRooks,
                        bool promotQueen = false);
    void legalKnightMoves(u64 knightMask, u64 moveMask, u64 captureMask, Move* knightMoves,
                          int& nbMoves);
    void legalSlidingMoves(u64 moveMask, u64 captureMask, Move* slidingMoves, int& nbMoves,
                           u64 allPieces);
    template <bool IsWhite>
    bool initDangersImpl(const GameState& state);
    template <bool IsWhite, bool InCheck>
    int generateLegalMovesImpl(const GameState& state, bool& inCheck, Move* legalMoves,
                               u64& dangerPositions, bool onlyCapture);
    template <bool IsWhite>
    Move getLVAImpl(int posCapture, GameState& state);

    u64 friendlyPieces[6];
    u64 enemyPieces[6];
    u64 allFriends;
    u64 allEnemies;
    u64 allPieces;

    int friendlyKingPosition;
    int enemyKingPosition;
    u64 allDangerSquares;
    int nbCheckers;
    int checkerPos;

   public:
    bool isCheck() const;
    bool initDangers(const GameState& state);
    int generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves,
                           u64& dangerPositions, bool onlyCapture = false);
    Move getLVA(int posCapture, GameState& state);
};
#endif