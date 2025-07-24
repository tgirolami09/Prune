#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include <fstream>
#include <utility>
using namespace std;


class constTable{
public:
    int bits;
    int decR;
    big magic;
};

//Class to generate legal moves
class LegalMoveGenerator{
    int KnightOffSets[8] = {15, 17, -17, -15, 10, -6, 6, -10};

    big KnightMoves[64] = {0}; //Knight moves for each position of the board

    void PrecomputeKnightMoveData(){
        const pair<int, int> moves[8] = {
            {-2,  1},
            {-2, -1},
            {-1,  2},
            {-1, -2},
            { 1,  2},
            { 1, -2},
            { 2,  1},
            { 2, -1}};
        for (int row = 0; row<8; ++row){
            for (int col = 0; col<8; ++col){
                int squareIndex = row * 8 + col;
                //Precompute knight moves
                big knightMoveMask = 0;
                for(pair<int, int> move:moves){
                    //0 is up and 1 is down
                    int square = squareIndex;
                    if(row+move.first < 8 && row+move.first >= 0)
                        square += 8*move.first;
                    else continue;
                    if(col+move.second < 8 && col+move.second >= 0)
                        square += move.second;
                    else continue;
                    knightMoveMask |= 1ULL << square;
                }
                KnightMoves[squareIndex] = knightMoveMask;
            }
        }  
    }

    constTable constantsMagic[128];
    big* tableMagic[129];
    void load_table(string name){
        ifstream file(name);
        big magic;
        int decR, minimum, size;
        big mask;
        int current = 0;
        while(file >> magic){
            file >> decR >> minimum >> size;
            constantsMagic[current] = {minimum, decR, magic};
            tableMagic[current] = (big*)calloc(size, sizeof(big));
            for(int i=0; i<size; i++){
                file >> mask;
                tableMagic[current][i] = mask;
            }
            current++;
        }
    }

    big moves_table(int index, big mask_pieces){
        return tableMagic[index][(mask_pieces*constantsMagic[index].magic & (MAX_BIG >> constantsMagic[index].decR)) >> (64-constantsMagic[index].decR-constantsMagic[index].bits)];
    }


    big directions[64][64];
    big fullDir[64][8];
    static constexpr int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    void precomputeDirections(){
        for(int row=0; row<8; row++){
            for(int col=0; col<8; col++){
                int square = row*8+col;
                for(int idDir=0; idDir<8; idDir++){
                    int r=row+dirs[idDir][0];
                    int c=col+dirs[idDir][1];
                    big mask = 0;
                    while(r >= 0 && r < 8 && c >= 0 && c < 8){
                        int sq = (r*8+c);
                        mask |= 1ULL << sq;
                        directions[square][sq] = mask; // line of 1 between square and sq
                        r += dirs[idDir][0];
                        c += dirs[idDir][1];
                    }
                    fullDir[square][idDir] = mask;
                    //printf("%d (%d %d) %d %d\n", square, col, row, dirs[idDir][0], dirs[idDir][1]);
                    //print_mask(fullDir[square][idDir]);
                }
            }
        }
    }

    inline int isAttacking(big maskPiece, int dir){
        if(maskPiece & enemyPieces[QUEEN])return QUEEN;
        if(dir >= 4)dir -= 3;
        if(dir%2)return (maskPiece&enemyPieces[ROOK]) != 0?ROOK:-1;
        else return (maskPiece&enemyPieces[BISHOP]) != 0?BISHOP:-1;
    }
    inline int firstPiece(big mask, int dir){
        if(dir < 4)
            return 63-__builtin_clzll(mask);
        return __builtin_ctzll(mask);
    }
    big pinned(int square, bool isCheck, Move* pinnedMoves, int& nbMoves, bool color){
        big maskPinned=0;
        for(int idDir = 0; idDir<8; idDir++){
            big mask = fullDir[square][idDir]&allPieces;
            if(!mask)continue;
            int posFirst = firstPiece(mask, idDir);
            big maskFirst=1ULL << posFirst;
            if(maskFirst & allEnemyPieces)continue; //may be a checker if needed
            big newMask = mask&(~maskFirst);
            if(!newMask)continue;
            int posSecond = firstPiece(newMask, idDir);
            big maskSecond = 1ULL << posSecond;
            int8_t pieceAttack = isAttacking(maskSecond, idDir);
            if(pieceAttack != -1){
                maskPinned |= maskFirst;
                if(!isCheck){
                    int piece=-1;
                    for(int typePiece:{BISHOP, ROOK, QUEEN, PAWN}){
                        if(maskFirst&friendlyPieces[typePiece])
                            piece = typePiece;
                    }
                    int mdir = idDir >= 4?idDir-3:idDir;
                    if(piece == QUEEN)
                        maskToMoves<QUEEN>(posFirst, directions[square][posSecond]&~maskFirst, pinnedMoves, nbMoves);
                    else if(mdir%2 == 1 && piece == ROOK)
                        maskToMoves<ROOK>(posFirst, directions[square][posSecond]&~maskFirst, pinnedMoves, nbMoves);
                    else if(mdir%2 == 0 && piece == BISHOP)
                        maskToMoves<BISHOP>(posFirst, directions[square][posSecond]&~maskFirst, pinnedMoves, nbMoves);
                    else if(piece == PAWN){
                        int moveFactor = color ? -1 : 1;
                        big maskPawnMoves = 0;
                        if(mdir%2 == 0){
                            if(posFirst+7*moveFactor == posSecond || posFirst+9*moveFactor == posSecond){
                                maskPawnMoves |= 1ULL << posSecond;
                            }
                        }else if(idDir == 1 || idDir == 6){
                            maskPawnMoves |= ((1ul<<(posFirst + 8 * moveFactor)) & (~allPieces));
                            //Double pawn push
                            int rowPiece = row(posFirst);
                            if (((rowPiece==1 && color == 0) || (rowPiece==6 && color == 1)) && (maskPawnMoves!=0)){
                                maskPawnMoves |= ((1ul<<(posFirst + 16 * moveFactor)) & (~allPieces));
                            }
                        }
                        maskToMoves<PAWN>(posFirst, maskPawnMoves, pinnedMoves, nbMoves);
                    }
                }
            }
        }
        return maskPinned;
    }

    bool isAttacked(int square, bool color, big pieces){
        int moveFactor = color ? -1 : 1;
        for(int idDir=0; idDir<8; idDir++){
            big mask = fullDir[square][idDir]&pieces;
            if(!mask)continue;
            int posFirst = firstPiece(mask, idDir);
            big maskFirst=1ULL << posFirst;
            if(isAttacking(maskFirst, idDir) != -1)
                return true;
            int mdir = idDir >= 4?idDir-3:idDir;
            if(mdir%2 == 0 && maskFirst&enemyPieces[PAWN]){
                int diff = (posFirst-square)*moveFactor;
                if(maskFirst&enemyPieces[PAWN]){
                    if(diff == 7 || diff == 9)
                        return true;
                }
            }
        }
        if(KnightMoves[square]&enemyPieces[KNIGHT])
            return true;
        int enemyKing = __builtin_ctzll(enemyPieces[KING]);
        int dist = abs(square-enemyKing);
        if(dist == 1 || (dist >= 7 && dist <= 9))
            return true;
        return false;
    }

    short fullDirAttacks(int square, bool color, big pieces){
        short dirsAtk=0;
        int moveFactor = color ? -1 : 1;
        for(int idDir=0; idDir<8; idDir++){
            big mask = fullDir[square][idDir]&pieces;
            if(!mask)continue;
            int posFirst = firstPiece(mask, idDir);
            big maskFirst=1ULL << posFirst;
            if(isAttacking(maskFirst, idDir) != -1)
                dirsAtk |= 1 << idDir;
            int mdir = idDir >= 4?idDir-3:idDir;
            if(mdir%2 == 0 && maskFirst&enemyPieces[PAWN]){
                int diff = (posFirst-square)*moveFactor;
                if(maskFirst&enemyPieces[PAWN]){
                    if(diff == 7 || diff == 9)
                        dirsAtk |= 1<<(diff+1);
                }
            }
        }
        if(KnightMoves[square]&enemyPieces[KNIGHT])
            dirsAtk |= 1 << 9;
        return dirsAtk;
    }

    big attackPawns[128];
    void precomputePawnsAttack(){
        for(int c=0; c<2; c++){
            int leftLimit = c ?  7 : 0;
            int rightLimit = leftLimit^7;
            int moveFactor = c ? -1 : 1;
            for(int square=0; square<64; square++){
                int key = c*64+square;
                attackPawns[key] = 0;
                if(square+8*moveFactor < 0 || square+8*moveFactor >= 64)continue;
                int pieceCol = col(square);
                if(pieceCol != leftLimit)
                    attackPawns[key] |= 1ul<<(square + 7 * moveFactor);
                if(pieceCol != rightLimit)
                    attackPawns[key] |= 1ul<<(square + 9 * moveFactor);
            }
        }
    }

    big maskCastling[2][2];
    void precomputeCastlingMasks(){
        maskCastling[0][1] = 0b00000110;
        maskCastling[0][0] = 0b01110000;
        maskCastling[1][1] = maskCastling[0][1] << 56;
        maskCastling[1][0] = maskCastling[0][0] << 56;
    }
public:
    LegalMoveGenerator(){
        PrecomputeKnightMoveData();
        load_table("magics.out");
        init_lines();
        precomputeDirections();
        precomputePawnsAttack();
        precomputeCastlingMasks();
    }
    ~LegalMoveGenerator(){
        for(int i=0; i<128; i++){
            free(tableMagic[i]);
        }
    }

private: 
    //Transforms a bitboard of valid end positions into a list of the corresponding moves
    template<int piece>
    void maskToMoves(int start, big mask, Move* moves, int& nbMoves){
        while(mask){
            int bit = __builtin_ctzll(mask);
            mask &= mask-1;
            //Need to add logic for pawn promotion
            Move base = {(int8_t)start, (int8_t)bit, piece};
            big mask = 1ULL << bit;
            for(int i=0; i<6; i++)
                if(enemyPieces[i]&mask)base.capture = i;
            if(piece == PAWN && (row(bit) == 7 || row(bit) == 0)){
                for(int8_t typePiece:{KNIGHT, BISHOP, ROOK, QUEEN}){
                    moves[nbMoves] = base;
                    moves[nbMoves].promoteTo = typePiece;
                    nbMoves++;
                }
            }else{
                if(piece == PAWN && (col(start) != col(bit)) && base.capture == -2){
                    base.capture = -1;
                }
                moves[nbMoves] = base;
                nbMoves++;
            }
        }
    }

    int pseudoLegalBishopMoves(big positions, big allPieces, big friendlyPieces, big* allMasks){
        big bishopMask = positions;
        ubyte pos[10]; //max number of bishop
        int nbPos=places(bishopMask, pos);
        //Need to rewrite each time
        //vector<big> allMasks(nbPos+1,0);
        //1 more with all possible moves from piece type
        allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            //La logique de recuperation des coups
            big bishopMoveMask=moves_table(pos[p], allPieces&mask_empty_bishop(pos[p]));
            bishopMoveMask &= ~friendlyPieces;
            allMasks[p] = bishopMoveMask;
            allMasks[nbPos] |= bishopMoveMask;
        }
        //return allMasks;
        return nbPos;
    }

    int pseudoLegalRookMoves(big positions, big allPieces, big friendlyPieces, big* allMasks){
        big rookMask = positions;
        ubyte pos[10]; //max number of rooks
        int nbPos=places(rookMask, pos);
        //vector<big> allMasks(nbPos+1,0);
        allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            //La logique de recuperation des coups
            big rookMoveMask=moves_table(pos[p]+64, allPieces&mask_empty_rook(pos[p]));
            rookMoveMask &= ~friendlyPieces;
            allMasks[p] = rookMoveMask;
            allMasks[nbPos] |= rookMoveMask;
        }
        //return allMasks;
        return nbPos;
    }

    int pseudoLegalQueenMoves(big positions, big allPieces, big friendlyPieces, big* allMasks){
        big queenMask = positions;
        ubyte pos[10]; //max number of queens
        int nbPos=places(queenMask, pos);
        //vector<big> allMasks(nbPos+1,0);
        big bishopMoves[11];
        big rookMoves[11];
        int nbBishops = pseudoLegalBishopMoves(positions,allPieces,friendlyPieces, bishopMoves);
        int nbRooks = pseudoLegalRookMoves(positions,allPieces,friendlyPieces, rookMoves);

        // +1 to get the extra bitboard (contains all possible moves of a piece type)
        for (int i = 0;i<nbPos+1;++i){
            allMasks[i] = bishopMoves[i] | rookMoves[i];
        }

        //return allMasks;
        return nbPos;
    }

    int pseudoLegalKnightMoves(big positions, big friendlyPieces, big* allMasks){
        big knightMask = positions;
        ubyte pos[10]; //max number of knight
        int nbPos=places(knightMask, pos);
        //vector<big> allMasks(nbPos+1,0);
        allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            big knightEndMask = KnightMoves[pos[p]] & (~friendlyPieces);
            allMasks[p] = knightEndMask;
            allMasks[nbPos] |= knightEndMask;
        }
        return nbPos;
    }

    int pseudoLegalPawnMoves(big positions, big allPieces, big enemyPieces, bool color, big moveMask, big captureMask, int enPassant, big* allMasks){
        //If color == 0 -> white (add to move)
        //If color == 1 -> black (subtract to move)
        big pawnMask = positions;
        ubyte pos[8]; //max number of pawns
        int nbPos=places(pawnMask, pos);
        //vector<big> allMasks(nbPos+1,0);
        allMasks[nbPos] = 0;
        //Should be if color is 1 (true) then moveFactor is -1
        int moveFactor = color ? -1 : 1;
        int leftLimit = color ?  7 : 0;
        int rightLimit = leftLimit^7;
        int rowPassant = color?3:4;
        if(enPassant != -1){
            enemyPieces |= 1ULL << ((rowPassant^1)*8|enPassant);
        }
        int startRow = color?6:1;
        for (int p = 0;p<nbPos;++p){
            big pawnMoveMask = 0;
            big pawnCaptureMask = 0;
            int pieceRow = row(pos[p]);
            int pieceCol = col(pos[p]);
            //Single pawn push (check there are no pieces on target square)
            big pushMask = 1ULL << (pos[p]+8*moveFactor);
            pawnMoveMask |= (pushMask) & (~allPieces);

            //Double pawn push
            if (pieceRow == startRow && pawnMoveMask){
                if(color)pushMask >>= 8;
                else pushMask <<= 8;
                pawnMoveMask |= pushMask & (~allPieces);
            }
            pawnCaptureMask = attackPawns[color*64+pos[p]]&enemyPieces;
            /*//Capture left
            if(pieceCol != leftLimit)
                pawnCaptureMask |= ((1ul<<(pos[p] + 7 * moveFactor)) & (enemyPieces));

            //Capture right
            if(pieceCol != rightLimit)
                pawnCaptureMask |= ((1ul<<(pos[p] + 9 * moveFactor)) & (enemyPieces));
            */
            //TODO : capture en-passant
            //if(enPassant != -1 && abs(pieceCol-enPassant) == 1 && pieceRow == rowPassant)
            //    pawnCaptureMask |= 1ul << ((rowPassant^1)*8+enPassant);

            allMasks[p] = (pawnMoveMask & moveMask) | (pawnCaptureMask & captureMask);
            allMasks[nbPos] |= allMasks[p];
        }
        //return allMasks;
        return nbPos;
    }  
    
    big pseudoLegalKingMoves(big positions, big friendlyPieces, bool color, bool kingCastling, bool queenCastling, bool inCheck){
        big kingMask = positions;
        int kingPos = __builtin_ctzll(kingMask);

        big kingEndMask = 0;
        //UP, DOWN, LEFT, RIGHT
        int transitionsRow[2] = {8, -8};
        int transitionsCol[2] = {-1, 1};
        int nbCol=2;
        int nbRow=2;
        if (row(kingPos)==7){
            swap(transitionsRow[0], transitionsRow[1]);
            nbRow--;
        }else if (row(kingPos)==0){
            nbRow--;
        }
        if (col(kingPos)==0){
            swap(transitionsCol[0], transitionsCol[1]);
            nbCol--;
        }else if (col(kingPos)==7){
            nbCol--;
        }
        for (int i = 0; i < nbRow;++i){
            int trans1 = transitionsRow[i];
            int cardEnd = kingPos + trans1;
            if(!isAttacked(cardEnd, color, allPieces ^ kingMask))
                kingEndMask |= (1ul<< cardEnd);
            for (int j = 0; j < nbCol;++j){
                int trans2 = transitionsCol[j];
                int diagEnd = cardEnd + trans2;
                if(!isAttacked(diagEnd, color, allPieces ^ kingMask))
                    kingEndMask |= (1ul<< diagEnd);
            }
        }
        for(int i=0; i<nbCol; i++){
            int cardEnd = kingPos+transitionsCol[i];
            if(!isAttacked(cardEnd, color, allPieces ^ kingMask))
                kingEndMask |= (1ul<< cardEnd);
        }
        int posCastle=color*56+1;
        if(!inCheck && kingCastling && (kingEndMask & maskCastling[color][1]) && !(maskCastling[color][1]&allPieces) && !isAttacked(posCastle, color, allPieces)){
            kingEndMask |= 1ULL << posCastle;
        posCastle = color*56+5;
        }if(!inCheck && queenCastling && (kingEndMask & maskCastling[color][0]) && !(maskCastling[color][0]&allPieces) && !isAttacked(posCastle, color, allPieces)){
            kingEndMask |= 1ULL << posCastle;
        }
        return kingEndMask & (~friendlyPieces);

    }

    const big* enemyPieces;
    big allEnemyPieces = 0;
    const big* friendlyPieces;
    big allFriendlyPieces = 0;
    big allPieces = 0;
    //Squares that are being targeted by the enemy (ignoring the friendly king)
    /*big allDangerSquares = 0;
    big slidingDangerSquares;
    big allEnemyBishopDangers;
    big allEnemyRookDangers;
    big allEnemyQueenDangers;
    big allEnemyKnightDangers;
    big allEnemyPawnDangers;
    big allEnemyKingDangers;

    big kingAsBishopAttacks;
    big kingAsRookAttacks;
    big kingAsQueenAttacks;
    big kingAsSlidingPieceAttacks;*/

    //An OR between danger squares and kingAsSlidingPieceAttacks finalised with and AND of all the pieces
    //For now ban pinned pieces to move (TODO : allow them the ray)
    big pinnedPiecesMasks = 0;

    void recalculateAllMasks(const GameState& state){
        enemyPieces = state.enemyPieces();
        allEnemyPieces = 0;

        friendlyPieces = state.friendlyPieces();
        allFriendlyPieces = 0;

        for (int i = 0; i < 6 ; ++i){            
            allEnemyPieces |= enemyPieces[i];
            allFriendlyPieces |= friendlyPieces[i];
        }

        allPieces = allEnemyPieces | allFriendlyPieces;

        //There are no friendly pieces here because we are looking for protected pieces
        //We also remove the friendly king for checking sliders
        /*big enemyBishopDangers[11];
        big enemyRookDangers[11];
        big enemyQueenDangers[11];
        int nbBishops = pseudoLegalBishopMoves(enemyPieces[BISHOP],allPieces ^ friendlyPieces[KING],0, enemyBishopDangers);
        int nbRooks = pseudoLegalRookMoves(enemyPieces[ROOK],allPieces ^ friendlyPieces[KING],0, enemyRookDangers);
        int nbQueens = pseudoLegalQueenMoves(enemyPieces[QUEEN],allPieces ^ friendlyPieces[KING],0, enemyQueenDangers);

        big enemyKnightDangers[11];
        int nbKnights = pseudoLegalKnightMoves(enemyPieces[KNIGHT],0, enemyKnightDangers);
        //Set the move mask to 0 because we are only interested in dangers (captures or protections)
        //All pieces are enemies because we are only interested in dangers (captures or protections)
        // -1 instead of enemy pieces because we want all possible attacks even if not possible this turn
        big enemyPawnDangers[11];
        int nbPawns = pseudoLegalPawnMoves(enemyPieces[PAWN],allPieces,-1,state.enemyColor(),0,-1, enemyPawnDangers);
        big enemyKingDangers = pseudoLegalKingMoves(enemyPieces[KING],0, state.enemyColor());

        allEnemyBishopDangers = enemyBishopDangers[nbBishops];
        allEnemyRookDangers = enemyRookDangers[nbRooks];
        allEnemyQueenDangers = enemyQueenDangers[nbQueens];
        allEnemyKnightDangers = enemyKnightDangers[nbKnights];
        allEnemyPawnDangers = enemyPawnDangers[nbPawns];
        //There should only be one king of a color
        allEnemyKingDangers = enemyKingDangers;

        slidingDangerSquares = (allEnemyBishopDangers | allEnemyRookDangers | allEnemyQueenDangers);
        allDangerSquares = (slidingDangerSquares | allEnemyKnightDangers | allEnemyPawnDangers | allEnemyKingDangers);

        //Note the use of [0] at the end of the next three lines -> because the functions return vectors
        big intermediate[2];
        pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, 0, intermediate);
        kingAsBishopAttacks = intermediate[0];
        pseudoLegalRookMoves(friendlyPieces[KING], allPieces, 0, intermediate);
        kingAsRookAttacks = intermediate[0];
        pseudoLegalQueenMoves(friendlyPieces[KING], allPieces, 0, intermediate);
        kingAsQueenAttacks = intermediate[0];
        kingAsSlidingPieceAttacks = (kingAsBishopAttacks | kingAsRookAttacks | kingAsQueenAttacks);
        //pinnedPiecesMasks = ((slidingDangerSquares & kingAsSlidingPieceAttacks) & allPieces);*/
    }

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    void kingInCheck(const GameState& state, bool& inCheck, big& otherPieceMoveMask, big& otherPieceCaptureMask, Move* moves, int& nbMoves){
        //Similuate the king being all types of pieces to find the number of checkers
        big kingAsBishop;
        big kingAsRook;
        //big kingAsQueen[2];
        //big kingAsKnight[2];
        big kingAsPawn;
        int kingPos = __builtin_ctzll(friendlyPieces[KING]);
        kingAsBishop = moves_table(kingPos, allPieces&mask_empty_bishop(kingPos));
        kingAsRook = moves_table(kingPos+64, allPieces&mask_empty_rook(kingPos));
        kingAsPawn = attackPawns[64*state.friendlyColor()+kingPos];
        //pseudoLegalPawnMoves(friendlyPieces[KING], allPieces , allEnemyPieces, state.friendlyColor(), -1, -1, -1, kingAsPawn);
        big checkDetection[5] = {kingAsPawn & enemyPieces[PAWN],
                                 KnightMoves[kingPos] & enemyPieces[KNIGHT],
                                 kingAsBishop & enemyPieces[BISHOP],
                                 kingAsRook & enemyPieces[ROOK],
                                 (kingAsRook|kingAsBishop) & enemyPieces[QUEEN],
                                 };

        int nbCheckers = 0;
        int checkerPosition = -1;
        //Id given the consts in the project if they change from 0 to 4 then problem
        int checkerId = -1;
        big kingAsChecker = 0;
        big checkerAttacks = 0;
        big intermediate[2];

        for (int i = 0; i < 5; ++i){
            big possibleChecker = checkDetection[i];
            nbCheckers += countbit(possibleChecker);
            if (countbit(possibleChecker) == 1){
                checkerPosition = __builtin_ctzll(possibleChecker);
                checkerId = i;
            }
        }

        inCheck = (nbCheckers!=0);

        otherPieceMoveMask = -1;
        otherPieceCaptureMask = -1;

        //On peut bouger seulement le roi
        if (nbCheckers>=2){
            otherPieceMoveMask = 0;
            otherPieceCaptureMask = 0;
        }

        //Capture the checker 
        //TODO : (get in the ray of possible blocks if it exists)
        else if (nbCheckers==1){
            //For now no ray only taking the checker is allowed
            otherPieceMoveMask = 0;
            if (checkerId > 1){
                otherPieceMoveMask = directions[kingPos][checkerPosition];//&~(1ULL << checkerPosition);
                //There is olny one piece checking the king so only one type of ray is checking him
                //Only look at bishop rays from checking position
                /*pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, intermediate);
                kingAsChecker = intermediate[0];
                pseudoLegalBishopMoves(1ul<<checkerPosition,allPieces, allEnemyPieces, intermediate);
                checkerAttacks = intermediate[0];

                //A bishop ray is not checking the king so it must be a rook ray
                if ((checkerAttacks & friendlyPieces[KING]) == 0){
                    //Only look at rook rays from checking position
                    pseudoLegalRookMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, intermediate);
                    kingAsChecker = intermediate[0];
                    pseudoLegalRookMoves(1ul<<checkerPosition,allPieces, allEnemyPieces, intermediate);
                    checkerAttacks = intermediate[0];
                }
                otherPieceMoveMask = (kingAsChecker & checkerAttacks);*/
            }
            //otherPieceCaptureMask = 0;
            //Need to get the checker's positions
            otherPieceCaptureMask = (1ul<<checkerPosition);
        }

        pinnedPiecesMasks = pinned(__builtin_ctzll(friendlyPieces[KING]), inCheck, moves, nbMoves, state.friendlyColor());
        //return {otherPieceMoveMask,otherPieceCaptureMask};
    }

    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves, bool inCheck){
        big kingMask = friendlyPieces[KING];
        int kingPos = __builtin_ctzll(kingMask);
        bool curColor = state.friendlyColor();
        big kingEndMask = pseudoLegalKingMoves(kingMask,allFriendlyPieces, state.friendlyColor(), state.castlingRights[curColor][1], state.castlingRights[curColor][0], inCheck);

        //kingEndMask &= (~allDangerSquares);

        maskToMoves<KING>(kingPos,kingEndMask, moves, nbMoves);
    }

    void legalPawnMoves(const GameState& state, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves){
        big pawnMask = friendlyPieces[PAWN] & (~pinnedPiecesMasks);
        big pawnMasks[9];
        int nbPawns = pseudoLegalPawnMoves(pawnMask,allPieces,allEnemyPieces,state.friendlyColor(),moveMask,captureMask, state.lastDoublePawnPush, pawnMasks);
        ubyte pos[8]; //max number of friendly pawn
        int nbPos=places(pawnMask, pos);
        for (int p = 0;p<nbPos;++p){
            big pawnMoveMask = pawnMasks[p];

            maskToMoves<PAWN>(pos[p], pawnMoveMask, pawnMoves, nbMoves);
            //pawnMoves.insert(pawnMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
        }
    }

    void legalKnightMoves(const GameState& state, big moveMask, big captureMask, Move* knightMoves, int& nbMoves){
        big knightMask = friendlyPieces[KNIGHT] & (~pinnedPiecesMasks);
        big knightMasks[11];
        int nbKnight = pseudoLegalKnightMoves(knightMask,allFriendlyPieces, knightMasks);
        ubyte pos[10]; //max number of friendly knight
        int nbPos=places(knightMask, pos);
        for (int p = 0;p<nbPos;++p){
            big knightEndMask = knightMasks[p] & (moveMask | captureMask);

            maskToMoves<KNIGHT>(pos[p], knightEndMask, knightMoves, nbMoves);
            //knightMoves.insert(knightMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
        }
    }

    void legalSlidingMoves(const GameState& state, big moveMask, big captureMask, Move* slidingMoves, int& nbMoves){

        int types[3] = {BISHOP,ROOK,QUEEN};
        big typeMasks[11];
        for (int pieceType : types){
            //Change function
            //vector<big> typeMasks;
            big typeMask = friendlyPieces[pieceType] & (~pinnedPiecesMasks);
            if (pieceType==BISHOP){
                pseudoLegalBishopMoves(typeMask,allPieces,allFriendlyPieces, typeMasks);
            }
            else if (pieceType == ROOK){
                pseudoLegalRookMoves(typeMask,allPieces,allFriendlyPieces, typeMasks);
            }
            else if (pieceType ==QUEEN){
                pseudoLegalQueenMoves(typeMask,allPieces,allFriendlyPieces, typeMasks);
            }
            ubyte pos[10]; //max number of the friendly piece
            int nbPos=places(typeMask, pos);
            for (int p = 0;p<nbPos;++p){
                big typeEndMask = typeMasks[p] & (moveMask | captureMask);

                //vector<Move> intermediateMoves;
                if (pieceType == BISHOP){
                    maskToMoves<BISHOP>(pos[p], typeEndMask, slidingMoves, nbMoves);
                }
                else if (pieceType == ROOK){
                    maskToMoves<ROOK>(pos[p], typeEndMask, slidingMoves, nbMoves);
                }
                else if (pieceType == QUEEN){
                    maskToMoves<QUEEN>(pos[p], typeEndMask, slidingMoves, nbMoves);
                }
                //slidingMoves.insert(slidingMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
            }
        }
    }

    //Returns all legal moves for a position (still missing pins; en-passant; castling; promotion)
    public : int generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves){
        recalculateAllMasks(state);
        //All allowed spots for a piece to move (not allowed if king is in check)
        big moveMask = -1; //Totaly true
        //All allowed spots for a piece to capture another one (not allowed if there is a checker)
        big captureMask = -1; //Totaly true

        int nbMoves = 0;
        kingInCheck(state, inCheck, moveMask, captureMask, legalMoves, nbMoves);
        //moveMask = currentMasks[0];
        //captureMask = currentMasks[1];
        legalKingMoves(state, legalMoves, nbMoves, inCheck);
        legalPawnMoves(state,moveMask,captureMask, legalMoves, nbMoves);
        legalKnightMoves(state,moveMask,captureMask, legalMoves, nbMoves);
        legalSlidingMoves(state,moveMask,captureMask, legalMoves, nbMoves);

        /*legalMoves.insert(legalMoves.end(),legalKingMoves.begin(),legalKingMoves.end());
        legalMoves.insert(legalMoves.end(),legalPawnMoves.begin(),legalPawnMoves.end());
        legalMoves.insert(legalMoves.end(),legalKnightMoves.begin(),legalKnightMoves.end());
        legalMoves.insert(legalMoves.end(),legalSlidingMoves.begin(),legalSlidingMoves.end());*/
        return nbMoves;
    }
};

#endif