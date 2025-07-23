#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Functions.hpp"
#include "GameState.hpp"
#include <fstream>
#include <utility>
#include <vector>
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

    inline bool isAttacking(big maskPiece, int dir){
        if(maskPiece & enemyPieces[QUEEN])return true;
        if(dir >= 4)dir -= 3;
        if(dir%2)return (maskPiece&enemyPieces[ROOK]) != 0;
        else return (maskPiece&enemyPieces[BISHOP]) != 0;
    }
    inline int firstPiece(big mask, int dir){
        if(dir < 4)
            return 63-__builtin_clzll(mask);
        return __builtin_ctzll(mask);
    }
    big pinned(int square){
        big maskPinned=0;
        for(int idDir = 0; idDir<8; idDir++){
            big mask = fullDir[square][idDir]&allPieces;
            if(!mask)continue;
            big maskFirst=1ULL << firstPiece(mask, idDir);
            if(maskFirst & allEnemyPieces)continue; //may be a checker if needed
            big newMask = mask&(~maskFirst);
            if(!newMask)continue;
            big maskSecond = 1ULL << firstPiece(newMask, idDir);
            if(isAttacking(maskSecond, idDir))
                maskPinned |= maskFirst;
        }
        return maskPinned;
    }

public:
    LegalMoveGenerator(){
        PrecomputeKnightMoveData();
        load_table("magics.out");
        init_lines();
        precomputeDirections();
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
            moves[nbMoves] = {(int8_t)start, (int8_t)bit, piece};
            nbMoves++;
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

    int pseudoLegalPawnMoves(big positions, big allPieces, big enemyPieces, bool color, big moveMask, big captureMask, big* allMasks){
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
        for (int p = 0;p<nbPos;++p){
            big pawnMoveMask = 0;
            big pawnCaptureMask = 0;
            int pieceRow = row(pos[p]);
            int pieceCol = col(pos[p]);
            //Single pawn push (check there are no pieces on target square)
            pawnMoveMask |= ((1ul<<(pos[p] + 8 * moveFactor)) & (~allPieces));

            //Double pawn push
            if (((pieceRow==1 && color == 0) || (pieceRow==6 && color == 1)) && (pawnMoveMask!=0)){
                pawnMoveMask |= ((1ul<<(pos[p] + 16 * moveFactor)) & (~allPieces));
            }

            //Capture left
            if(pieceCol != leftLimit)
                pawnCaptureMask |= ((1ul<<(pos[p] + 7 * moveFactor)) & (enemyPieces));

            //Capture right
            if(pieceCol != rightLimit)
                pawnCaptureMask |= ((1ul<<(pos[p] + 9 * moveFactor)) & (enemyPieces));

            //TODO : capture en-passant

            allMasks[p] = (pawnMoveMask & moveMask) | (pawnCaptureMask & captureMask);
            allMasks[nbPos] |= allMasks[p];
        }
        //return allMasks;
        return nbPos;
    }  
    
    big pseudoLegalKingMoves(big positions, big friendlyPieces){
        big kingMask = positions;
        int kingPos = __builtin_ctzll(kingMask);

        big kingEndMask = 0;
        //UP, DOWN, LEFT, RIGHT
        int transitions[4] = {8, -8, -1, 1};
        if (row(kingPos)==7){
            transitions[0] = 0;
        }
        if (row(kingPos)==0){
            transitions[1] = 0;
        }
        if (col(kingPos)==0){
            transitions[2] = 0;
        }
        if (col(kingPos)==7){
            transitions[3] = 0;
        }
        for (int i = 0; i < 4;++i){
            if(transitions[i] == 0)continue;
            int trans1 = transitions[i];
            int cardEnd = kingPos + trans1;
            kingEndMask |= (1ul<< cardEnd);
            for (int j = i+1; j < 4;++j){
                if(transitions[j] == 0)continue;
                int trans2 = transitions[j];
                int diagEnd = kingPos + trans1 + trans2;
                kingEndMask |= (1ul<< diagEnd);
            }
        }
        return kingEndMask & (~friendlyPieces);

    }

    const big* enemyPieces;
    big allEnemyPieces = 0;
    const big* friendlyPieces;
    big allFriendlyPieces = 0;
    big allPieces = 0;
    //Squares that are being targeted by the enemy (ignoring the friendly king)
    big allDangerSquares = 0;
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
    big kingAsSlidingPieceAttacks;

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
        big enemyBishopDangers[11];
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
        big enemyKingDangers = pseudoLegalKingMoves(enemyPieces[KING],0);

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
        pinnedPiecesMasks = pinned(__builtin_ctzll(friendlyPieces[KING]));
        //pinnedPiecesMasks = ((slidingDangerSquares & kingAsSlidingPieceAttacks) & allPieces);
    }

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    void kingInCheck(const GameState& state, bool& inCheck, big& otherPieceMoveMask, big& otherPieceCaptureMask){
        //Similuate the king being all types of pieces to find the number of checkers
        big kingAsBishop[2];
        big kingAsRook[2];
        big kingAsQueen[2];
        big kingAsKnight[2];
        big kingAsPawn[2];
        pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, kingAsBishop);
        pseudoLegalRookMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, kingAsRook);
        pseudoLegalQueenMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, kingAsQueen);
        pseudoLegalKnightMoves(friendlyPieces[KING], allFriendlyPieces, kingAsKnight);
        pseudoLegalPawnMoves(friendlyPieces[KING], allPieces , allEnemyPieces, state.friendlyColor(), -1, -1, kingAsPawn);

        big checkDetection[5] = {kingAsPawn[1] & enemyPieces[PAWN],
                                 kingAsKnight[1] & enemyPieces[KNIGHT],
                                 kingAsBishop[1] & enemyPieces[BISHOP],
                                 kingAsRook[1] & enemyPieces[ROOK],
                                 kingAsQueen[1] & enemyPieces[QUEEN],
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
            if (checkerId != 0 && checkerId != 1){
                //There is olny one piece checking the king so only one type of ray is checking him
                //Only look at bishop rays from checking position
                pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, allFriendlyPieces, intermediate);
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
                otherPieceMoveMask = (kingAsChecker & checkerAttacks);
            }
            otherPieceCaptureMask = 0;
            //Need to get the checker's positions
            otherPieceCaptureMask |= (1ul<<checkerPosition);
        }

        //return {otherPieceMoveMask,otherPieceCaptureMask};
    }

    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves){
        big kingMask = friendlyPieces[KING];
        int kingPos = __builtin_ctzll(kingMask);

        big kingEndMask = pseudoLegalKingMoves(kingMask,allFriendlyPieces);

        kingEndMask &= (~allDangerSquares);

        maskToMoves<KING>(kingPos,kingEndMask, moves, nbMoves);
    }

    void legalPawnMoves(const GameState& state, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves){
        big pawnMask = friendlyPieces[PAWN] & (~pinnedPiecesMasks);
        big pawnMasks[9];
        int nbPawns = pseudoLegalPawnMoves(pawnMask,allPieces,allEnemyPieces,state.friendlyColor(),moveMask,captureMask, pawnMasks);
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

        kingInCheck(state, inCheck, moveMask, captureMask);
        //moveMask = currentMasks[0];
        //captureMask = currentMasks[1];
        int nbMoves = 0;
        legalKingMoves(state, legalMoves, nbMoves);
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