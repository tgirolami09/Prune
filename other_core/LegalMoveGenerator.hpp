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

public:
    LegalMoveGenerator(){
        PrecomputeKnightMoveData();
        load_table("magics.out");
        init_lines();
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

    vector<big> pseudoLegalBishopMoves(big positions, big allPieces, big friendlyPieces){
        big bishopMask = positions;
        ubyte pos[10]; //max number of bishop
        int nbPos=places(bishopMask, pos);
        //Need to rewrite each time
        vector<big> allMasks(nbPos+1,0);
        //1 more with all possible moves from piece type
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            //La logique de recuperation des coups
            big bishopMoveMask=moves_table(pos[p], allPieces&mask_empty_bishop(pos[p]));
            bishopMoveMask &= ~friendlyPieces;
            allMasks[p] = bishopMoveMask;
            allMasks[nbPos] |= bishopMoveMask;
        }
        return allMasks;
    }

    vector<big> pseudoLegalRookMoves(big positions, big allPieces, big friendlyPieces){
        big rookMask = positions;
        ubyte pos[10]; //max number of rooks
        int nbPos=places(rookMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            //La logique de recuperation des coups
            big rookMoveMask=moves_table(pos[p]+64, allPieces&mask_empty_rook(pos[p]));
            rookMoveMask &= ~friendlyPieces;
            allMasks[p] = rookMoveMask;
            allMasks[nbPos] |= rookMoveMask;
        }
        return allMasks;
    }

    vector<big> pseudoLegalQueenMoves(big positions, big allPieces, big friendlyPieces){
        big queenMask = positions;
        ubyte pos[10]; //max number of queens
        int nbPos=places(queenMask, pos);
        vector<big> allMasks(nbPos+1,0);

        vector<big> bishopMoves = pseudoLegalBishopMoves(positions,allPieces,friendlyPieces);
        vector<big> rookMoves = pseudoLegalRookMoves(positions,allPieces,friendlyPieces);

        // +1 to get the extra bitboard (contains all possible moves of a piece type)
        for (int i = 0;i<nbPos+1;++i){
            allMasks[i] = bishopMoves[i] | rookMoves[i];
        }

        return allMasks;
    }

    vector<big> pseudoLegalKnightMoves(big positions, big friendlyPieces){
        big knightMask = positions;
        ubyte pos[10]; //max number of knight
        int nbPos=places(knightMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            big knightEndMask = KnightMoves[pos[p]] & (~friendlyPieces);
            allMasks[p] = knightEndMask;
            allMasks[nbPos] |= knightEndMask;
        }
        return allMasks;
    }

    vector<big> pseudoLegalPawnMoves(big positions, big allPieces, big enemyPieces, bool color, big moveMask = -1, big captureMask = -1){
        //If color == 0 -> white (add to move)
        //If color == 1 -> black (subtract to move)
        big pawnMask = positions;
        ubyte pos[8]; //max number of pawns
        int nbPos=places(pawnMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
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
        return allMasks;
        
    }  
    
    big pseudoLegalKingMoves(big positions, big friendlyPieces){
        big kingMask = positions;
        int kingPos = __builtin_ctzll(kingMask);

        big kingEndMask = 0;
        //UP, DOWN, LEFT, RIGHT
        vector<int> transitions;
        if (row(kingPos)!=7){
            transitions.push_back(8);
        }
        if (row(kingPos)!=0){
            transitions.push_back(-8);
        }
        if (col(kingPos)!=0){
            transitions.push_back(-1);
        }
        if (col(kingPos)!=7){
            transitions.push_back(+1);
        }
        for (int i = 0; i < transitions.size();++i){
            int trans1 = transitions[i];
            int cardEnd = kingPos + trans1;
            kingEndMask |= (1ul<< cardEnd);
            for (int j = i+1; j < transitions.size();++j){
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
    big dangerSquares = 0;
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

    void recalculateAllMasks(GameState state){
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
        vector<big> enemyBishopDangers = pseudoLegalBishopMoves(enemyPieces[BISHOP],allPieces ^ friendlyPieces[KING],0);
        vector<big> enemyRookDangers = pseudoLegalRookMoves(enemyPieces[ROOK],allPieces ^ friendlyPieces[KING],0);
        vector<big> enemyQueenDangers = pseudoLegalQueenMoves(enemyPieces[QUEEN],allPieces ^ friendlyPieces[KING],0);

        vector<big> enemyKnightDangers = pseudoLegalKnightMoves(enemyPieces[KNIGHT],0);
        //Set the move mask to 0 because we are only interested in dangers (captures or protections)
        //All pieces are enemies because we are only interested in dangers (captures or protections)
        // -1 instead of enemy pieces because we want all possible attacks even if not possible this turn
        vector<big> enemyPawnDangers = pseudoLegalPawnMoves(enemyPieces[PAWN],allPieces,-1,state.enemyColor(),0,-1);
        big enemyKingDangers = pseudoLegalKingMoves(enemyPieces[KING],0);

        allEnemyBishopDangers = enemyBishopDangers[countbit(enemyPieces[BISHOP])];
        allEnemyRookDangers = enemyRookDangers[countbit(enemyPieces[ROOK])];
        allEnemyQueenDangers = enemyQueenDangers[countbit(enemyPieces[QUEEN])];
        allEnemyKnightDangers = enemyKnightDangers[countbit(enemyPieces[KNIGHT])];
        allEnemyPawnDangers = enemyPawnDangers[countbit(enemyPieces[PAWN])];
        //There should only be one king of a color
        allEnemyKingDangers = enemyKingDangers;

        dangerSquares = (allEnemyBishopDangers | allEnemyRookDangers | allEnemyQueenDangers | allEnemyKnightDangers | allEnemyPawnDangers | allEnemyKingDangers);
    
        //Note the use of [0] at the end of the next three lines -> because the functions return vectors
        kingAsBishopAttacks = pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, 0)[0];
        kingAsRookAttacks = pseudoLegalRookMoves(friendlyPieces[KING], allPieces, 0)[0];
        kingAsQueenAttacks = pseudoLegalQueenMoves(friendlyPieces[KING], allPieces, 0)[0];
        kingAsSlidingPieceAttacks = (kingAsBishopAttacks | kingAsRookAttacks | kingAsQueenAttacks);

        pinnedPiecesMasks = ((dangerSquares & kingAsSlidingPieceAttacks) & allPieces);
    }

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    vector<big> kingInCheck(const GameState& state, bool& inCheck){
        //Similuate the king being all types of pieces to find the number of checkers
        vector<big> kingAsBishop = pseudoLegalBishopMoves(friendlyPieces[KING], allPieces, allFriendlyPieces);
        vector<big> kingAsRook = pseudoLegalRookMoves(friendlyPieces[KING], allPieces, allFriendlyPieces);
        vector<big> kingAsQueen = pseudoLegalQueenMoves(friendlyPieces[KING], allPieces, allFriendlyPieces);
        vector<big> kingAsKnight = pseudoLegalKnightMoves(friendlyPieces[KING], allFriendlyPieces);
        vector<big> kingAsPawn = pseudoLegalPawnMoves(friendlyPieces[KING], allPieces , allEnemyPieces, state.friendlyColor());

        big checkDetection[5] = {kingAsBishop[1] & enemyPieces[BISHOP],
                                 kingAsRook[1] & enemyPieces[ROOK],
                                 kingAsQueen[1] & enemyPieces[QUEEN],
                                 kingAsKnight[1] & enemyPieces[KNIGHT],
                                 kingAsPawn[1] & enemyPieces[PAWN]};

        int nbCheckers = 0;
        int checkerPosition = -1;

        for (int i = 0; i < 5; ++i){
            big possibleChecker = checkDetection[i];
            nbCheckers += countbit(possibleChecker);
            if (countbit(possibleChecker) == 1){
                checkerPosition = __builtin_ctzll(possibleChecker);
            }
        }

        inCheck = (nbCheckers!=0);

        big otherPieceMoveMask = -1;
        big otherPieceCaptureMask = -1;

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
            otherPieceCaptureMask = 0;
            //Need to get the checker's positions
            otherPieceCaptureMask |= (1ul<<checkerPosition);
        }

        return {otherPieceMoveMask,otherPieceCaptureMask};
    }

    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves){
        big kingMask = friendlyPieces[KING];
        int kingPos = __builtin_ctzll(kingMask);

        big kingEndMask = pseudoLegalKingMoves(kingMask,allFriendlyPieces);

        kingEndMask &= (~dangerSquares);

        maskToMoves<KING>(kingPos,kingEndMask, moves, nbMoves);
    }

    void legalPawnMoves(const GameState& state, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves){
        vector<big> pawnMasks = pseudoLegalPawnMoves(friendlyPieces[PAWN],allPieces,allEnemyPieces,state.friendlyColor(),moveMask,captureMask);
        // vector<Move> pawnMoves;
        big pawnMask = friendlyPieces[PAWN] & (~pinnedPiecesMasks);
        ubyte pos[8]; //max number of friendly pawn
        int nbPos=places(pawnMask, pos);
        for (int p = 0;p<nbPos;++p){
            big pawnMoveMask = pawnMasks[p];

            maskToMoves<PAWN>(pos[p], pawnMoveMask, pawnMoves, nbMoves);
            //pawnMoves.insert(pawnMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
        }
    }

    void legalKnightMoves(const GameState& state, big moveMask, big captureMask, Move* knightMoves, int& nbMoves){
        vector<big> knightMasks = pseudoLegalKnightMoves(friendlyPieces[KNIGHT],allFriendlyPieces);
        // vector<Move> knightMoves;
        big knightMask = friendlyPieces[KNIGHT] & (~pinnedPiecesMasks);
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

        for (int pieceType : types){
            //Change function
            vector<big> typeMasks;
            if (pieceType==BISHOP){
                typeMasks = pseudoLegalBishopMoves(friendlyPieces[pieceType],allPieces,allFriendlyPieces);
            }
            else if (pieceType == ROOK){
                typeMasks = pseudoLegalRookMoves(friendlyPieces[pieceType],allPieces,allFriendlyPieces);
            }
            else if (pieceType ==QUEEN){
                typeMasks = pseudoLegalQueenMoves(friendlyPieces[pieceType],allPieces,allFriendlyPieces);
            }
            big typeMask = friendlyPieces[pieceType] & (~pinnedPiecesMasks);
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

        vector<big> currentMasks = kingInCheck(state, inCheck);
        moveMask = currentMasks[0];
        captureMask = currentMasks[1];
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