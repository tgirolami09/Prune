#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Functions.hpp"
#include "GameState.hpp"
using namespace std;


//Class to generate legal moves
class LegalMoveGenerator{
    int KnightOffSets[8] = {15, 17, -17, -15, 10, -6, 6, -10};

    big KnightMoves[64] = {0}; //Knight moves for each position of the board

    bool knightWasPrecomputed = false;

    void PrecomputeKnightMoveData(){
        knightWasPrecomputed = true;
        for (int file = 0; file<8; ++file){
            for (int rank = 0; rank<8; ++rank){
                int squareIndex = rank * 8 + file;

                //Precompute knight moves
                big knightMoveMask;

                for (int distVertical = 1; distVertical<=2; ++distVertical){
                    for (int directionVertical = 0; directionVertical<=1; ++directionVertical){
                        //0 is up and 1 is down

                        int VerticalSign = (directionVertical==0 ? 1 : -1);
                        int targetRank = row(squareIndex) + distVertical * VerticalSign; //This is 1-indexed
                        if (targetRank < 1 || 8 < targetRank){
                            continue; //Move is not possible
                        }

                        for (int directionHorizontal = 0; directionHorizontal<=1; ++directionHorizontal){
                            //0 is left and 1 is right
                            int distHorizontal = (distVertical==1 ? 2 : 1);
                            int HorizontalSign = (directionHorizontal==0 ? -1 : 1); //this is right 
                            int targetFile = col(squareIndex) + distHorizontal * HorizontalSign; //This is 1-indexed
                            if (targetFile < 1 || 8 < targetFile){
                                continue; //Move is not possible
                            }

                            int targetSquare = squareIndex;
                            targetSquare += ( (distVertical * 8 * VerticalSign) + (distHorizontal * HorizontalSign) );

                            knightMoveMask = addBitToMask(knightMoveMask,targetSquare);
                        }
                    }
                }
                KnightMoves[squareIndex] = knightMoveMask;
            }
        }  
    }
    
    public:LegalMoveGenerator(){
        PrecomputeKnightMoveData();
    }

private: 
    //Transforms a bitboard of valid end positions into a list of the corresponding moves
    template<int piece>
    vector<Move> maskToMoves(int start, big mask){
        vector<Move> res;
        while(mask){
            int bit = __builtin_ctzll(mask);
            mask &= mask-1;
            //Need to add logic for pawn promotion
            res.push_back({(int8_t)start,(int8_t)bit,piece});
        }
        return res;
    }

    vector<big> pseudoLegalBishopMoves(big positions, big allPieces, big friendlyPieces){
        big bishopMask = positions;
        ubyte* pos;
        int nbPos=places(bishopMask, pos);
        //Need to rewrite each time
        vector<big> allMasks(nbPos+1,0);
        //1 more with all possible moves from piece type
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            if (bishopMask & (1ul<< pos[p])){
                //La logique de recuperation des coups
                big bishopMoveMask;
                allMasks[p] = bishopMoveMask;
                allMasks[nbPos] |= bishopMoveMask;
            }
        }
        return allMasks;
    }

    vector<big> pseudoLegalRookMoves(big positions, big allPieces, big friendlyPieces){
        big rookMask = positions;
        ubyte* pos;
        int nbPos=places(rookMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            if (rookMask & (1ul<< pos[p])){
                //La logique de recuperation des coups
                big rookMoveMask;
                allMasks[p] = rookMoveMask;
                allMasks[nbPos] |= rookMoveMask;
            }
        }
        return allMasks;
    }

    vector<big> pseudoLegalQueenMoves(big positions, big allPieces, big friendlyPieces){
        big queenMask = positions;
        ubyte* pos;
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
        if (!knightWasPrecomputed){
            PrecomputeKnightMoveData();
        }
        big knightMask = positions;
        ubyte* pos;
        int nbPos=places(knightMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
        for (int p = 0;p<nbPos;++p){
            if (knightMask & (1ul<< pos[p])){
                big knightEndMask = KnightMoves[pos[p]] & (~friendlyPieces);
                allMasks[p] = knightEndMask;
                allMasks[nbPos] |= knightEndMask;
            }
        }
        return allMasks;
    }

    vector<big> pseudoLegalPawnMoves(big positions, big allPieces, big enemyPieces, bool color, big moveMask = -1, big captureMask = -1){
        //If color == 0 -> white (add to move)
        //If color == 1 -> black (subtract to move)
        big pawnMask = positions;
        ubyte* pos;
        int nbPos=places(pawnMask, pos);
        vector<big> allMasks(nbPos+1,0);
        // allMasks[nbPos] = 0;
        //Should be if color is 1 (true) then moveFactor is -1
        int moveFactor = color ? -1 : 1;
        for (int p = 0;p<nbPos;++p){
            if (pawnMask & (1ul<< pos[p])){
                big pawnMoveMask = 0;
                big pawnCaptureMask = 0;
                int pieceRow = row(pos[p]);
                //Single pawn push (check there are no pieces on target square)
                pawnMoveMask |= ((1ul<<(pos[p] + 8 * moveFactor)) & (~allPieces));

                //Double pawn push
                if ((pieceRow==1 && color == 0) || (pieceRow==6 && color == 1) && pawnMoveMask!=0){
                    pawnMoveMask |= ((1ul<<(pos[p] + 16 * moveFactor)) & (~allPieces));
                }

                //Capture left
                pawnCaptureMask |= ((1ul<<(pos[p] + 7 * moveFactor)) & (enemyPieces));

                //Capture right
                pawnCaptureMask |= ((1ul<<(pos[p] + 9 * moveFactor)) & (enemyPieces));

                //TODO : capture en-passant

                allMasks[p] = (pawnMoveMask & moveMask) | (pawnCaptureMask & captureMask);
                allMasks[nbPos] |= allMasks[p];
            }
        }
        return allMasks;
        
    }  
    
    big pseudoLegalKingMoves(big positions, big friendlyPieces){
        big kingMask = positions;
        int kingPos = ffsll(kingMask)-1;

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
        return kingEndMask | (~friendlyPieces);

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
        vector<big> enemyPawnDangers = pseudoLegalPawnMoves(enemyPieces[PAWN],allPieces,allPieces,state.enemyColor(),0);
        big enemyKingDangers = pseudoLegalKingMoves(enemyPieces[KING],allEnemyPieces);

        allEnemyBishopDangers = enemyBishopDangers[countbit(enemyPieces[BISHOP])];
        allEnemyRookDangers = enemyRookDangers[countbit(enemyPieces[ROOK])];
        allEnemyQueenDangers = enemyQueenDangers[countbit(enemyPieces[QUEEN])];
        allEnemyKnightDangers = enemyKnightDangers[countbit(enemyPieces[KNIGHT])];
        allEnemyPawnDangers = enemyPawnDangers[countbit(enemyPieces[PAWN])];
        //There should only be one king of a color
        allEnemyKingDangers = enemyKingDangers;

        dangerSquares = (allEnemyBishopDangers | allEnemyRookDangers | allEnemyQueenDangers | allEnemyKnightDangers | allEnemyPawnDangers | allEnemyKingDangers);
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
                checkerPosition = ffsll(possibleChecker)-1;
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

    vector<Move> kingMoves(const GameState& state){
        big kingMask = friendlyPieces[KING];
        int kingPos = ffsll(kingMask)-1;

        big kingEndMask = pseudoLegalKingMoves(kingMask,allFriendlyPieces);

        kingEndMask |= (~dangerSquares);

        return maskToMoves<KING>(kingPos,kingEndMask);
    }

    vector<Move> pawnMoves(const GameState& state, big moveMask, big captureMask){
        vector<big> pawnMasks = pseudoLegalPawnMoves(friendlyPieces[PAWN],allPieces,allEnemyPieces,state.friendlyColor(),moveMask,captureMask);
        vector<Move> pawnMoves;
        big pawnMask = friendlyPieces[PAWN];
        ubyte* pos;
        int nbPos=places(pawnMask, pos);
        for (int p = 0;p<nbPos;++p){
            if (pawnMask & (1ul<< pos[p])){
                
                big pawnMoveMask = pawnMasks[p];

                vector<Move> intermediateMoves = maskToMoves<PAWN>(pos[p], pawnMask);
                pawnMoves.insert(pawnMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
            }
        }
        return pawnMoves;
    }

    vector<Move> knightMoves(const GameState& state, big moveMask, big captureMask){

        vector<big> knightMasks = pseudoLegalKnightMoves(friendlyPieces[KNIGHT],allFriendlyPieces);
        vector<Move> knightMoves;
        big knightMask = friendlyPieces[KNIGHT];
        ubyte pos[12];
        int nbPos=places(knightMask, pos);
        for (int p = 0;p<nbPos;++p){
            if (knightMask & (1ul<< pos[p])){
                big knightEndMask = knightMasks[p] & (moveMask | captureMask);

            vector<Move> intermediateMoves = maskToMoves<KNIGHT>(pos[p], knightEndMask);
            knightMoves.insert(knightMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
            }
        }
        return knightMoves;
    }

    vector<Move> slidingMoves(const GameState& state, big moveMask, big captureMask){
        vector<Move> slidingMoves;

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
            big typeMask = friendlyPieces[pieceType];
            ubyte* pos;
            int nbPos=places(typeMask, pos);
            for (int p = 0;p<nbPos;++p){
                if (typeMask & (1ul<< pos[p])){
                    big typeEndMask = typeMasks[p] & (moveMask | captureMask);

                    vector<Move> intermediateMoves;
                    if (pieceType == BISHOP){
                        intermediateMoves = maskToMoves<BISHOP>(pos[p], typeEndMask);
                    }
                    else if (pieceType == ROOK){
                        intermediateMoves = maskToMoves<ROOK>(pos[p], typeEndMask);
                    }
                    else if (pieceType == QUEEN){
                        intermediateMoves = maskToMoves<QUEEN>(pos[p], typeEndMask);
                    }
                    slidingMoves.insert(slidingMoves.end(),intermediateMoves.begin(),intermediateMoves.end());
                }
            }
        }
        
        return slidingMoves;
    }

    //Returns all legal moves for a position (still missing pins; en-passant; castling; promotion)
    public : vector<Move> generateLegalMoves(const GameState& state, bool& inCheck){
        recalculateAllMasks(state);
        //All allowed spots for a piece to move (not allowed if king is in check)
        big moveMask = -1; //Totaly true
        //All allowed spots for a piece to capture another one (not allowed if there is a checker)
        big captureMask = -1; //Totaly true

        vector<big> currentMasks = kingInCheck(state, inCheck);
        moveMask = currentMasks[0];
        captureMask = currentMasks[1];

        vector<Move> legalMoves;

        vector<Move> legalKingMoves = kingMoves(state);
        vector<Move> legalPawnMoves = pawnMoves(state,moveMask,captureMask);
        vector<Move> legalKnightMoves = knightMoves(state,moveMask,captureMask);
        vector<Move> legalSlidingMoves = slidingMoves(state,moveMask,captureMask);

        legalMoves.insert(legalMoves.end(),legalKingMoves.begin(),legalKingMoves.end());
        legalMoves.insert(legalMoves.end(),legalPawnMoves.begin(),legalPawnMoves.end());
        legalMoves.insert(legalMoves.end(),legalKnightMoves.begin(),legalKnightMoves.end());
        legalMoves.insert(legalMoves.end(),legalSlidingMoves.begin(),legalSlidingMoves.end());

        return legalMoves;
    }
};

#endif