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
    
    //Transforms a bitboard of valid end positions into a list of the corresponding moves
    vector<Move> maskToMoves(int start, big mask){
        vector<Move> res;
        while(mask){
            int bit = ffsll(mask);
            mask ^= 1 << bit;
            //Need to add logic for pawn promotion
            res.push_back({start,bit,-1});
        }
        return res;
    }

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    vector<big> kingInCheck(GameState state){
        //Look at the squares the enemy attacks
        return {};
    }

    vector<Move> kingMoves(GameState state){
        return {};
    }

    vector<Move> pawnMoves(GameState state, big moveMask, big captureMask){
        return {};
    }

    //Does not take pins into account
    vector<Move> knightMoves(GameState state, big moveMask, big captureMask){
        if (!knightWasPrecomputed){
            PrecomputeKnightMoveData();
        }
        vector<Move> moves;
        big* allFriendlyPieces = state.friendlyPieces();
        big knightMask = allFriendlyPieces[KNIGHT/2];
        for (int pos = 0;pos<64;++pos){
            //Friendly knight at position 'pos'
            if (knightMask & (1ul<< pos)){
                //Get the end positions that are allowed given the state of the king 
                big knightEndMask = KnightMoves[pos] & (moveMask | captureMask);

                vector<Move> intermediateMoves = maskToMoves(pos,knightEndMask);
                moves.insert(moves.end(),intermediateMoves.begin(),intermediateMoves.end());
            }
        }
        return moves;
    }

    vector<Move> slidingMoves(GameState state, big moveMask, big captureMask){
        return {};
    }

    //Returns all legal moves for a position
    public : vector<Move> generateLegalMoves(GameState state){
        //All allowed spots for a piece to move (not allowed if king is in check)
        big moveMask = -1; //Totaly true
        //All allowed spots for a piece to capture another one (not allowed if there is a checker)
        big captureMask = -1; //Totaly true

        vector<big> currentMasks = kingInCheck(state);
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