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

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    vector<big> kingInCheck(const GameState& state, bool& inCheck){
        //Look at the squares the enemy attacks
        return {};
    }

    vector<Move> kingMoves(const GameState& state){
        return {};
    }

    vector<Move> pawnMoves(const GameState& state, big moveMask, big captureMask){
        return {};
    }

    //Does not take pins into account
    vector<Move> knightMoves(const GameState& state, big moveMask, big captureMask){
        vector<Move> moves;
        const big* allFriendlyPieces = state.friendlyPieces();
        big knightMask = allFriendlyPieces[KNIGHT];
        ubyte pos[12];
        int nbPos=places(knightMask, pos);
        for (int p = 0;p<nbPos;++p){
            //Friendly knight at position 'pos[p]'
            //Get the end positions that are allowed given the state of the king 
            big knightEndMask = KnightMoves[pos[p]] & (moveMask | captureMask);

            vector<Move> intermediateMoves = maskToMoves<KNIGHT>(pos[p], knightEndMask);
            moves.insert(moves.end(),intermediateMoves.begin(),intermediateMoves.end());
        }
        return moves;
    }

    vector<Move> slidingMoves(const GameState& state, big moveMask, big captureMask){
        return {};
    }

    //Returns all legal moves for a position
    public : vector<Move> generateLegalMoves(const GameState& state, bool& inCheck){
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