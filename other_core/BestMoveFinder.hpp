#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include "Compressor.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include "LegalMoveGenerator.hpp"
#include <cmath>
//Class to find the best in a situation
class BestMoveFinder{
    //Returns the best move given a position and time to use
    LegalMoveGenerator generator;
    Evaluator eval;
    transpositionTable transposition;
    int negamax(int deep, GameState state, int alpha, int beta){
        if(deep == 0)
            return eval.positionEvaluator(state);
        bool isEvaluated=false;
        int last_eval=transposition.get_eval(state, isEvaluated);
        if(isEvaluated){
            return last_eval;
        }
        int max_eval=eval.MINIMUM;
        bool isCheck;
        vector<Move> moves=generator.generateLegalMoves(state, isCheck);
        if(moves.size() == 0){
            if(isCheck)
                return eval.MINIMUM;
            return eval.MIDDLE;
        }
        for(Move move:moves){
            state.playMove(move);
            int score = -negamax(deep-1, state, -beta, -alpha);
            state.undoLastMove();
            if(score > alpha){
                if(score > beta){
                    transposition.push(state, {score, beta, alpha});
                    return score;
                }
                alpha = score;
            }
            if(score > max_eval)
                max_eval = score;
        }
        transposition.push(state, {max_eval, alpha, beta});
        return max_eval;
    }
    public : Move bestMove(GameState state, int alloted_time){
        //Calls evaluator here to determine what to look at
        Move bestMove;
        int alpha=eval.MINIMUM;
        int beta=eval.MAXIMUM;
        bool inCheck;
        vector<Move> moves=generator.generateLegalMoves(state, inCheck);
        if(moves.size() == 0)
            return {}; // no possible moves
        for(Move move:moves){
            state.playMove(move);
            int score = -negamax(4, state, alpha, beta);
            state.undoLastMove();
            if(score > alpha){
                alpha = score;
                bestMove = move;
            }
        }
        return bestMove;
    }
};
#endif