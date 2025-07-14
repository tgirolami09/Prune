#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include "Move.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include "LegalMoveGenerator.hpp"
#include <cmath>
//Class to find the best in a situation
class BestMoveFinder{
    //Returns the best move given a position and time to use
    float negamax(int deep, GameState state, float alpha, float beta){
        Evaluator eval;
        if(deep == 0)
            return eval.positionEvaluator(state);
        float max_eval=-INFINITY;
        LegalMoveGenerator generator;
        bool isCheck;
        vector<Move> moves=generator.generateLegalMoves(state, isCheck);
        if(moves.size() == 0){
            if(isCheck)
                return -INFINITY;
            return 0;
        }
        for(Move move:moves){
            state.playMove(move);
            float score = -negamax(deep-1, state, -beta, -alpha);
            state.undoLastMove();
            if(score > max_eval){
                max_eval = score;
                if(score > beta)
                    return score;
            }
            if(score > alpha)
                alpha = score;
        }
        return max_eval;
    }
    public : Move bestMove(GameState state, float alloted_time){
        //Calls evaluator here to determine what to look at
        Evaluator eval;
        LegalMoveGenerator generator;
        Move bestMove;
        float alpha=-INFINITY;
        float beta=INFINITY;
        bool inCheck;
        vector<Move> moves=generator.generateLegalMoves(state, inCheck);
        if(moves.size() == 0)
            return {}; // no possible moves
        for(Move move:moves){
            state.playMove(move);
            float score = -negamax(4, state, alpha, beta);
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