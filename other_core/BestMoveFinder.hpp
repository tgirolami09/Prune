#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include "TranspositionTable.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include "LegalMoveGenerator.hpp"
#include <cmath>
#include <chrono>
#include <atomic>
#include <thread>
//Class to find the best in a situation
class BestMoveFinder{
    //Returns the best move given a position and time to use
public:
    LegalMoveGenerator generator;
    Evaluator eval;
    transpositionTable transposition;
private:
    std::atomic<bool> running;
public:
    BestMoveFinder(int memory):transposition(memory/sizeof(infoScore)){}
private:
    int alloted_time;
    void stopAfter(int seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        running = false; // Set running to false after the specified time
    }
    int negamax(int deep, GameState& state, int alpha, int beta){
        if(!running)return 0;
        if(deep == 0)
            return eval.positionEvaluator(state);
        bool isEvaluated=false;
        Move bMove;
        int last_eval=transposition.get_eval(state, alpha, beta, isEvaluated, deep, bMove);
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
        Move bestMove;
        if(bMove.start_pos != bMove.end_pos){
            state.playMove<false>(bMove);
            int score = -negamax(deep-1, state, -beta, -alpha);
            if(score > alpha){
                if(score > beta){
                    transposition.push(state, {score, beta, alpha, bMove});
                    return score;
                }
                alpha = score;
            }
            max_eval = score;
            bestMove = bMove;
        }
        for(Move move:moves){
            if(move.start_pos == bMove.start_pos && move.end_pos == bMove.end_pos)continue;
            state.playMove<false>(move);
            int score = -negamax(deep-1, state, -beta, -alpha);
            state.undoLastMove();
            if(!running)return 0;
            if(score > alpha){
                if(score > beta){
                    transposition.push(state, {score, beta, alpha, move, deep});
                    return score;
                }
                alpha = score;
            }
            if(score > max_eval){
                max_eval = score;
                bestMove = move;
            }
        }
        transposition.push(state, {max_eval, alpha, beta, bestMove, deep});
        return max_eval;
    }
    public : Move bestMove(GameState& state, int alloted_time){
        //Calls evaluator here to determine what to look at
        Move bestMove;
        running = true;
        this->alloted_time = alloted_time;
        Move lastBest;
        std::thread timerThread(&BestMoveFinder::stopAfter, this);
        for(int depth=1; running; depth++){
            int alpha=eval.MINIMUM;
            int beta=eval.MAXIMUM;
            bool inCheck;
            vector<Move> moves=generator.generateLegalMoves(state, inCheck);
            if(moves.size() == 0)
                return {}; // no possible moves
            for(Move move:moves){
                state.playMove<false>(move);
                int score = -negamax(depth, state, -beta, -alpha);
                state.undoLastMove();
                if(!running)break;
                if(score > alpha){
                    alpha = score;
                    bestMove = move;
                }
            }
            lastBest = bestMove;
            transposition.clear();
        }
        if(bestMove.start_pos == bestMove.end_pos)return lastBest;
        return bestMove;
    }
    big perft(GameState& state, int depth, int curDepth=0){
        if(depth == 0)return 1;
        bool inCheck;
        vector<Move> moves=generator.generateLegalMoves(state, inCheck);
        big count=0;
        for(Move move:moves){
            state.playMove<false>(move);
            big nbNodes=perft(state, depth-1, curDepth+1);
            state.undoLastMove();
            if(curDepth == 0){
                printf("%s: %lld", move.to_str().c_str(), nbNodes);
            }
            count += nbNodes;
        }
        return count;
    }
};
#endif