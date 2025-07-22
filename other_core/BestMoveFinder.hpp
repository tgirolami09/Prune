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
#include <algorithm>

bool compScoreMove(const pair<int, Move>& a, const pair<int, Move>& b){
    return a.first > b.first;
}
const int maxMoves=218;
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
    int alloted_time;
    void stopAfter() {
        std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        running = false; // Set running to false after the specified time
    }
    
private:

    int negamax(int depth, GameState& state, int alpha, int beta){
        if(!running)return 0;
        if(depth == 0)
            return eval.positionEvaluator(state);
        bool isEvaluated=false;
        Move bMove;
        int last_eval=transposition.get_eval(state, alpha, beta, isEvaluated, depth, bMove);
        if(isEvaluated){
            return last_eval;
        }
        int max_eval=eval.MINIMUM;
        bool isCheck;
        Move moves[maxMoves];
        int nbMoves=generator.generateLegalMoves(state, isCheck, moves);
        if(nbMoves == 0){
            if(isCheck)
                return eval.MINIMUM;
            return eval.MIDDLE;
        }
        Move bestMove;
        if(bMove.start_pos != bMove.end_pos){
            state.playMove<false>(bMove);
            int score = -negamax(depth-1, state, -beta, -alpha);
            state.undoLastMove();
            if(!running)return 0;
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
        vector<pair<int, Move>> sortedMoves(nbMoves);
        for(int i=0; i<nbMoves; i++){
            sortedMoves[i] = {eval.score_move(moves[i]), moves[i]};
        }
        sort(sortedMoves.begin(), sortedMoves.end(), compScoreMove);
        for(pair<int, Move> move_score:sortedMoves){
            Move move=move_score.second;
            if(move.start_pos == bMove.start_pos && move.end_pos == bMove.end_pos)continue;
            state.playMove<false>(move);
            int score = -negamax(depth-1, state, -beta, -alpha);
            state.undoLastMove();
            if(!running)return 0;
            if(score > alpha){
                if(score > beta){
                    transposition.push(state, {score, beta, alpha, move, depth});
                    return score;
                }
                alpha = score;
            }
            if(score > max_eval){
                max_eval = score;
                bestMove = move;
            }
        }
        transposition.push(state, {max_eval, alpha, beta, bestMove, depth});
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
            Move moves[maxMoves];
            int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
            if(nbMoves == 0)
                return {}; // no possible moves
            for(int i=0; i<nbMoves; i++){
                state.playMove<false>(moves[i]);
                int score = -negamax(depth, state, -beta, -alpha);
                state.undoLastMove();
                if(!running)break;
                if(score > alpha){
                    alpha = score;
                    bestMove = moves[i];
                }
            }
            lastBest = bestMove;
            transposition.clear();
        }
        if(bestMove.start_pos == bestMove.end_pos)return lastBest;
        return bestMove;
    }

};


class Perft{
public:
    TTperft tt;
    LegalMoveGenerator generator;
    Perft(size_t space):tt(space){}
    int visitedNodes;
    int _perft(GameState& state, ubyte depth){
        visitedNodes++;
        if(depth == 0)return 1;
        int lastCall=tt.get_eval(state.zobristHash, depth);
        if(lastCall != -1)return lastCall;
        bool inCheck;
        Move moves[maxMoves];
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
        int count=0;
        for(int i=0; i<nbMoves; i++){
            state.playMove<false>(moves[i]);
            big nbNodes=_perft(state, depth-1);
            state.undoLastMove();
            count += nbNodes;
        }
        tt.push({state.zobristHash, count, depth});
        return count;
    }
    int perft(GameState& state, ubyte depth){
        visitedNodes = 0;
        //state.print();
        if(depth == 0)return 1;
        clock_t start=clock();
        bool inCheck;
        Move moves[maxMoves];
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
        int count=0;
        for(int i=0; i<nbMoves; i++){
            state.playMove<false>(moves[i]);
            big nbNodes=_perft(state, depth-1);
            state.undoLastMove();
            printf("%s: %lld\n", moves[i].to_str().c_str(), nbNodes);
            count += nbNodes;
        }
        tt.push({state.zobristHash, count, depth});
        clock_t end=clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("%.3f : %.3f nps %d visited nodes\n", tcpu, visitedNodes/tcpu, visitedNodes);
        return count;
    }
};

#endif