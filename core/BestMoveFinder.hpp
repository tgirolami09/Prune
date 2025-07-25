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
    void orderMove(Move* moves, int nbMoves, Move possibleBest){
        vector<pair<int, Move>> sortedMoves(nbMoves);
        int start=0;
        for(int i=0; i<nbMoves; i++){
            sortedMoves[i] = {eval.score_move(moves[i]), moves[i]};
            if(moves[i].start_pos == possibleBest.start_pos && moves[i].end_pos == possibleBest.end_pos){
                swap(sortedMoves[0], sortedMoves[i]);
                start++;
            }
        }
        sort(sortedMoves.begin()+start, sortedMoves.end(), compScoreMove);
        for(int i=0; i<nbMoves; i++){
            moves[i] = sortedMoves[i].second;
        }
    }

    int quiescenceSearch(GameState& state, int alpha, int beta){
        if(!running)return 0;
        int evaluation = eval.positionEvaluator(state);
        if(evaluation >= beta)return beta;
        alpha = max(alpha, evaluation);
        bool inCheck;
        Move captureMoves[12*8+4*4]; //maximum number of capture : each piece can capture in each direction
        int nbMoves = generator.generateLegalMoves(state, inCheck, captureMoves, true);
        if(nbMoves == 0)return evaluation;
        vector<pair<int, Move>> sortedMoves(nbMoves);
        for(int i=0; i<nbMoves; i++){
            sortedMoves[i] = {eval.score_move(captureMoves[i]), captureMoves[i]};
        }
        sort(sortedMoves.begin(), sortedMoves.end(), compScoreMove);
        for(int idMove=0; idMove<nbMoves; idMove++){
            Move move = sortedMoves[idMove].second;
            state.playMove<false>(move);
            int score = -quiescenceSearch(state, -beta, -alpha);
            state.undoLastMove();
            if(!running)return 0;
            if(score >= beta){
                return beta;
            }
            alpha = max(alpha, score);
        }
        return alpha;
    }

    int negamax(int depth, GameState& state, int alpha, int beta){
        if(!running)return 0;
        if(depth == 0)
            return quiescenceSearch(state, alpha, beta);
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
        orderMove(moves, nbMoves, bMove);
        for(int i=0; i<nbMoves; i++){
            Move move=moves[i];
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
        printf("%dms to search\n", alloted_time);
        Move lastBest;
        std::thread timerThread(&BestMoveFinder::stopAfter, this);
        for(int depth=1; running; depth++){
            printf("running with depth: %d\n", depth);
            int alpha=eval.MINIMUM;
            int beta=eval.MAXIMUM;
            bool inCheck;
            Move moves[maxMoves];
            int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
            if(nbMoves == 0)
                return {}; // no possible moves
            orderMove(moves, nbMoves, lastBest);
            for(int i=0; i<nbMoves; i++){
                state.playMove<false>(moves[i]);
                int score = -negamax(depth, state, -beta, -alpha);
                state.undoLastMove();
                if(!running)break;
                if(score > alpha){
                    alpha = score;
                    bestMove = moves[i];
                    //printf("new best score: %d with move: %s\n", alpha, bestMove.to_str().c_str());
                }
            }
            printf("best move at depth %d is %s with score %d\n", depth, bestMove.to_str().c_str(), alpha);
            lastBest = bestMove;
            transposition.clear();
        }
        timerThread.join();
        if(bestMove.start_pos == bestMove.end_pos)return lastBest;
        return bestMove;
    }

};


class Perft{
public:
    TTperft tt;
    LegalMoveGenerator generator;
    Perft(size_t space):tt(space){}
    big visitedNodes;
    big _perft(GameState& state, ubyte depth){
        visitedNodes++;
        if(depth == 0)return 1;
        big lastCall=tt.get_eval(state.zobristHash, depth);
        if(lastCall != -1)return lastCall;
        bool inCheck;
        Move moves[maxMoves];
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
        if(depth == 1)return nbMoves;
        big count=0;
        for(int i=0; i<nbMoves; i++){
            state.playMove<false, false>(moves[i]);
            big nbNodes=_perft(state, depth-1);
            state.undoLastMove<false>();
            count += nbNodes;
        }
        tt.push({state.zobristHash, count, depth});
        return count;
    }
    big perft(GameState& state, ubyte depth){
        visitedNodes = 0;
        if(depth == 0)return 1;
        clock_t start=clock();
        bool inCheck;
        Move moves[maxMoves];
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves);
        big count=0;
        for(int i=0; i<nbMoves; i++){
            state.playMove<false, false>(moves[i]);
            big nbNodes=_perft(state, depth-1);
            state.undoLastMove<false>();
            printf("%s: %lld\n", moves[i].to_str().c_str(), nbNodes);
            fflush(stdout);
            count += nbNodes;
        }
        tt.push({state.zobristHash, count, depth});
        clock_t end=clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("%.3f : %.3f nps %lld visited nodes\n", tcpu, visitedNodes/tcpu, visitedNodes);
        fflush(stdout);
        return count;
    }
};

#endif