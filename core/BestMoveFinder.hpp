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
#include <ctime>
#include <thread>
//#define USE_TT
//#define USE_QTT
int compScoreMove(const void* a, const void*b){
    return ((pair<int, Move>*)b)->first-((pair<int, Move>*)a)->first;
}
void augmentMate(int& score, const Evaluator& eval){
    if(score >= eval.MAXIMUM)
        score++;
    else if(score <= eval.MINIMUM)
        score--;
}
const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
//Class to find the best in a situation
class BestMoveFinder{
    //Returns the best move given a position and time to use
public:
    LegalMoveGenerator generator;
    Evaluator eval;
private:
    std::atomic<bool> running;
public:
    BestMoveFinder(int memory){}
    int alloted_time;
    void stopAfter() {
        std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        running = false; // Set running to false after the specified time
    }
    
private:
    int nodes;
    int negamax(int depth, GameState& state){
        if(!running)return 0;
        if(depth == 0)return eval.positionEvaluator(state);
        nodes++;
        int score_max = -eval.INF;
        Move moves[maxMoves];
        bool inCheck;
        int nbMoves = generator.generateLegalMoves(state, inCheck, moves);
        if(nbMoves == 0){
            if(inCheck)
                return eval.MINIMUM;
            return eval.MIDDLE;
        }
        for(int i=0; i<nbMoves; i++){
            Move curMove = moves[i];
            int score;
            if(state.playMove<false>(curMove) > 1) // 2 repetition, calulated as the same as 3 repetition
                score = 0;
            else
                score = -negamax(depth-1, state);
            state.undoLastMove();
            if(!running)return 0;
            augmentMate(score, eval);
            if(score > score_max)
                score_max = score;
        }
        return score_max;
    }
public:
    Move bestMove(GameState& state, int alloted_time){
        running = true;
        this->alloted_time = alloted_time;
        thread timerThread(&BestMoveFinder::stopAfter, this);
        Move bestMove;
        Move lastBest;
        for(int depth=1; depth<255 && running; depth++){
            lastBest = bestMove;
            clock_t start=clock();
            Move moves[maxMoves];
            bool inCheck;
            int nbMoves = generator.generateLegalMoves(state, inCheck, moves);
            int max_score = -eval.INF;
            assert(nbMoves > 0); //the game is over, which should not append
            int idMove;
            for(idMove=0; idMove<nbMoves; idMove++){
                Move curMove = moves[idMove];
                int score;
                if(state.playMove<false>(curMove) > 1)
                    score = 0;
                else score = -negamax(depth, state);
                augmentMate(score, eval);
                state.undoLastMove();
                if(!running)break;
                if(score > max_score){
                    bestMove = curMove;
                    max_score = score;
                }
            }
            clock_t end = clock();
            double tcpu = double(end-start)/CLOCKS_PER_SEC;
            printf("depth: %d ; speed %d n/s ; score %d cp ; best move %s (%d/%d)\n", depth, (int)(nodes/tcpu), max_score, bestMove.to_str().c_str(), idMove, nbMoves);
            if(abs(max_score) >= eval.MAXIMUM){
                lastBest = bestMove;
                break;
            }
        }
        timerThread.join();
        return lastBest;
    }
    int testQuiescenceSearch(GameState& state){
        return 0;
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
            clock_t startMove=clock();
            int startVisitedNodes = visitedNodes;
            state.playMove<false, false>(moves[i]);
            big nbNodes=_perft(state, depth-1);
            state.undoLastMove<false>();
            clock_t end=clock();
            double tcpu = double(end-startMove)/CLOCKS_PER_SEC;
            printf("%s: %lld (%d/%d %.2fs => %.0f n/s)\n", moves[i].to_str().c_str(), nbNodes, i+1, nbMoves, tcpu, (visitedNodes-startVisitedNodes)/tcpu);
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