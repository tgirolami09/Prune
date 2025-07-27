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
#define USE_TT
#define USE_QTT
#define MoveScore pair<int, Move>
int compScoreMove(const void* a, const void*b){
    return ((pair<int, Move>*)b)->first-((pair<int, Move>*)a)->first;
}
void augmentMate(int& score, const Evaluator& eval){
    if(score >= MAXIMUM)
        score++;
    else if(score <= MINIMUM)
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
#ifdef USE_TT
    transpositionTable transposition;
#endif
#ifdef USE_QTT
    QuiescenceTT QTT;
#endif
private:
    std::atomic<bool> running;
public:
#ifdef USE_TT
    #ifdef USE_QTT
        BestMoveFinder(int memory):transposition(memory/sizeof(infoScore)), QTT(memory/sizeof(infoQ)){}
    #else
        BestMoveFinder(int memory):transposition(memory/sizeof(infoScore)){}
    #endif
#else
    #ifdef USE_QTT
        BestMoveFinder(int memory):QTT(memory/sizeof(infoQ)){}
    #else
        BestMoveFinder(int memory){}
    #endif
#endif
    int alloted_time;
    void stopAfter() {
        std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        running = false; // Set running to false after the specified time
    }
    
private:
    int nodes;
    template<int maxmoves> //because of the quiescence search, where there are less moves at most
    void moveOrder(Move* moves, int nbMoves, bool color){
        MoveScore sortedMoves[maxmoves];
        int start = 0;
        for(int i=0; i<nbMoves; i++){
            sortedMoves[i].first = eval.score_move(moves[i], color);
            sortedMoves[i].second = moves[i];
        }
        qsort(&sortedMoves[start], nbMoves, sizeof(MoveScore), compScoreMove);
        for(int i=0; i<nbMoves; i++){
            moves[i] = sortedMoves[i].second;
        }
    }

    int quiescenceSearch(GameState& state, int alpha, int beta){
#ifdef USE_QTT
        int lastEval=QTT.get_eval(state, alpha, beta);
        if(lastEval != INF)
            return lastEval;
#endif
        int staticEval = eval.positionEvaluator(state);
        if(staticEval >= beta)return staticEval;
        int bestEval = staticEval;
        Move captures[maxCaptures];
        bool inCheck;
        int nbCaptures = generator.generateLegalMoves(state, inCheck, captures, true);
        moveOrder<maxCaptures>(captures, nbCaptures, state.friendlyColor());
        for(int i=0; i<nbCaptures; i++){
            state.playMove<false, false>(captures[i]);//don't care about repetition
            int score = -quiescenceSearch(state, -beta, -alpha);
            state.undoLastMove<false>();
            if(score >= beta){
#ifdef USE_QTT
                QTT.push(state, score, alpha, beta);
#endif
                return score;
            }
            if(score > bestEval)bestEval = score;
            if(score > alpha)alpha = score;
        }
        QTT.push(state, bestEval, alpha, beta);
        return bestEval;
    }

    int negamax(int depth, GameState& state, int alpha, int beta){
        if(!running)return 0;
        if(depth == 0)return eval.positionEvaluator(state);
        nodes++;
        int score_max = -INF;
        Move moves[maxMoves];
        bool inCheck;
        int nbMoves = generator.generateLegalMoves(state, inCheck, moves);
        if(nbMoves == 0){
            if(inCheck)
                return MINIMUM;
            return MIDDLE;
        }
        moveOrder<maxMoves>(moves, nbMoves, state.friendlyColor());
        for(int i=0; i<nbMoves; i++){
            Move curMove = moves[i];
            int score;
            if(state.playMove<false>(curMove) > 1) // 2 repetition, calulated as the same as 3 repetition
                score = MIDDLE;
            else
                score = -negamax(depth-1, state, -beta, -alpha);
            state.undoLastMove();
            if(!running)return 0;
            augmentMate(score, eval);
            if(score > score_max)
                score_max = score;
            if(score > alpha)alpha = score;
            if(score >= beta)
                return score;
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
            nodes = 0;
            lastBest = bestMove;
            clock_t start=clock();
            Move moves[maxMoves];
            bool inCheck;
            int nbMoves = generator.generateLegalMoves(state, inCheck, moves);
            int alpha = -INF;
            int beta = INF;
            assert(nbMoves > 0); //the game is over, which should not append
            moveOrder<maxMoves>(moves, nbMoves, state.friendlyColor());
            int idMove;
            for(idMove=0; idMove<nbMoves; idMove++){
                Move curMove = moves[idMove];
                int score;
                if(state.playMove<false>(curMove) > 1)
                    score = MIDDLE;
                else score = -negamax(depth, state, -beta, -alpha);
                augmentMate(score, eval);
                state.undoLastMove();
                if(!running)break;
                if(score > alpha){
                    bestMove = curMove;
                    alpha = score;
                }
            }
            clock_t end = clock();
            double tcpu = double(end-start)/CLOCKS_PER_SEC;
            printf("depth: %d ; speed %d n/s ; %d nodes ; score %d cp ; best move %s (%d/%d)\n", depth, (int)(nodes/tcpu), nodes, alpha, bestMove.to_str().c_str(), idMove, nbMoves);
            fflush(stdout);
            if(abs(alpha) >= MAXIMUM && idMove == nbMoves){
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