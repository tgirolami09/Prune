#ifndef BESTMOVEFINDER_HPP
#define BESTMOVEFINDER_HPP
#include "TranspositionTable.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include "LegalMoveGenerator.hpp"
#include "MoveOrdering.hpp"
#include "loadpolyglot.hpp"
#include <cmath>
#include <chrono>
#include <atomic>
#include <ctime>
#include <string>
#include <thread>
#define MoveScore pair<int, Move>
const int maxDepth=200;
int compScoreMove(const void* a, const void*b){
    int first = ((MoveScore*)a)->first;
    int second = ((MoveScore*)b)->first;
    return second-first; //https://stackoverflow.com/questions/8115624/using-quick-sort-in-c-to-sort-in-reverse-direction-descending
}
void augmentMate(int& score){
    if(score > MAXIMUM-maxDepth)
        score--;
    else if(score < MINIMUM+maxDepth)
        score++;
}

class Score{
public:
    int score;
    bool usable;
    big useInTT;//when, if the score is 0, is usable in the tt
    Score(){}
    Score(int _score):score(_score), usable(true), useInTT(-1){}
    Score(int _score, bool _usable, big repet):score(_score), usable(_usable), useInTT(repet){}
    void augmentMate(){
        if(score > MAXIMUM-maxDepth)
            score--;
        else if(score < MINIMUM+maxDepth)
            score++;
    }

    void update(big repet){
        if(!usable && repet == useInTT)
            usable = true;
    }

    Score operator-(){
        return Score(-score, usable, useInTT);
    }
    bool operator>(Score o){
        return score > o.score;
    }
    bool operator>=(Score o){
        return score >= o.score;
    }
    bool operator<(Score o){
        return score < o.score;
    }
    bool operator<=(Score o){
        return score <= o.score;
    }
    bool operator>(int otherScore){
        return score > otherScore;
    }
    bool operator>=(int otherScore){
        return score >= otherScore;
    }
    bool operator<(int otherScore){
        return score < otherScore;
    }
    bool operator<=(int otherScore){
        return score <= otherScore;
    }
};

string scoreToStr(int score){
    if(score > MAXIMUM-maxDepth)
        return ((string)"mate ")+to_string((MAXIMUM-score)/2);
    if(score < MINIMUM+maxDepth)
        return ((string)"mate ")+to_string((-(MAXIMUM+score))/2);
    return "cp "+to_string(score);
}

const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
const int maxExtension = 16;
//Class to find the best in a situation
class BestMoveFinder{
    unordered_map<uint64_t,PolyglotEntry> book;

    //Returns the best move given a position and time to use
public:
    LegalMoveGenerator generator;
    Evaluator eval;
    transpositionTable transposition;
    QuiescenceTT QTT;
private:
    std::atomic<bool> running;
public:

    BestMoveFinder(int memory):transposition(memory), QTT(memory){
        book = load_book("./book.bin");
    }

    int alloted_time;
    void stopAfter() {
        std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        running = false; // Set running to false after the specified time
    }
    void stop(){
        running = false;
    }

private:
    int nodes, Qnodes;
    template<int maxmoves> //because of the quiescence search, where there are less moves at most
    void moveOrder(Move* moves, int nbMoves, bool color, big& dangerPositions, Move lastBest=nullMove){
        MoveScore sortedMoves[maxmoves];
        int start = 0;
        for(int i=0; i<nbMoves; i++){
            sortedMoves[i].second = moves[i];
            if(moves[i].start_pos == lastBest.start_pos && moves[i].end_pos == lastBest.end_pos){
                sortedMoves[i].first = sortedMoves[0].first;
                sortedMoves[i].second = sortedMoves[0].second;
                sortedMoves[0].second = moves[i];
                start++;
            }else
                sortedMoves[i].first = eval.score_move(moves[i], color, dangerPositions);
        }
        qsort(sortedMoves+start, nbMoves-start, sizeof(MoveScore), compScoreMove);
        for(int i=0; i<nbMoves; i++){
            moves[i] = sortedMoves[i].second;
        }
    }

    int quiescenceSearch(GameState& state, int alpha, int beta){
        if(!running)return 0;
        Qnodes++;
        int lastEval=QTT.get_eval(state, alpha, beta);
        if(lastEval != INVALID)
            return lastEval;
        int staticEval = eval.positionEvaluator(state);
        if(staticEval >= beta){
            QTT.push(state, staticEval, LOWERBOUND);
            return staticEval;
        }
        int typeNode = UPPERBOUND;
        if(staticEval > alpha){
            alpha = staticEval;
            typeNode = EXACT;
        }
        int bestEval = staticEval;
        Order<maxCaptures> order;
        bool inCheck;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions, true);
        order.init(eval, state.friendlyColor(), nullMove);
        for(int i=0; i<order.nbMoves; i++){
            Move capture = order.pop_max();
            state.playMove<false, false>(capture);//don't care about repetition
            int score = -quiescenceSearch(state, -beta, -alpha);
            state.undoLastMove<false>();
            if(!running)return 0;
            if(score >= beta){
                QTT.push(state, score, LOWERBOUND);
                return score;
            }
            if(score > bestEval)bestEval = score;
            if(score > alpha){
                alpha = score;
                typeNode = EXACT;
            }
        }
        QTT.push(state, bestEval, typeNode);
        return bestEval;
    }

     Score negamax(int depth, GameState& state, int alpha, int beta, int numExtension){
        if(!running)return 0;
        if(depth == 0)return Score(quiescenceSearch(state, alpha, beta), true, -1);
        nodes++;
        Move lastBest = nullMove;
        int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
        if(lastEval != INVALID)
            return Score(lastEval, true, -1);
        ubyte typeNode = UPPERBOUND;
        Order<maxMoves> order;
        bool inCheck=false;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
            if(inCheck)
                return Score(MINIMUM, true, -1);
            return Score(MIDDLE, true, -1);
        }
        if(inCheck && numExtension < maxExtension){
            numExtension++;
            depth++;
        }
        order.init(eval, state.friendlyColor(), lastBest);
        Move bestMove;
        Score bestScore(-INF, false, -1);
        for(int i=0; i<order.nbMoves; i++){
            Move curMove = order.pop_max();
            Score score;
            if(state.playMove<false>(curMove) > 1){
                score = Score(MIDDLE, false, state.zobristHash);
            }else{
                score = -negamax(depth-1, state, -beta, -alpha, numExtension);
                score.update(state.zobristHash);
            }
            state.undoLastMove();
            if(!running)return 0;
            score.augmentMate();
            if(score >= beta){
                if(score.usable){
                    transposition.push(state, score.score, LOWERBOUND, curMove, depth);
                }
                return score;
            }
            if(score > alpha){
                alpha = score.score;
                typeNode=EXACT;
            }
            if(score > bestScore)bestScore = score;
        }
        if(bestScore.usable){
            transposition.push(state, bestScore.score, typeNode, bestMove, depth);
        }
        return bestScore;
    }
public:
    Move bestMove(GameState& state, int alloted_time){
        bool moveInTable = false;
        Move bookMove = findPolyglot(state,moveInTable,book);
        //Return early because a move was found in a book
        if (moveInTable){
            printf("Found book move for fen : %s\n",state.toFen().c_str());
            return bookMove;
        }
        running = true;
        this->alloted_time = alloted_time;
        thread timerThread(&BestMoveFinder::stopAfter, this);
        printf("info string use a tt of %d entries (%ld MB)\n", transposition.modulo, transposition.modulo*sizeof(infoScore)*2/1000000);
        printf("info string use a quiescence tt of %d entries (%ld MB)\n", QTT.modulo, QTT.modulo*sizeof(infoQ)/1000000);
        Move bestMove=nullMove;
        Move lastBest=nullMove;
        Qnodes = nodes = 0;
        clock_t start=clock();
        for(int depth=1; depth<255 && running; depth++){
            lastBest = bestMove;
            Order<maxMoves> order;
            bool inCheck;
            order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
            int alpha = -INF;
            int beta = INF;
            assert(order.nbMoves > 0); //the game is over, which should not append
            int idMove;
            order.init(eval, state.friendlyColor(), lastBest);
            for(idMove=0; idMove<order.nbMoves; idMove++){
                Move curMove = order.pop_max();
                int score;
                if(state.playMove<false>(curMove) > 1)
                    score = MIDDLE;
                else score = -negamax(depth, state, -beta, -alpha, 0).score;
                augmentMate(score);
                //printf("info string %s : %d\n", curMove.to_str().c_str(), score);
                state.undoLastMove();
                if(!running)break;
                if(score > alpha){
                    bestMove = curMove;
                    alpha = score;
                }
            }
            clock_t end = clock();
            double tcpu = double(end-start)/CLOCKS_PER_SEC;
            if(idMove == order.nbMoves)
                printf("info depth %d score %s nodes %d nps %d time %d pv %s\n", depth, scoreToStr(alpha).c_str(), nodes, (int)(nodes/tcpu), (int)(tcpu*1000), bestMove.to_str().c_str());
            else if(idMove)printf("info depth %d score %s nodes %d nps %d time %d pv %s string %d/%d moves\n", depth, scoreToStr(alpha).c_str(), nodes, (int)(nodes/tcpu), (int)(tcpu*1000), bestMove.to_str().c_str(), idMove, order.nbMoves);
            fflush(stdout);
            if(abs(alpha) >= MAXIMUM-maxDepth && idMove == order.nbMoves){//checkmate found, stop the thread
                timerThread.join();
                return bestMove;
            }
        }
        timerThread.join();
        return bestMove;
    }
    int testQuiescenceSearch(GameState& state){
        Qnodes = 0;
        clock_t start=clock();
        int score = quiescenceSearch(state, -INF, INF);
        clock_t end = clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("speed: %d; Qnodes:%d\n\n", (int)(Qnodes/tcpu), Qnodes);
        return 0;
    }
};


class Perft{
public:
    TTperft tt;
    LegalMoveGenerator generator;
    size_t space;
    Perft(size_t _space):tt(0), space(_space){}
    big visitedNodes;
    big _perft(GameState& state, ubyte depth){
        visitedNodes++;
        if(depth == 0)return 1;
        big lastCall=tt.get_eval(state.zobristHash, depth);
        if(lastCall != -1)return lastCall;
        bool inCheck;
        Move moves[maxMoves];
        big dangerPositions = 0;
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves, dangerPositions);
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
        tt.reinit(space);
        visitedNodes = 0;
        if(depth == 0)return 1;
        clock_t start=clock();
        bool inCheck;
        Move moves[maxMoves];
        big dangerPositions = 0;
        int nbMoves=generator.generateLegalMoves(state, inCheck, moves, dangerPositions);
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
        tt.clearMem();
        return count;
    }
};

#endif