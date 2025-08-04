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
    ubyte depth;//when, if the score is 0, is usable in the tt
    Score(){}
    Score(int _score):score(_score), depth(-1){}
    Score(int _score, ubyte _depth):score(_score), depth(_depth){}
    void augmentMate(){
        if(score > MAXIMUM-maxDepth)
            score--;
        else if(score < MINIMUM+maxDepth)
            score++;
    }

    bool usable(ubyte _depth){
        return _depth <= depth;
    }

    Score operator-(){
        return Score(-score, depth);
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
    std::atomic<bool> midtime;
public:

    BestMoveFinder(int memory):transposition(memory), QTT(memory){
        book = load_book("./book.bin");
    }

    int alloted_time;
    void stopAfter() {
        //std::this_thread::sleep_for(std::chrono::milliseconds(alloted_time));
        for(int i=0; i<2 && running; i++){
            auto start=chrono::high_resolution_clock::now();
            auto end=start;
            do{
                this_thread::sleep_for(chrono::milliseconds(10));
                end = chrono::high_resolution_clock::now();
            }while(chrono::duration_cast<chrono::milliseconds>(end-start).count() < alloted_time/2 && running);
            midtime = true;
        }
        running = false; // Set running to false after the specified time
    }
    void stop(){
        running = false;
    }

private:
    int nodes, Qnodes;
    big isInSearch[maxDepth];
    template<bool timeLimit>
    int quiescenceSearch(GameState& state, int alpha, int beta){
        if(timeLimit && !running)return 0;
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
            int score = -quiescenceSearch<timeLimit>(state, -beta, -alpha);
            state.undoLastMove<false>();
            if(timeLimit && !running)return 0;
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

    ubyte isRepet(big hash, int lastChange, int pos){
        for(int i=lastChange; i<pos-3; i++){
            if(isInSearch[i] == hash)
                return i;
        }
        return (ubyte)-1;
    }
    void setElement(big hash, int pos){
        isInSearch[pos] = hash;
    }

    bool isOnlyPawns(const GameState& state){
        const big* fp = state.friendlyPieces();
        const big* ep = state.enemyPieces();
        return fp[BISHOP] || fp[KNIGHT] || fp[ROOK] || fp[QUEEN] || ep [BISHOP] || ep[KNIGHT] || ep[ROOK] || ep[QUEEN];
    }

    template <bool isPV, bool timeLimit>
    Score negamax(int depth, GameState& state, int alpha, int beta, int numExtension, int lastChange, int relDepth){
        if(timeLimit && !running)return 0;
        if(depth == 0)return Score(quiescenceSearch<timeLimit>(state, alpha, beta), -1);
        nodes++;
        Move lastBest = nullMove;
        int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
        if(lastEval != INVALID)
            return Score(lastEval, -1);
        ubyte typeNode = UPPERBOUND;
        Order<maxMoves> order;
        bool inCheck=false;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
            if(inCheck)
                return Score(MINIMUM, -1);
            return Score(MIDDLE, -1);
        }
        if(inCheck && numExtension < maxExtension){
            numExtension++;
            depth++;
        }
        int r = 3;
        if(depth > r && !inCheck && !isPV && eval.positionEvaluator(state) >= beta){
            state.playNullMove();
            Score v = -negamax<false, timeLimit>(depth-r, state, -beta, -beta+1, numExtension, lastChange, relDepth+1);
            state.undoNullMove();
            if(v.score >= beta)return v;
        }
        order.init(eval, state.friendlyColor(), lastBest);
        Move bestMove;
        Score bestScore(-INF, -1);
        for(int i=0; i<order.nbMoves; i++){
            Move curMove = order.pop_max();
            Score score;
            state.playMove<false, false>(curMove);
            int newLastChange = lastChange;
            if(curMove.capture != -2 || curMove.piece == PAWN)
                newLastChange = relDepth;
            ubyte usableDepth = isRepet(state.zobristHash, newLastChange, relDepth);
            if(usableDepth != (ubyte)-1){
                score = Score(MIDDLE, relDepth);
            }else{
                setElement(state.zobristHash, relDepth);
                if(i != 0){
                    score = -negamax<false, timeLimit>(depth-1, state, -alpha-1, -alpha, numExtension, newLastChange, relDepth+1);
                    if(score > alpha && isPV){
                        score = -negamax<true, timeLimit>(depth-1, state, -beta, -alpha, numExtension, newLastChange, relDepth+1);
                    }
                }else
                    score = -negamax<isPV, timeLimit>(depth-1, state, -beta, -alpha, numExtension, newLastChange, relDepth+1);
            }
            state.undoLastMove<false>();
            if(timeLimit && !running)return 0;
            score.augmentMate();
            if(score >= beta){
                if(score.usable(relDepth)){
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
        if(bestScore.usable(relDepth)){
            transposition.push(state, bestScore.score, typeNode, bestMove, depth);
        }
        return bestScore;
    }
public:
    template <bool timeLimit=true>
    Move bestMove(GameState& state, int alloted){
        bool moveInTable = false;
        Move bookMove = findPolyglot(state,moveInTable,book);
        //Return early because a move was found in a book
        if (moveInTable){
            printf("Found book move for fen : %s\n",state.toFen().c_str());
            return bookMove;
        }
        running = true;
        midtime = false;
        thread timerThread;
        int depthMax = maxDepth;
        if(timeLimit){
            this->alloted_time = alloted;
            timerThread = thread(&BestMoveFinder::stopAfter, this);
        }else{
            depthMax = alloted;
        }
        printf("info string use a tt of %d entries (%ld MB)\n", transposition.modulo, transposition.modulo*sizeof(infoScore)*2/1000000);
        printf("info string use a quiescence tt of %d entries (%ld MB)\n", QTT.modulo, QTT.modulo*sizeof(infoQ)/1000000);
        Move bestMove=nullMove;
        Move lastBest=nullMove;
        Qnodes = nodes = 0;
        clock_t start=clock();
        for(int depth=1; depth<depthMax && running && !midtime; depth++){
            lastBest = bestMove;
            Order<maxMoves> order;
            bool inCheck;
            order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
            int alpha = -INF;
            int beta = INF;
            assert(order.nbMoves > 0); //the game is over, which should not append
            int idMove;
            setElement(state.zobristHash, 0);
            order.init(eval, state.friendlyColor(), lastBest);
            for(idMove=0; idMove<order.nbMoves; idMove++){
                Move curMove = order.pop_max();
                int score;
                //printf("%016llx\n", state.zobristHash);
                if(state.playMove<false>(curMove) > 1)
                    score = MIDDLE;
                else{
                    setElement(state.zobristHash, 1);
                    score = -negamax<true, timeLimit>(depth, state, -beta, -alpha, 0, (curMove.capture == -2 && curMove.piece == PAWN), 2).score;
                }
                augmentMate(score);
                //printf("info string %s : %d\n", curMove.to_str().c_str(), score);
                //printf("%016llx\n", state.zobristHash);
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
                if(timeLimit){
                    running = false;
                    timerThread.join();
                }
                return bestMove;
            }
        }
        if(timeLimit){
            running = false;
            timerThread.join();
        }
        return bestMove;
    }
    int testQuiescenceSearch(GameState& state){
        Qnodes = 0;
        clock_t start=clock();
        int score = quiescenceSearch<false>(state, -INF, INF);
        clock_t end = clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("speed: %d; Qnodes:%d\n\n", (int)(Qnodes/tcpu), Qnodes);
        return 0;
    }

    void clear(){
        transposition.clear();
        QTT.clear();
    }

    void reinit(size_t count){
        transposition.reinit(count);
        QTT.reinit(count);
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
    void reinit(size_t count){
        space = count;
    }
};

#endif