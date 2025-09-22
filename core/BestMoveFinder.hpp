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
//#define CalculatePV
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

class LINE{
public:
    int cmove;
    int16_t argMoves[maxDepth];
};

string scoreToStr(int score){
    if(score > MAXIMUM-maxDepth)
        return ((string)"mate ")+to_string((MAXIMUM-score+1)/2);
    if(score < MINIMUM+maxDepth)
        return ((string)"mate ")+to_string((-(MAXIMUM+score))/2);
    return "cp "+to_string(score);
}

//Class to find the best in a situation
class BestMoveFinder{
    unordered_map<uint64_t,PolyglotEntry> book;

    //Returns the best move given a position and time to use
    transpositionTable transposition;
    QuiescenceTT QTT;
    std::atomic<bool> running;
    std::atomic<bool> midtime;
    HelpOrdering history;
public:
    IncrementalEvaluator eval;
    BestMoveFinder(int memory, bool mute=false):transposition(memory), QTT(memory){
        book = load_book("./book.bin", mute);
        history.init();
    }

    int hardBound;
    using timeMesure=chrono::high_resolution_clock;
    timeMesure::time_point startSearch;
    chrono::milliseconds hardBoundTime;
    void stop(){
        running = false;
    }

private:
    inline auto getElapsedTime(){
        return timeMesure::now()-startSearch;
    }

    LegalMoveGenerator generator;
    int nodes;
    int nbCutoff;
    int nbFirstCutoff;
    big isInSearch[1000];
    template<int limitWay>
    int quiescenceSearch(GameState& state, int alpha, int beta){
        if(limitWay <= 1 && !running)return 0;
        if(limitWay == 0 && (nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        if(eval.isInsufficientMaterial())return 0;
        nodes++;
        int lastEval=QTT.get_eval(state, alpha, beta);
        if(lastEval != INVALID)
            return lastEval;
        int staticEval = eval.getScore(state.friendlyColor());
        if(staticEval >= beta){
            QTT.push(state, staticEval, LOWERBOUND);
            return staticEval;
        }
        int typeNode = UPPERBOUND;
        if(staticEval > alpha){
            alpha = staticEval;
            typeNode = EXACT;
        }else{
            int delta = value_pieces[QUEEN];
            big promotion_row = state.friendlyColor() == WHITE ? row1 << 8*6 : row1 << 8;
            if(state.friendlyPieces()[PAWN]&promotion_row)
                delta += value_pieces[QUEEN];
            if(staticEval+delta <= alpha)
                return alpha;
        }
        int bestEval = staticEval;
        Order<maxCaptures> order;
        bool inCheck;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions, true);
        order.init(state.friendlyColor(), nullMove.moveInfo, history, -1, state, generator, false);
        for(int i=0; i<order.nbMoves; i++){
            Move capture = order.pop_max();
            state.playMove<false, false>(capture);//don't care about repetition
            eval.playMove(capture, !state.friendlyColor());
            int score = -quiescenceSearch<limitWay>(state, -beta, -alpha);
            eval.undoMove(capture, !state.friendlyColor());
            state.undoLastMove<false>();
            if(limitWay <= 1 && !running)return 0;
            if(score >= beta){
                nbCutoff++;
                if(i == 0)nbFirstCutoff++;
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
        return fp[BISHOP] || fp[KNIGHT] || fp[ROOK] || fp[QUEEN] || ep[BISHOP] || ep[KNIGHT] || ep[ROOK] || ep[QUEEN];
    }
    template<int limitWay> 
    pair<int, Move> MatedSearch(GameState& state, int depth, int lastChange, int relDepth){
        if(!running)return {-1, nullMove};
        if(depth <= 0 || relDepth-lastChange >= 100)
            return {-1, nullMove};
        nodes++;
        if(limitWay == 1 && nodes > hardBound)running=false;
        if(limitWay == 0 && (nodes&1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        Order<maxMoves> order;
        bool inCheck=false;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
            //return the number of plies gained by going in this way
            if(inCheck) return {depth, nullMove};
            else return {-1, nullMove};
        }
        int bestReduction = 100;
        Move bestMove = nullMove;
        order.init(state.friendlyColor(), nullMove.moveInfo, history, relDepth, state, generator, false);
        for(int rankMove=0; rankMove < order.nbMoves; rankMove++){
            Move curMove = order.pop_max();
            int score;
            state.playMove<false, false>(curMove);
            int newLastChange = lastChange;
            if(curMove.isChanger())
                newLastChange = relDepth;
            ubyte usableDepth = isRepet(state.zobristHash, newLastChange, relDepth);
            if(usableDepth != (ubyte)-1){
                score = -1;
            }else{
                setElement(state.zobristHash, relDepth);
                score = MaterSearch<limitWay>(state, depth-1, newLastChange, relDepth+1).first;
            }
            state.undoLastMove<false>();
            if(score == -1)return {-1, curMove}; //found a way to escape the mate in this line
            if(score < bestReduction){
                bestReduction = score;
                bestMove = curMove;
            }
            if(!running)return {bestReduction, bestMove};// there we can use the last iteration, and we must return the info for the mate because otherwise we cannot make at least one move
        }
        return {bestReduction, bestMove};

    }
    template<int limitWay>
    pair<int, Move> MaterSearch(GameState& state, int depth, int lastChange, int relDepth){
        //when detect mate in x
        if(!running)return {-1, nullMove};
        if(depth <= 0 || relDepth-lastChange >= 100)
            return {-1, nullMove};
        nodes++;
        if(limitWay == 1 && nodes > hardBound)running=false;
        if(limitWay == 0 && (nodes&1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        Order<maxMoves> order;
        bool inCheck=false;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
            return {-1, nullMove}; // we search the mate for our side, not for the other
        }
        int bestReduction = -1;
        Move bestMove = nullMove;
        order.init(state.friendlyColor(), nullMove.moveInfo, history, relDepth, state, generator, false);
        for(int rankMove=0; rankMove < order.nbMoves; rankMove++){
            Move curMove = order.pop_max();
            int score;
            state.playMove<false, false>(curMove);
            int newLastChange = lastChange;
            if(curMove.isChanger())
                newLastChange = relDepth;
            ubyte usableDepth = isRepet(state.zobristHash, newLastChange, relDepth);
            if(usableDepth != (ubyte)-1){
                score = -1;
            }else{
                setElement(state.zobristHash, relDepth);
                score = MatedSearch<limitWay>(state, depth-2-bestReduction, newLastChange, relDepth+1).first;
            }
            state.undoLastMove<false>();
            if(!running)return {bestReduction, bestMove}; // cannot take the las iteration, because the other side has not searched all the possible responses
            if(score != -1){//this acts like a max
                bestReduction += score+1;
                //printf("%s : red %d move %s score %d\n", state.toFen().c_str(), bestReduction, curMove.to_str().c_str(), score);
                bestMove = curMove;
            }
        }
        return {bestReduction, bestMove};
    }
#ifdef CalculatePV
    int startRelDepth;
    LINE PVlines[maxDepth]; //store only the move info, because it only need that to print the pv

    string PVprint(LINE pvLine){
        string resLine = "";
        for(int i=0; i<pvLine.cmove; i++){
            Move mv;
            mv.moveInfo = pvLine.argMoves[i];
            if(i != 0)resLine += " ";
            resLine += mv.to_str();
        }
        return resLine;
    }

    void transfer(int relDepth, Move move){
        PVlines[relDepth-1].argMoves[0] = move.moveInfo;
        memcpy(&PVlines[relDepth-1].argMoves[1], PVlines[relDepth].argMoves, PVlines[relDepth].cmove * sizeof(int16_t));
        PVlines[relDepth-1].cmove = PVlines[relDepth].cmove+1;
    }
    void beginLine(int relDepth){
        PVlines[relDepth-1].cmove = 0;
    }

    void beginLineMove(int relDepth, Move move){
        PVlines[relDepth-1].argMoves[0] = move.moveInfo;
        PVlines[relDepth-1].cmove = 1;
    }

    void resetLines(){
        for(int i=0; i<maxDepth; i++){
            PVlines[i].cmove = 0;
        }
    }
#endif
    enum{PVNode=0, CutNode=1, AllNode=-1};
    template <int nodeType, int limitWay>
    Score negamax(int depth, GameState& state, int alpha, int beta, int numExtension, int lastChange, int relDepth){
        if(limitWay <= 1 && !running)return 0;
        if(limitWay == 0 && (nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        if(relDepth-lastChange >= 100 || eval.isInsufficientMaterial()){
#ifdef CalculatePV
            if(nodeType == PVNode)beginLine(relDepth-startRelDepth);
#endif
            return Score(0, -1);
        }
        int static_eval = eval.getScore(state.friendlyColor());
        if(depth == 0 || (depth == 1 && (static_eval+100 < alpha || static_eval > beta+100))){
#ifdef CalculatePV
            if(nodeType == PVNode)beginLine(relDepth-startRelDepth);
#endif
            Score score = Score(quiescenceSearch<limitWay>(state, alpha, beta), -1);
            if(limitWay == 1 && nodes > hardBound)running=false;
            return score;
        }
        nodes++;
        int16_t lastBest = nullMove.moveInfo;
        int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
        if(lastEval != INVALID){
#ifdef CalculatePV
            if(nodeType == PVNode)beginLine(relDepth-startRelDepth);
#endif
            return Score(lastEval, -1);
        }
        ubyte typeNode = UPPERBOUND;
        Order<maxMoves> order;
        bool inCheck=false;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
#ifdef CalculatePV
            if(nodeType == PVNode)beginLine(relDepth-startRelDepth);
#endif
            if(inCheck)
                return Score(MINIMUM, -1);
            return Score(MIDDLE, -1);
        }
        if((inCheck || order.nbMoves == 1) && numExtension < maxExtension){
            numExtension++;
            depth++;
        }else if(!inCheck && nodeType != PVNode){
            int margin = 150*depth;
            if(static_eval >= beta+margin){
                Score score = Score(quiescenceSearch<limitWay>(state, alpha, beta), -1);
                if(limitWay == 1 && nodes > hardBound)running=false;
                return score;
            }
        }
        if(order.nbMoves == 1){
            state.playMove<false, false>(order.moves[0]);
            eval.playMove(order.moves[0], !state.friendlyColor());
            Score sc = -negamax<-nodeType, limitWay>(depth-1, state, -beta, -alpha, numExtension, lastChange, relDepth+1);
            eval.undoMove(order.moves[0], !state.friendlyColor());
            state.undoLastMove<false>();
#ifdef CalculatePV
            if((running || limitWay == 2) && nodeType == PVNode && sc.score > alpha)transfer(relDepth-startRelDepth, order.moves[0]);
#endif
            return sc;
        }
        int r = 3;
        if(depth >= r && !inCheck && nodeType != PVNode && !eval.isOnlyPawns() && eval.getScore(state.friendlyColor()) >= beta){
            state.playNullMove();
            Score v = -negamax<CutNode, limitWay>(depth-r, state, -beta, -beta+1, numExtension, lastChange, relDepth+1);
            state.undoNullMove();
            if(v.score >= beta)return Score(beta, v.depth);
        }
        order.init(state.friendlyColor(), lastBest, history, relDepth, state, generator, depth > 5);
        Move bestMove;
        Score bestScore(-INF, -1);
        for(int rankMove=0; rankMove<order.nbMoves; rankMove++){
            Move curMove = order.pop_max();
            Score score;
            state.playMove<false, false>(curMove);
            int newLastChange = lastChange;
            if(curMove.isChanger())
                newLastChange = relDepth;
            ubyte usableDepth = isRepet(state.zobristHash, newLastChange, relDepth);
#ifdef CalculatePV
            bool isDraw = false;
#endif
            if(usableDepth != (ubyte)-1){
                score = Score(MIDDLE, usableDepth);
#ifdef CalculatePV
                isDraw = true;
#endif
            }else{
                eval.playMove(curMove, !state.friendlyColor());
                setElement(state.zobristHash, relDepth);
                if(rankMove > 0){
                    int reductionDepth = 0;
                    if(rankMove > 3 && depth > 3 && !curMove.isTactical()){
                        reductionDepth = static_cast<int>(0.9 + log(depth) * log(rankMove) / 3);
                    }
                    score = -negamax<((nodeType == CutNode)?AllNode:CutNode), limitWay>(depth-1-reductionDepth, state, -alpha-1, -alpha, numExtension, newLastChange, relDepth+1);
                    bool fullSearch = false;
                    if((score > alpha && score < beta) || (nodeType == PVNode && score.score == beta && beta == alpha+1)){
                        fullSearch = true;
                    }
                    if(reductionDepth && score >= beta)
                        fullSearch = true;
                    if(fullSearch)
                        score = -negamax<nodeType, limitWay>(depth-1, state, -beta, -alpha, numExtension, newLastChange, relDepth+1);
                }else
                    score = -negamax<-nodeType, limitWay>(depth-1, state, -beta, -alpha, numExtension, newLastChange, relDepth+1);
                eval.undoMove(curMove, !state.friendlyColor());
            }
            state.undoLastMove<false>();
            if(limitWay <= 1 && !running)return 0;
            score.augmentMate();
            if(score >= beta){ //no need to copy the pv, because it will fail low on the parent
                if(score.usable(relDepth)){
                    transposition.push(state, score.score, LOWERBOUND, curMove, depth);
                }
                nbCutoff++;
                if(rankMove == 0)nbFirstCutoff++;
                history.addKiller(curMove, depth, relDepth, state.friendlyColor(), state.getLastMove());
                return score;
            }
            if(score > alpha){
                alpha = score.score;
                typeNode=EXACT;
#ifdef CalculatePV
                if(nodeType == PVNode){
                    if(isDraw)beginLineMove(relDepth-startRelDepth, curMove);
                    else transfer(relDepth-startRelDepth, curMove);
                }
#endif
            }
            if(score > bestScore)bestScore = score;
        }
        if(nodeType==CutNode && bestScore.score == alpha)
            return bestScore;
        if(bestScore.usable(relDepth)){
            transposition.push(state, bestScore.score, typeNode, bestMove, depth);
        }
        return bestScore;
    }
    template<int limitWay>
    Move bestMoveClipped(int depth, GameState& state, int alpha, int beta, int& bestScore, Move lastBest, int& idMove, RootOrder& order, int actDepth, int lastChange){
        bestScore = -INF;
        Move bestMove = nullMove;
        order.reinit(lastBest.moveInfo);
#ifdef CalculatePV
        startRelDepth = actDepth-1;
#endif
        for(idMove=0; idMove < order.nbMoves; idMove++){
            Move curMove = order.pop_max();
            //printf("%s\n", curMove.to_str().c_str());
            int startNodes = nodes;
            int score;
            int curLastChange = lastChange;
            if(curMove.isChanger())
                curLastChange = actDepth;
            if(state.playMove<false>(curMove) > 1)
                score = MIDDLE;
            else{
                eval.playMove(curMove, !state.friendlyColor());
                setElement(state.zobristHash, actDepth);
                score = -negamax<PVNode, limitWay>(depth, state, -beta, -alpha, 0, curLastChange, actDepth+1).score;
                eval.undoMove(curMove, !state.friendlyColor());
            }
            augmentMate(score);
            state.undoLastMove();
            if(!running)return bestMove.from() == bestMove.to()?curMove:bestMove;
            int nodeUsed = nodes-startNodes;
            order.pushNodeUsed(nodeUsed);
            if(score >= beta){
                bestScore = score;
                order.cutoff();
                nbCutoff++;
                if(idMove == 0)nbFirstCutoff++;
                return curMove;
            }if(score > alpha){
                bestMove = curMove;
                alpha = score;
                bestScore = score;
#ifdef CalculatePV
                transfer(1, curMove);
#endif
            }else if(score > bestScore){
                bestMove = curMove;
                bestScore = score;
            }
        }
        return bestMove;
    }

public:
    template <int limitWay=0>
    pair<Move, int> bestMove(GameState& state, int softBound, int hardBound, vector<Move> movesFromRoot, bool verbose=true, bool mateHardBound=true){
        startSearch = timeMesure::now();
        hardBoundTime = chrono::milliseconds{hardBound};
        chrono::milliseconds softBoundTime{softBound};
        int actDepth=0;
        int lastChange = 0;
        for(Move move:movesFromRoot){
            setElement(state.zobristHash, actDepth);
            move = state.playPartialMove(move);
            if(move.isChanger())
                lastChange = actDepth;
            actDepth++;
        }
        setElement(state.zobristHash, actDepth);
        bool moveInTable = false;
        Move bookMove = findPolyglot(state,moveInTable,book);
        bool inCheck;
        RootOrder order;
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        //Return early because a move was found in a book
        if (moveInTable){
            moveInTable = false;
            for(int i=0; i<order.nbMoves; i++){
                if(order.moves[i].moveInfo == bookMove.moveInfo){
                    moveInTable = true;
                    break;
                }
            }
            if(moveInTable){
                if(verbose)
                    printf("Found book move for fen : %s\n",state.toFen().c_str());
                return {bookMove, INF};
            }else if(verbose){
                printf("bad move find in table %s (in %s)\n", bookMove.to_str().c_str(), state.toFen().c_str());
            }
        }
        eval.init(state);
        running = true;
        int depthMax = maxDepth;
        if(limitWay == 0){
            this->hardBound = hardBound;
        }else if(limitWay == 1){
            this->hardBound = hardBound; //hard bound
        }else{
            depthMax = hardBound;
        }
        if(order.nbMoves == 1){
            running = false;
            return {order.moves[0], INF};
        }
        if(verbose){
            printf("info string use a tt of %d entries (%ld MB) (%ldB by entry)\n", transposition.modulo, transposition.modulo*sizeof(infoScore)*2/1000000, sizeof(infoScore));
            printf("info string use a quiescence tt of %d entries (%ld MB)\n", QTT.modulo, QTT.modulo*sizeof(infoQ)/1000000);
        }
        Move bestMove=nullMove;
        nodes = 0;
        nbCutoff = nbFirstCutoff = 0;
        clock_t start=clock();
        big lastNodes = 1;
        int lastScore = eval.getScore(state.friendlyColor());
        order.init(state.friendlyColor(), history, state, generator);
        for(int depth=1; depth<depthMax && running; depth++){
            int deltaUp = 10;
            int deltaDown = 10;
            int startNodes = nodes;
            int idMove;
            int bestScore;
            do{
                int alpha = lastScore-deltaDown;
                int beta = lastScore+deltaUp;
                bestMove = bestMoveClipped<limitWay>(depth, state, alpha, beta, bestScore, bestMove, idMove, order, actDepth, lastChange);
                if(bestScore <= alpha)deltaDown = max(deltaDown*2, lastScore-bestScore+1);
                else if(bestScore >= beta)deltaUp = max(deltaUp*2, bestScore-lastScore+1);
                else break;
            }while(running);
            if(idMove)
                lastScore = bestScore;
            clock_t end = clock();
            double tcpu = double(end-start)/CLOCKS_PER_SEC;
            big totNodes = nodes;
            big usedNodes = totNodes-startNodes;
#ifdef CalculatePV
            string PV = PVprint(PVlines[0]);
#else
            string PV = bestMove.to_str().c_str();
#endif
            if(verbose){
                if(idMove == order.nbMoves)
                    printf("info depth %d score %s nodes %ld nps %d time %d pv %s string branching factor %.3f first cutoff %.3f\n", depth+1, scoreToStr(bestScore).c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), PV.c_str(), (double)usedNodes/lastNodes, (double)nbFirstCutoff/nbCutoff);
                else if(idMove)printf("info depth %d score %s nodes %ld nps %d time %d pv %s string %d/%d moves\n", depth+1, scoreToStr(bestScore).c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), PV.c_str(), idMove, order.nbMoves);
                fflush(stdout);
            }
            if(abs(bestScore) >= MAXIMUM-maxDepth && idMove == order.nbMoves){//checkmate found, stop the thread
                pair<int, Move> result;
                int searchDepth = MAXIMUM-abs(bestScore)+2;
                this->hardBound = mateHardBound ? hardBound : softBound;
                if(bestScore >= MAXIMUM-maxDepth){//we can mate our opponent
                    result = MaterSearch<limitWay>(state, searchDepth, lastChange, actDepth);
                }else{//we are on the path to be mated
                    result = MatedSearch<limitWay>(state, searchDepth, lastChange, actDepth);
                }
                clock_t end = clock();
                double tcpu = double(end-start)/CLOCKS_PER_SEC;
                if(result.second.moveInfo != nullMove.moveInfo){
                    int sign = bestScore < 0 ? -1 : 1;
                    bestScore = (abs(bestScore)+result.first)*sign;
                    if(verbose)
                        printf("info depth %d time %d score %s pv %s\n", searchDepth, (int)(tcpu*1000), scoreToStr(bestScore).c_str(), result.second.to_str().c_str());
                }
                if(result.second.moveInfo != nullMove.moveInfo)//have enough time to find the mate
                    bestMove = result.second;
                if(bestScore < 0)
                    lastScore = -INF-1;
                else lastScore = INF+1;
                break;
            }
            lastNodes = usedNodes;
            if(limitWay == 1 && nodes > softBound)break;
            if(limitWay == 0 && getElapsedTime() > softBoundTime)break;
        }
        for(unsigned long i=0; i<movesFromRoot.size(); i++)
            state.undoLastMove();
        return {bestMove, lastScore};
    }
    int testQuiescenceSearch(GameState& state){
        nodes = 0;
        clock_t start=clock();
        int score = quiescenceSearch<false>(state, -INF, INF);
        clock_t end = clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("speed: %d; Qnodes:%d score %s\n\n", (int)(nodes/tcpu), nodes, scoreToStr(score).c_str());
        return 0;
    }

    void clear(){
        transposition.clear();
        QTT.clear();
        history.init();
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
        if(lastCall != MAX_BIG)return lastCall;
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
            printf("%s: %ld (%d/%d %.2fs => %.0f n/s)\n", moves[i].to_str().c_str(), nbNodes, i+1, nbMoves, tcpu, (visitedNodes-startVisitedNodes)/tcpu);
            fflush(stdout);
            count += nbNodes;
        }
        tt.push({state.zobristHash, count, depth});
        clock_t end=clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        printf("%.3f : %.3f nps %ld visited nodes\n", tcpu, visitedNodes/tcpu, visitedNodes);
        fflush(stdout);
        tt.clearMem();
        return count;
    }
    void reinit(size_t count){
        space = count;
    }
};

#endif