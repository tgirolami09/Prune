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

static int fromTT(int score, int rootDist){
    if(score < MINIMUM+maxDepth)
        return score + rootDist;
    else if(score > MAXIMUM-maxDepth)
        return score - rootDist;
    return score;
}

int absoluteScore(int score, int rootDist){
    if(score < MINIMUM+maxDepth)
        return score - rootDist;
    else if(score > MAXIMUM-maxDepth)
        return score + rootDist;
    return score;

}

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
    BestMoveFinder(int memory, bool mute=false):transposition(memory*2/3), QTT(memory/3){
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
    int seldepth;
    big isInSearch[1000];
    template<int limitWay>
    int quiescenceSearch(GameState& state, int alpha, int beta, int relDepth){
        if(limitWay <= 1 && !running)return 0;
        if(limitWay == 0 && (nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        if(eval.isInsufficientMaterial())return 0;
        nodes++;
        if(relDepth > seldepth)seldepth = relDepth;
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
        generator.initDangers(state);
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions, true);
        order.init(state.friendlyColor(), nullMove.moveInfo, nullMove.moveInfo, history, -1, state, generator, false);
        for(int i=0; i<order.nbMoves; i++){
            Move capture = order.pop_max();
            state.playMove<false, false>(capture);//don't care about repetition
            eval.playMove(capture, !state.friendlyColor());
            int score = -quiescenceSearch<limitWay>(state, -beta, -alpha, relDepth+1);
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
    LINE lastPV;
    void transferLastPV(){
        lastPV.cmove = PVlines[0].cmove;
        for(int i=0; i<PVlines[0].cmove; i++)
            lastPV.argMoves[i] = PVlines[0].argMoves[i];
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

    int16_t getPVMove(int relDepth){
        if(lastPV.cmove < relDepth)
            return lastPV.argMoves[relDepth];
        return nullMove.moveInfo;
    }

    enum{PVNode=0, CutNode=1, AllNode=-1};
    template <int nodeType, int limitWay, bool mateSearch>
    int negamax(const int depth, GameState& state, int alpha, int beta, const int lastChange, const int relDepth){
        const int rootDist = relDepth-startRelDepth;
        seldepth = max(seldepth, relDepth);
        if(rootDist >= MAXIMUM-alpha)return MAXIMUM-maxDepth;
        if(MINIMUM+rootDist >= beta)return MINIMUM+rootDist;
        if constexpr(limitWay <= 1)if(!running)return 0;
        if constexpr(limitWay == 0)if((nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
        if(relDepth-lastChange >= 100 || eval.isInsufficientMaterial()){
            if constexpr (nodeType == PVNode)beginLine(rootDist);
            return 0;
        }
        int static_eval = eval.getScore(state.friendlyColor());
        if(depth == 0 || (depth == 1 && (static_eval+100 < alpha || static_eval > beta+100))){
            if constexpr(nodeType == PVNode)beginLine(rootDist);
            if(mateSearch)return static_eval;
            int score = quiescenceSearch<limitWay>(state, alpha, beta, relDepth);
            if constexpr(limitWay == 1)if(nodes > hardBound)running=false;
            return score;
        }
        nodes++;
        int16_t lastBest = nullMove.moveInfo;
        if constexpr(nodeType != PVNode){
            int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
            if(lastEval != INVALID){
                beginLine(rootDist);
                return fromTT(lastEval, rootDist);
            }
        }else{
            lastBest = transposition.getMove(state);
        }
        ubyte typeNode = UPPERBOUND;
        Order<maxMoves> order;
        bool inCheck=generator.isCheck();
        if(!inCheck){
            if constexpr(nodeType != PVNode){
                if(!mateSearch){
                    int margin = 150*depth;
                    if(static_eval >= beta+margin){
                        int score = quiescenceSearch<limitWay>(state, alpha, beta, relDepth);
                        if constexpr(limitWay == 1)if(nodes > hardBound)running=false;
                        return score;
                    }
                }
                int r = 3;
                if(depth >= r && !eval.isOnlyPawns() && static_eval >= beta){
                    state.playNullMove();
                    generator.initDangers(state);
                    int v = -negamax<CutNode, limitWay, mateSearch>(depth-r, state, -beta, -beta+1, lastChange, relDepth+1);
                    state.undoNullMove();
                    if(v >= beta)return beta;
                    generator.initDangers(state);
                }
            }
        }
        order.nbMoves = generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
        if(order.nbMoves == 0){
            int score;
            if(inCheck)
                score = MINIMUM+rootDist;
            else score = MIDDLE;
            if constexpr(nodeType == PVNode)beginLine(rootDist);
            return score;
        }
        if(order.nbMoves == 1){
            state.playMove<false, false>(order.moves[0]);
            eval.playMove(order.moves[0], !state.friendlyColor());
            generator.initDangers(state);
            int sc = -negamax<-nodeType, limitWay, mateSearch>(depth, state, -beta, -alpha, lastChange, relDepth+1);
            eval.undoMove(order.moves[0], !state.friendlyColor());
            state.undoLastMove<false>();
            if constexpr(nodeType != PVNode)transfer(rootDist, order.moves[0]);
            return sc;
        }
        order.init(state.friendlyColor(), lastBest, getPVMove(rootDist), history, relDepth, state, generator, depth > 5);
        Move bestMove;
        int bestScore = -INF;
        for(int rankMove=0; rankMove<order.nbMoves; rankMove++){
            Move curMove = order.pop_max();
            int score;
            state.playMove<false, false>(curMove);
            int newLastChange = lastChange;
            if(curMove.isChanger())
                newLastChange = relDepth;
            ubyte usableDepth = isRepet(state.zobristHash, newLastChange, relDepth);
            bool isDraw = false;
            if(usableDepth != (ubyte)-1){
                score = MIDDLE;
                isDraw = true;
            }else{
                eval.playMove(curMove, !state.friendlyColor());
                bool inCheckPos = generator.initDangers(state);
                int reductionDepth = 1;
                if(inCheckPos){
                    reductionDepth--;
                }
                setElement(state.zobristHash, relDepth);
                if(rankMove > 0){
                    int addRedDepth = 0;
                    if(rankMove > 3 && depth > 3 && !curMove.isTactical()){
                        addRedDepth = static_cast<int>(0.9 + log(depth) * log(rankMove) / 3);
                        if(mateSearch && inCheckPos)
                            addRedDepth--;
                    }
                    score = -negamax<((nodeType == CutNode)?AllNode:CutNode), limitWay, mateSearch>(depth-reductionDepth-addRedDepth, state, -alpha-1, -alpha, newLastChange, relDepth+1);
                    bool fullSearch = false;
                    if((score > alpha && score < beta) || (nodeType == PVNode && score == beta && beta == alpha+1)){
                        fullSearch = true;
                    }
                    if(addRedDepth && score >= beta)
                        fullSearch = true;
                    if(fullSearch){
                        generator.initDangers(state);
                        score = -negamax<nodeType, limitWay, mateSearch>(depth-reductionDepth, state, -beta, -alpha, newLastChange, relDepth+1);
                    }
                }else
                    score = -negamax<-nodeType, limitWay, mateSearch>(depth-reductionDepth, state, -beta, -alpha, newLastChange, relDepth+1);
                eval.undoMove(curMove, !state.friendlyColor());
            }
            state.undoLastMove<false>();
            if constexpr(limitWay <= 1)if(!running)return 0;
            if(score >= beta){ //no need to copy the pv, because it will fail low on the parent
                transposition.push(state, absoluteScore(score, rootDist), LOWERBOUND, curMove, depth);
                nbCutoff++;
                if(rankMove == 0)nbFirstCutoff++;
                history.addKiller(curMove, depth, relDepth, state.friendlyColor(), state.getLastMove());
                return score;
            }
            if(score > alpha){
                alpha = score;
                typeNode=EXACT;
                if constexpr(nodeType == PVNode){
                    if(isDraw)beginLineMove(rootDist, curMove);
                    else transfer(rootDist, curMove);
                }
            }
            if(score > bestScore)bestScore = score;
        }
        if constexpr(nodeType==CutNode)if(bestScore == alpha)
            return bestScore;
        transposition.push(state, absoluteScore(bestScore, rootDist), typeNode, bestMove, depth);
        return bestScore;
    }
    template<int limitWay, bool mateSearch>
    Move bestMoveClipped(int depth, GameState& state, int alpha, int beta, int& bestScore, Move lastBest, int& idMove, RootOrder& order, int actDepth, int lastChange){
        bestScore = -INF;
        Move bestMove = nullMove;
        order.reinit(lastBest.moveInfo);
        startRelDepth = actDepth-1;
        for(idMove=0; idMove < order.nbMoves; idMove++){
            Move curMove = order.pop_max();
            //printf("%s\n", curMove.to_str().c_str());
            int startNodes = nodes;
            int score;
            int curLastChange = lastChange;
            bool isDraw = false;
            if(curMove.isChanger())
                curLastChange = actDepth;
            if(state.playMove<false>(curMove) > 1){
                score = MIDDLE;
                isDraw = true;
            }else{
                eval.playMove(curMove, !state.friendlyColor());
                setElement(state.zobristHash, actDepth);
                generator.initDangers(state);
                score = -negamax<PVNode, limitWay, mateSearch>(depth, state, -beta, -alpha, curLastChange, actDepth+1);
                eval.undoMove(curMove, !state.friendlyColor());
            }
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
                if(isDraw)beginLineMove(1, curMove);
                else transfer(1, curMove);
            }else if(score > bestScore){
                bestMove = curMove;
                bestScore = score;
            }
        }
        return bestMove;
    }

public:
    template <int limitWay=0>
    tuple<Move, int, vector<depthInfo>> bestMove(GameState& state, int softBound, int hardBound, vector<Move> movesFromRoot, bool verbose=true, bool mateHardBound=true){
        startSearch = timeMesure::now();
        hardBoundTime = chrono::milliseconds{hardBound};
        chrono::milliseconds softBoundTime{softBound};
        vector<depthInfo> allInfos;
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
        generator.initDangers(state);
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
                return make_tuple(bookMove, INF, vector<depthInfo>());
            }else if(verbose){
                printf("bad move find in table %s (in %s)\n", bookMove.to_str().c_str(), state.toFen().c_str());
            }
        }
        if(order.nbMoves == 0){
            if(inCheck){
                return make_tuple(nullMove, MINIMUM, vector<depthInfo>());
            }else{
                return make_tuple(nullMove, 0, vector<depthInfo>());
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
            return make_tuple(order.moves[0], INF, vector<depthInfo>(0));
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
            seldepth = 0;
            if(abs(lastScore) > MAXIMUM-maxDepth)
                deltaDown = 1;
            int startNodes = nodes;
            int idMove;
            int bestScore;
            do{
                int alpha = lastScore-deltaDown;
                int beta = lastScore+deltaUp;
                if(abs(lastScore) > MAXIMUM-maxDepth)
                    bestMove = bestMoveClipped<limitWay, true>(depth, state, alpha, beta, bestScore, bestMove, idMove, order, actDepth, lastChange);
                else
                    bestMove = bestMoveClipped<limitWay, false>(depth, state, alpha, beta, bestScore, bestMove, idMove, order, actDepth, lastChange);

                if(bestScore <= alpha)deltaDown = max(deltaDown*2, lastScore-bestScore+1);
                else if(bestScore >= beta)deltaUp = max(deltaUp*2, bestScore-lastScore+1);
                else break;
            }while(running);
            if(idMove)
                lastScore = bestScore;
            transferLastPV();
            clock_t end = clock();
            double tcpu = double(end-start)/CLOCKS_PER_SEC;
            big totNodes = nodes;
            big usedNodes = totNodes-startNodes;
            string PV;
            PV = PVprint(PVlines[0]);
            PV = bestMove.to_str().c_str();
            if(verbose){
                if(idMove == order.nbMoves)
                    printf("info depth %d seldepth %d score %s nodes %ld nps %d time %d pv %s string branching factor %.3f first cutoff %.3f\n", depth+1, seldepth-startRelDepth, scoreToStr(bestScore).c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), PV.c_str(), (double)usedNodes/lastNodes, (double)nbFirstCutoff/nbCutoff);
                else if(idMove)printf("info depth %d seldepth %d score %s nodes %ld nps %d time %d pv %s string %d/%d moves\n", depth+1, seldepth-startRelDepth, scoreToStr(bestScore).c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), PV.c_str(), idMove, order.nbMoves);
                fflush(stdout);
            }
            if(idMove == order.nbMoves)
                allInfos.push_back({nodes, (int)(tcpu*1000), (int)(totNodes/tcpu), depth+1, seldepth-startRelDepth, bestScore});
            if(abs(bestScore) > MAXIMUM-maxDepth && mateHardBound){
                softBound = hardBound;
                softBoundTime = hardBoundTime;
            }
            lastNodes = usedNodes;
            if(limitWay == 1 && nodes > softBound)break;
            if(limitWay == 0 && getElapsedTime() > softBoundTime)break;
        }
        for(unsigned long i=0; i<movesFromRoot.size(); i++)
            state.undoLastMove();
        return make_tuple(bestMove, lastScore, allInfos);
    }
    int testQuiescenceSearch(GameState& state){
        nodes = 0;
        clock_t start=clock();
        int score = quiescenceSearch<false>(state, -INF, INF, 0);
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
        generator.initDangers(state);
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
        generator.initDangers(state);
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