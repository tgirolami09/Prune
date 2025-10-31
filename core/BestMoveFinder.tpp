#include "BestMoveFinder.hpp"
#include <chrono>
#include <cmath>
#include <ctime>
#include <string>

template<int limitWay, bool isPV>
int BestMoveFinder::quiescenceSearch(GameState& state, int alpha, int beta, int relDepth){
    if(!running)return 0;
    if(limitWay == 0 && (nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
    if(eval.isInsufficientMaterial())return 0;
    nodes++;
    if(relDepth > seldepth)seldepth = relDepth;
    dbyte hint;
    if(isPV)
        hint = transposition.getMove(state);
    else{
        int lastEval=transposition.get_eval(state, alpha, beta, 0, hint);
        if(lastEval != INVALID)
            return lastEval;
    }
    int staticEval = eval.getScore(state.friendlyColor());
    if(staticEval >= beta){
        transposition.push(state, staticEval, LOWERBOUND, nullMove, 0);
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
    Move bestCapture;
    for(int i=0; i<order.nbMoves; i++){
        Move capture = order.pop_max();
        state.playMove(capture);//don't care about repetition
        eval.playMove(capture, !state.friendlyColor());
        int score = -quiescenceSearch<limitWay, isPV>(state, -beta, -alpha, relDepth+1);
        eval.undoMove(capture, !state.friendlyColor());
        state.undoLastMove();
        if(!running)return 0;
        if(score >= beta){
            nbCutoff++;
            if(i == 0)nbFirstCutoff++;
            transposition.push(state, score, LOWERBOUND, capture, 0);
            return score;
        }
        if(score > bestEval){
            bestEval = score;
            bestCapture = capture;
            if(score > alpha){
                alpha = score;
                typeNode = EXACT;
            }
        }
    }
    transposition.push(state, bestEval, typeNode, bestCapture, 0);
    return bestEval;
}

enum{PVNode=0, CutNode=1, AllNode=-1};
template<int nodeType, int limitWay, bool mateSearch>
inline int BestMoveFinder::Evaluate(GameState& state, int alpha, int beta, int relDepth){
    if constexpr(mateSearch)return eval.getScore(state.friendlyColor());
    int score = quiescenceSearch<limitWay, nodeType==PVNode>(state, alpha, beta, relDepth);
    if constexpr(limitWay == 1)if(nodes > hardBound)running=false;
    return score;
}

template <int nodeType, int limitWay, bool mateSearch, bool isRoot>
int BestMoveFinder::negamax(const int depth, GameState& state, int alpha, const int beta, const int lastChange, const int relDepth){
    const int rootDist = relDepth-startRelDepth;
    seldepth = max(seldepth, relDepth);
    if(rootDist >= MAXIMUM-alpha)return MAXIMUM-maxDepth;
    if(MINIMUM+rootDist >= beta)return MINIMUM+rootDist;
    if constexpr(limitWay == 0)if((nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
    if(!running)return 0;
    if(state.rule50_count() >= 100 || eval.isInsufficientMaterial()){
        if constexpr (nodeType == PVNode)beginLine(rootDist);
        return 0;
    }
    int static_eval = eval.getScore(state.friendlyColor());
    if(depth == 0 || (!isRoot && depth == 1 && (static_eval+100 < alpha || static_eval > beta+100))){
        if constexpr(nodeType == PVNode)beginLine(rootDist);
        return Evaluate<nodeType, limitWay, mateSearch>(state, alpha, beta, relDepth);
    }
    nodes++;
    int16_t lastBest = nullMove.moveInfo;
    if constexpr(nodeType != PVNode){
        int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
        if(lastEval != INVALID)
            return fromTT(lastEval, rootDist);
    }else{
        lastBest = transposition.getMove(state);
    }
    ubyte typeNode = UPPERBOUND;
    Order<maxMoves> order;
    bool inCheck=generator.isCheck();
    if constexpr(nodeType != PVNode){
        if(!inCheck){
            if(beta > MINIMUM+maxDepth){
                int margin = 150*depth;
                if(static_eval >= beta+margin)
                    return Evaluate<nodeType, limitWay, mateSearch>(state, alpha, beta, relDepth);
            }
            int r = 3;
            if(depth >= r && !eval.isOnlyPawns() && static_eval >= beta){
                state.playNullMove();
                generator.initDangers(state);
                int v = -negamax<CutNode, limitWay, mateSearch>(depth-r, state, -beta, -beta+1, lastChange, relDepth+1);
                state.undoNullMove();
                if(v >= beta)return v;
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
        state.playMove(order.moves[0]);
        if(state.twofold()){
            state.undoLastMove();
            if constexpr(nodeType == PVNode)beginLineMove(rootDist, order.moves[0]);
            return MIDDLE;
        }
        eval.playMove(order.moves[0], !state.friendlyColor());
        generator.initDangers(state);
        int sc = -negamax<-nodeType, limitWay, mateSearch>(depth, state, -beta, -alpha, lastChange, relDepth+1);
        eval.undoMove(order.moves[0], !state.friendlyColor());
        state.undoLastMove();
        if constexpr(nodeType == PVNode)transfer(rootDist, order.moves[0]);
        return sc;
    }
    order.init(state.friendlyColor(), lastBest, getPVMove(rootDist), history, relDepth, state, generator, depth > 5);
    Move bestMove;
    int bestScore = -INF;
    for(int rankMove=0; rankMove<order.nbMoves; rankMove++){
        Move curMove = order.pop_max();
        if(isRoot && verbose && getElapsedTime() >= chrono::milliseconds{10000}){
            printf("info depth %d currmove %s currmovenumber %d nodes %d\n", depth+1, curMove.to_str().c_str(), rankMove+1, nodes);
        }
        int score;
        state.playMove(curMove);
        int newLastChange = lastChange;
        if(curMove.isChanger())
            newLastChange = relDepth;
        bool isDraw = false;
        if(state.twofold()){
            score = MIDDLE;
            isDraw = true;
        }else{
            eval.playMove(curMove, !state.friendlyColor());
            bool inCheckPos = generator.initDangers(state);
            int reductionDepth = 1;
            if(inCheckPos){
                reductionDepth--;
            }
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
        state.undoLastMove();
        if(!running)return bestScore;
        if(score >= beta){ //no need to copy the pv, because it will fail low on the parent
            transposition.push(state, absoluteScore(score, rootDist), LOWERBOUND, curMove, depth);
            nbCutoff++;
            if(isRoot)rootBestMove=curMove;
            if(rankMove == 0)nbFirstCutoff++;
            history.addKiller(curMove, depth, relDepth, state.friendlyColor(), state.getLastMove());
            return score;
        }
        if(score > alpha){
            if(isRoot)rootBestMove = curMove;
            alpha = score;
            typeNode=EXACT;
            bestMove = curMove;
            if constexpr(nodeType == PVNode){
                if(isDraw)beginLineMove(rootDist, curMove);
                else transfer(rootDist, curMove);
            }
        }
        if(score > bestScore)bestScore = score;
    }
    if constexpr(nodeType==CutNode)if(bestScore == alpha)
        return bestScore;
    if(!isRoot || typeNode != UPPERBOUND){
        transposition.push(state, absoluteScore(bestScore, rootDist), typeNode, bestMove, depth);
    }
    return bestScore;
}

template <int limitWay>
bestMoveResponse BestMoveFinder::bestMove(GameState& state, int softBound, int hardBound, vector<Move> movesFromRoot, bool verbose, bool mateHardBound){
    this->verbose = verbose;
    startSearch = timeMesure::now();
    hardBoundTime = chrono::milliseconds{hardBound};
    chrono::milliseconds softBoundTime{softBound};
    vector<depthInfo> allInfos;
    int actDepth=0;
    int lastChange = 0;
    for(Move move:movesFromRoot){
        move = state.playPartialMove(move);
        if(move.isChanger())
            lastChange = actDepth;
        actDepth++;
    }
    bool moveInTable = false;
    Move bookMove = findPolyglot(state,moveInTable,book);
    bool inCheck;
    Order<maxMoves> order;
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
            return make_tuple(bookMove, nullMove, INF, vector<depthInfo>());
        }else if(verbose){
            printf("bad move find in table %s (in %s)\n", bookMove.to_str().c_str(), state.toFen().c_str());
        }
    }
    if(order.nbMoves == 0){
        if(inCheck){
            return make_tuple(nullMove, nullMove, MINIMUM, vector<depthInfo>());
        }else{
            return make_tuple(nullMove, nullMove, 0, vector<depthInfo>());
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
        return make_tuple(order.moves[0], nullMove, INF, vector<depthInfo>(0));
    }
    if(verbose){
        printf("info string use a tt of %d entries (%" PRId64 " MB) (%" PRId64 "B by entry)\n", transposition.modulo, transposition.modulo*sizeof(infoScore)*2/hashMul, sizeof(infoScore));
    }
    Move bestMove=nullMove;
    nodes = 0;
    nbCutoff = nbFirstCutoff = 0;
    clock_t start=clock();
    big lastNodes = 1;
    int lastScore = eval.getScore(state.friendlyColor());
    order.init(state.friendlyColor(), nullMove.moveInfo, nullMove.moveInfo, history, 0, state, generator);
    int instability1side = 0;
    int instability2side = 1;
    Move ponderMove=nullMove;
    startRelDepth = actDepth-1;
    for(int depth=1; depth<depthMax && running; depth++){
        int deltaUp = 5<<(1+instability2side);
        int deltaDown = 5<<(1+instability2side);
        seldepth = 0;
        if(abs(lastScore) > MAXIMUM-maxDepth)
            deltaDown = 1;
        int startNodes = nodes;
        int bestScore;
        Move finalBestMove=bestMove;
        int countUp = 0, countDown=0;
        do{
            int alpha = lastScore-deltaDown;
            int beta = lastScore+deltaUp;
            rootBestMove = nullMove;
            generator.initDangers(state);
            if(abs(lastScore) > MAXIMUM-maxDepth) //is a mate score ?
                bestScore = negamax<PVNode, limitWay, true , true>(depth, state, alpha, beta, lastChange, actDepth);
            else
                bestScore = negamax<PVNode, limitWay, false, true>(depth, state, alpha, beta, lastChange, actDepth);
            bestMove = bestScore != -INF ? rootBestMove : finalBestMove;
            string limit;
            if(bestScore <= alpha){
                deltaDown = max(deltaDown*2, lastScore-bestScore+1);
                limit = "upperbound";
                countDown++;
            }else if(bestScore >= beta){
                deltaUp = max(deltaUp*2, bestScore-lastScore+1);
                finalBestMove = bestMove;
                limit = "lowerbound";
                countUp++;
            }else{
                finalBestMove = bestMove;
                break;
            }
            if(verbose && bestScore != -INF && getElapsedTime() >= chrono::milliseconds{10000}){
                big totNodes = nodes;
                clock_t end = clock();
                double tcpu = double(end-start)/CLOCKS_PER_SEC;
                printf("info depth %d seldepth %d score %s %s nodes %ld nps %d time %d pv %s\n", depth+1, seldepth-startRelDepth, scoreToStr(bestScore).c_str(), limit.c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), finalBestMove.to_str().c_str());
            }
        }while(running);
        instability1side = (instability1side+(countDown-countUp)+1)/2;
        instability2side = (instability2side+min(countDown, countUp)+1)/2;
        bestMove = finalBestMove;
        if(bestScore != -INF)
            lastScore = bestScore;
        transferLastPV();
        clock_t end = clock();
        double tcpu = double(end-start)/CLOCKS_PER_SEC;
        big totNodes = nodes;
        big usedNodes = totNodes-startNodes;
        string PV;
        PV = PVprint(PVlines[0]);
        if(PVlines[0].cmove > 1)
            ponderMove.moveInfo = PVlines[0].argMoves[1];
        if(verbose && bestScore != -INF){
            printf("info depth %d seldepth %d score %s nodes %" PRId64 " nps %d time %d pv %s string branching factor %.3f first cutoff %.3f\n", depth+1, seldepth-startRelDepth, scoreToStr(bestScore).c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), PV.c_str(), (double)usedNodes/lastNodes, (double)nbFirstCutoff/nbCutoff);
            fflush(stdout);
        }
        if(running)
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
    return make_tuple(bestMove, ponderMove, lastScore, allInfos);
}