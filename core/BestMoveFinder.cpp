#include "BestMoveFinder.hpp"
#include "Const.hpp"
#include "Evaluator.hpp"
#include "GameState.hpp"
#include "Move.hpp"
#include "TranspositionTable.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <cassert>

#ifdef DEBUG_MACRO
int nmpVerifAllNode=0,
    nmpVerifCutNode=0,
    nmpVerifPassCutNode=0,
    nmpVerifPassAllNode=0;
#endif

BestMoveFinder::usefull::usefull(const GameState& state):nodes(0), bestMoveNodes(0), seldepth(0), nbCutoff(0), nbFirstCutoff(0), rootBest(nullMove), mainThread(true){
    eval.init(state);
    generator.initDangers(state);
    history.init();
    correctionHistory.reset();
}
BestMoveFinder::usefull::usefull():nodes(0), bestMoveNodes(0), seldepth(0), nbCutoff(0), nbFirstCutoff(0), rootBest(nullMove), mainThread(true){}
void BestMoveFinder::usefull::reinit(const GameState& state){
    nodes = 0;
    bestMoveNodes = 0;
    seldepth = 0;
    nbCutoff = 0;
    nbFirstCutoff = 0;
    rootBest = nullMove;
    mainThread = true;
    eval.init(state);
    generator.initDangers(state);
}

int compScoreMove(const void* a, const void*b){
    int first = ((MoveScore*)a)->first;
    int second = ((MoveScore*)b)->first;
    return second-first; //https://stackoverflow.com/questions/8115624/using-quick-sort-in-c-to-sort-in-reverse-direction-descending
}

int fromTT(int score, int rootDist){
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

string scoreToStr(int score){
    if(score > MAXIMUM-maxDepth)
        return ((string)"mate ")+to_string((MAXIMUM-score+1)/2);
    if(score < MINIMUM+maxDepth)
        return ((string)"mate ")+to_string((-(MAXIMUM+score))/2);
    return "cp "+to_string(score);
}

//Class to find the best in a situation

BestMoveFinder::BestMoveFinder(int memory, bool mute):transposition(memory){
    book = load_book("book.bin", mute);
    helperThreads = NULL;
}

void BestMoveFinder::clear_helpers(){
    if(helperThreads == NULL)return;
    smp_end = true;
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].launch(0, 0, 0, 0, 0);
        helperThreads[i].t.join();
    }
    delete[] helperThreads;
}

BestMoveFinder::~BestMoveFinder(){
    clear_helpers();
}

void BestMoveFinder::setThreads(int nT){
    if(nT == nbThreads)return;
    if(nbThreads > 1){
        clear_helpers();
    }
    delete[] helperThreads;
    if(nT == 1){
        helperThreads = NULL;
    }else{
        smp_end = false;
        helperThreads = new HelperThread[nT-1];
        for(int i=0; i<nT-1; i++){
            helperThreads[i].t = thread(&BestMoveFinder::launchSMP, this, i);
        }
    }
}

void BestMoveFinder::stop(){
    running = false;
}
chrono::nanoseconds BestMoveFinder::getElapsedTime(){
    return timeMesure::now()-startSearch;
}

string BestMoveFinder::usefull::PVprint(LINE pvLine){
    string resLine = "";
    for(int i=0; i<pvLine.cmove; i++){
        Move mv;
        mv.moveInfo = pvLine.argMoves[i];
        if(i != 0)resLine += " ";
        resLine += mv.to_str();
    }
    return resLine;
}

void BestMoveFinder::usefull::transfer(int relDepth, Move move){
    PVlines[relDepth-1].argMoves[0] = move.moveInfo;
    memcpy(&PVlines[relDepth-1].argMoves[1], PVlines[relDepth].argMoves, PVlines[relDepth].cmove * sizeof(int16_t));
    PVlines[relDepth-1].cmove = PVlines[relDepth].cmove+1;
}
void BestMoveFinder::usefull::beginLine(int relDepth){
    PVlines[relDepth-1].cmove = 0;
}

void BestMoveFinder::usefull::beginLineMove(int relDepth, Move move){
    PVlines[relDepth-1].argMoves[0] = move.moveInfo;
    PVlines[relDepth-1].cmove = 1;
}

void BestMoveFinder::usefull::resetLines(){
    for(int i=0; i<maxDepth; i++){
        PVlines[i].cmove = 0;
    }
}

void BestMoveFinder::HelperThread::launch(int _depth, int _alpha, int _beta, int _relDepth, int _limitWay){
    depth = _depth;
    alpha = _alpha;
    beta = _beta;
    relDepth = _relDepth;
    limitWay = _limitWay;
    ans = 0;
    {
        lock_guard<mutex> lock(mtx);
        running = true;
    }
    cv.notify_one();
}

void BestMoveFinder::HelperThread::wait_thread(){
    unique_lock<mutex> lock(mtx);
    cv.wait(lock, [this]{return !running;});
}

template<int limitWay, bool isPV, bool isCalc>
int BestMoveFinder::quiescenceSearch(usefull& ss, GameState& state, int alpha, int beta, int relDepth){
    if(!running || smp_abort)return 0;
    if(limitWay == 0 && (ss.nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
    if(ss.eval.isInsufficientMaterial())return 0;
    ss.nodes++;
    if(relDepth > ss.seldepth)ss.seldepth = relDepth;
    dbyte hint;
    const int rootDist = relDepth-startRelDepth;
    if(isPV)
        hint = transposition.getMove(state);
    else{
        int lastEval=transposition.get_eval(state, alpha, beta, 0, hint);
        if(lastEval != INVALID)
            return fromTT(lastEval, rootDist);
    }
    if(rootDist >= maxDepth)return ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    int& staticEval = ss.stack[rootDist].static_score;
    if(!isCalc)
        staticEval = ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    int typeNode = UPPERBOUND;
    bool testCheck = ss.generator.initDangers(state);
    int bestEval = MINIMUM;
    if(!testCheck){
        if(staticEval >= beta){
            transposition.push(state, staticEval, LOWERBOUND, nullMove, 0);
            return staticEval;
        }
        if(staticEval > alpha){
            alpha = staticEval;
            typeNode = EXACT;
        }
        bestEval = staticEval;
    }
    Order& order = ss.stack[rootDist].order;
    bool inCheck;
    order.nbMoves = ss.generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions, !testCheck);
    if(order.nbMoves == 0 && testCheck){
        return MINIMUM+rootDist;
    }
    order.init(state.friendlyColor(), nullMove.moveInfo, ss.history, rootDist, state);
    Move bestCapture;
    for(int i=0; i<order.nbMoves; i++){
        int flag;
        Move capture = order.pop_max(flag);
        if(bestEval >= MINIMUM+maxDepth){
            if(capture.isTactical() && !(flag&1))continue;
            else if(!capture.isTactical())continue;
        }
        state.playMove(capture);//don't care about repetition
        ss.eval.playMove(capture, !state.friendlyColor());
        int score = -quiescenceSearch<limitWay, isPV, false>(ss, state, -beta, -alpha, relDepth+1);
        ss.eval.undoMove(capture, !state.friendlyColor());
        state.undoLastMove();
        if(!running || smp_abort)return 0;
        if(score >= beta){
            ss.nbCutoff++;
            if(i == 0)ss.nbFirstCutoff++;
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
    transposition.push(state, absoluteScore(bestEval, rootDist), typeNode, bestCapture, 0);
    return bestEval;
}

template<int nodeType, int limitWay, bool mateSearch>
inline int BestMoveFinder::Evaluate(usefull& ss, GameState& state, int alpha, int beta, int relDepth){
    if constexpr(mateSearch)return ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    int score = quiescenceSearch<limitWay, nodeType==PVNode, true>(ss, state, alpha, beta, relDepth);
    if constexpr(limitWay == 1)if(ss.nodes > hardBound)running=false;
    return score;
}

template <int nodeType, int limitWay, bool mateSearch, bool isRoot>
int BestMoveFinder::negamax(usefull& ss, int depth, GameState& state, int alpha, const int beta, const int relDepth, const int16_t excludedMove){
    const int rootDist = relDepth-startRelDepth;
    if(rootDist >= maxDepth)return ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    ss.seldepth = max(ss.seldepth, relDepth);
    transposition.prefetch(state);
    if(rootDist >= MAXIMUM-alpha)return MAXIMUM-maxDepth;
    if(MINIMUM+rootDist >= beta)return MINIMUM+rootDist;
    if constexpr(limitWay == 0)if((ss.nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
    if(!running || smp_abort)return 0;
    if(state.rule50_count() >= 100 || ss.eval.isInsufficientMaterial()){
        if constexpr (nodeType == PVNode)ss.beginLine(rootDist);
        return 0;
    }
    int& static_eval = ss.stack[rootDist].static_score;
    static_eval = ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    if(depth == 0 || (!isRoot && depth == 1 && (static_eval+100 < alpha || static_eval > beta+100))){
        if constexpr(nodeType == PVNode)ss.beginLine(rootDist);
        return Evaluate<nodeType, limitWay, mateSearch>(ss, state, alpha, beta, relDepth);
    }
    ss.nodes++;
    int16_t lastBest = nullMove.moveInfo;
    if(excludedMove == nullMove.moveInfo){
        if constexpr(nodeType != PVNode){
            int lastEval = transposition.get_eval(state, alpha, beta, depth, lastBest);
            if(lastEval != INVALID)
                return fromTT(lastEval, rootDist);
        }else{
            lastBest = transposition.getMove(state);
        }
    }
    bool ttHit;
    infoScore ttEntry = transposition.getEntry(state, ttHit);
    ubyte typeNode = UPPERBOUND;
    Order& order = ss.stack[rootDist].order;
    bool inCheck=ss.generator.isCheck();
    bool improving = false;
    if((!ttHit || ttEntry.depth+3 < depth) && depth >= 3 && nodeType != AllNode && excludedMove == nullMove.moveInfo)depth--;
    if(rootDist > 2)
        improving = ss.stack[rootDist-2].static_score < static_eval && excludedMove == nullMove.moveInfo;
    if constexpr(nodeType != PVNode){
        if(!inCheck && excludedMove == nullMove.moveInfo && beta > MINIMUM+maxDepth){
            int margin;
            if(improving)
                margin = 120*depth;
            else
                margin = 150*depth;
            if(static_eval >= beta+margin)
                return Evaluate<nodeType, limitWay, mateSearch>(ss, state, alpha, beta, relDepth);
            int r = depth/4+3;
            if(rootDist >= ss.min_nmp_ply && depth >= r && !ss.eval.isOnlyPawns() && static_eval >= beta){
                state.playNullMove();
                ss.generator.initDangers(state);
                int v = -negamax<CutNode, limitWay, mateSearch>(ss, depth-r, state, -beta, -beta+1, relDepth+1);
                state.undoNullMove();
                if(v >= beta){
                    if(depth <= 10 || ss.min_nmp_ply != 0)
                        return v;
#ifdef DEBUG_MACRO
                    if(nodeType == CutNode)
                        nmpVerifCutNode++;
                    else
                        nmpVerifAllNode++;
#endif
                    ss.min_nmp_ply = rootDist+r;
                    ss.generator.initDangers(state);
                    v = negamax<CutNode, limitWay, mateSearch>(ss, depth-r, state, beta-1, beta, relDepth);
                    ss.min_nmp_ply = 0;
                    if(v >= beta){
#ifdef DEBUG_MACRO
                    if(nodeType == CutNode)
                        nmpVerifPassCutNode++;
                    else
                        nmpVerifPassAllNode++;
#endif
                        return v;
                    };
                };
                ss.generator.initDangers(state);
            }
        }
    }
    int firstMoveExtension = 0;
    if(!isRoot && ttHit && ttEntry.depth + 3 >= depth && ttEntry.typeNode != UPPERBOUND && depth >= 6 && excludedMove == nullMove.moveInfo && abs(ttEntry.score) < MAXIMUM-maxDepth){
        int goal = ttEntry.score - depth;
        int score = negamax<CutNode, limitWay, mateSearch>(ss, (depth-1)/2, state, goal-1, goal, relDepth, ttEntry.bestMoveInfo);
        if(score < goal){
            firstMoveExtension++;
            if(nodeType != PVNode && score <= goal-20)
                firstMoveExtension++;
        }else if(ttEntry.score >= beta){
            firstMoveExtension--;
        }
        ss.generator.initDangers(state);
    }
    order.nbMoves = ss.generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
    if(order.nbMoves == 0){
        int score;
        if(inCheck)
            score = MINIMUM+rootDist;
        else score = MIDDLE;
        if constexpr(nodeType == PVNode)ss.beginLine(rootDist);
        return score;
    }
    if(order.nbMoves == 1){
        if(isRoot)
            ss.rootBest = order.moves[0];
        state.playMove(order.moves[0]);
        if(state.twofold()){
            state.undoLastMove();
            if constexpr(nodeType == PVNode)ss.beginLineMove(rootDist, order.moves[0]);
            return MIDDLE;
        }
        ss.eval.playMove(order.moves[0], !state.friendlyColor());
        ss.generator.initDangers(state);
        int sc = -negamax<-nodeType, limitWay, mateSearch>(ss, depth, state, -beta, -alpha, relDepth+1);
        ss.eval.undoMove(order.moves[0], !state.friendlyColor());
        state.undoLastMove();
        if (sc > alpha && sc < beta && nodeType == PVNode)ss.transfer(rootDist, order.moves[0]);
        return sc;
    }
    order.init(state.friendlyColor(), lastBest, ss.history, rootDist, state);
    Move bestMove = nullMove;
    int bestScore = -INF;
    for(int rankMove=0; rankMove<order.nbMoves; rankMove++){
        int flag;
        Move curMove = order.pop_max(flag);
        if(excludedMove == curMove.moveInfo)continue;
        sbig startNodes = ss.nodes;
        if(isRoot && verbose && ss.mainThread && DEBUG){
            printf("info depth %d currmove %s currmovenumber %d nodes %" PRId64 " string flag %d\n", depth, curMove.to_str().c_str(), rankMove+1, ss.nodes, flag);
            fflush(stdout);
        }
        if(!curMove.isTactical() && rankMove > depth*depth*4+4 && bestScore >= MINIMUM+maxDepth)continue;
        int moveHistory = curMove.isTactical() ? 0 : (order.scores[rankMove]>=KILLER_ADVANTAGE-maxHistory ? maxHistory : order.scores[rankMove]);
        if(moveHistory < -100*depth && rankMove > 1 && bestScore >= MINIMUM+maxDepth)
            continue;
        int futilityValue = static_eval+300+150*depth;
        if(nodeType != PVNode && !curMove.isTactical() && depth <= 5 && !inCheck && futilityValue <= alpha){
            continue;
        }
        int score;
        state.playMove(curMove);
        bool isDraw = false;
        if(state.twofold()){
            score = MIDDLE;
            isDraw = true;
        }else{
            ss.eval.playMove(curMove, !state.friendlyColor());
            bool inCheckPos = ss.generator.initDangers(state);
            int reductionDepth = 1;
            if(inCheckPos && firstMoveExtension == 0){
                reductionDepth--;
            }
            if(rankMove > 0){
                int addRedDepth = 0;
                if(rankMove > 3 && depth > 3 && !curMove.isTactical()){
                    addRedDepth = static_cast<int>((0.9 + log(depth) * log(rankMove) / 3)*1024);
                    if(mateSearch && inCheckPos)
                        addRedDepth -= 1024;
                    addRedDepth -= (moveHistory)*512/maxHistory;
                    addRedDepth /= 1024;
                    addRedDepth = max(addRedDepth, 0);
                }
                score = -negamax<((nodeType == CutNode)?AllNode:CutNode), limitWay, mateSearch>(ss, depth-reductionDepth-addRedDepth, state, -alpha-1, -alpha, relDepth+1);
                bool fullSearch = false;
                if((score > alpha && score < beta) || (nodeType == PVNode && score == beta && beta == alpha+1)){
                    fullSearch = true;
                }
                if(addRedDepth && score >= beta)
                    fullSearch = true;
                if(fullSearch){
                    ss.generator.initDangers(state);
                    score = -negamax<nodeType, limitWay, mateSearch>(ss, depth-reductionDepth, state, -beta, -alpha, relDepth+1);
                }
            }else
                score = -negamax<-nodeType, limitWay, mateSearch>(ss, depth-reductionDepth+firstMoveExtension, state, -beta, -alpha, relDepth+1);
            ss.eval.undoMove(curMove, !state.friendlyColor());
        }
        state.undoLastMove();
        if(!running || smp_abort)return bestScore;
        if(score >= beta){ //no need to copy the pv, because it will fail low on the parent
            transposition.push(state, absoluteScore(score, rootDist), LOWERBOUND, curMove, depth);
            ss.nbCutoff++;
            if(isRoot)ss.rootBest=curMove;
            if(rankMove == 0)ss.nbFirstCutoff++;
            ss.history.addKiller(curMove, depth, rootDist, state.friendlyColor());
            if(!curMove.isTactical()){
                ss.history.negUpdate(order.moves, rankMove, state.friendlyColor(), depth);
                if(score > static_eval && !inCheck)
                    ss.correctionHistory.update(state, score-static_eval, depth);
            }
            return score;
        }
        if(score > alpha){
            if(isRoot){
                ss.rootBest = curMove;
                ss.bestMoveNodes = ss.nodes-startNodes;
            }
            alpha = score;
            typeNode=EXACT;
            bestMove = curMove;
            if constexpr(nodeType == PVNode){
                if(isDraw)ss.beginLineMove(rootDist, curMove);
                else ss.transfer(rootDist, curMove);
            }
        }
        if(score > bestScore)bestScore = score;
    }
    if constexpr(nodeType==CutNode)if(bestScore == alpha)
        return bestScore;
    if((!isRoot || typeNode != UPPERBOUND) && excludedMove == nullMove.moveInfo){
        transposition.push(state, absoluteScore(bestScore, rootDist), typeNode, bestMove, depth);
    }
    if(!inCheck && (!bestMove.isTactical()) && abs(bestScore) < MAXIMUM-maxDepth &&
        (typeNode != UPPERBOUND || bestScore < static_eval)){
        ss.correctionHistory.update(state, bestScore-static_eval, depth);
    }
    return bestScore;
}

template<bool mateSearch>
int BestMoveFinder::launchSearch(int limitWay, HelperThread& ss){
    if(limitWay == 0)
        return negamax<PVNode, 0, mateSearch, true>(ss.local, ss.depth, ss.localState, ss.alpha, ss.beta, ss.relDepth);
    if(limitWay == 1)
        return negamax<PVNode, 1, mateSearch, true>(ss.local, ss.depth, ss.localState, ss.alpha, ss.beta, ss.relDepth);
    else
        return negamax<PVNode, 2, mateSearch, true>(ss.local, ss.depth, ss.localState, ss.alpha, ss.beta, ss.relDepth);
}

void BestMoveFinder::launchSMP(int idThread){
    /*HelperThread& ss = helperThreads[idThread-1];
    ss.local.reinit(ss.localState);
    ss.local.mainThread = false;
    negamax<PVNode, limitWay, mateSearch, true>(ss.local, depth, ss.localState, alpha, beta, relDepth);*/
    HelperThread& ss = helperThreads[idThread];
    ss.running = false;
    while(!smp_end){
        {
            unique_lock<mutex> lock(ss.mtx);
            ss.cv.wait(lock, [&ss]{return ss.running;});
        }
        if(smp_end)return;
        ss.local.reinit(ss.localState);
        ss.local.mainThread = false;
        if(abs(ss.alpha) > MAXIMUM-maxDepth) //is a mate score ?
            ss.ans = launchSearch<true>(ss.limitWay, ss);
        else
            ss.ans = launchSearch<false>(ss.limitWay, ss);
        {
            lock_guard<mutex> lock(ss.mtx);
            ss.running = false;
            ss.cv.notify_one();
        }
    }
}

template <int limitWay>
bestMoveResponse BestMoveFinder::bestMove(GameState& state, TM tm, vector<Move> movesFromRoot, bool _verbose, bool mateHardBound){
    this->verbose = _verbose;
    startSearch = timeMesure::now();
    int actDepth=0;
    for(int i=0; i<nbThreads-1; i++)
        helperThreads[i].localState.fromFen(state.toFen());
    for(Move move:movesFromRoot){
        state.playPartialMove(move);
        for(int i=0; i<nbThreads-1; i++)helperThreads[i].localState.playPartialMove(move);
        actDepth++;
    }
    bestMoveResponse res=goState<limitWay>(state, tm, verbose, mateHardBound, actDepth);
    for(unsigned long i=0; i<movesFromRoot.size(); i++)
        state.undoLastMove();
    return res;
}

template<int limitWay>
bestMoveResponse BestMoveFinder::goState(GameState& state, TM tm, bool _verbose, bool mateHardBound, int actDepth){
    verbose = _verbose;
    hardBoundTime = chrono::milliseconds{tm.hardBound};
    chrono::milliseconds softBoundTime{tm.softBound};
    vector<depthInfo> allInfos;
    bool moveInTable = false;
    Move bookMove = findPolyglot(state,moveInTable,book);
    bool inCheck;
    Order order;
    localSS.reinit(state);
    order.nbMoves = localSS.generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
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
    running = true;
    int depthMax = maxDepth;
    this->hardBound = INT64_MAX;
    if(limitWay == 2){
        depthMax = tm.hardBound;
    }
    if(order.nbMoves == 1 && limitWay == 0){
        running = false;
        return make_tuple(order.moves[0], nullMove, INF, vector<depthInfo>(0));
    }
    if(verbose){
        printf("info string use a tt of %" PRId64 "entries (%" PRId64 " MB) (%" PRId64 "B by entry)\n", transposition.modulo, (big)transposition.modulo*sizeof(infoScore)/hashMul, (big)sizeof(infoScore));
    }
    Move bestMove=nullMove;
    int lastScore = localSS.eval.getScore(state.friendlyColor(), localSS.correctionHistory, state);
    Move ponderMove=nullMove;
    startRelDepth = actDepth-1;
    for(int depth=1; depth<=depthMax && running; depth++){
        int deltaUp = 20;
        int deltaDown = 20;
        localSS.seldepth = 0;
        if(abs(lastScore) > MAXIMUM-maxDepth)
            deltaDown = 1;
        int bestScore;
        Move finalBestMove=bestMove;
        sbig lastUsedNodes = 0;
        string PV;
        do{
            int alpha = lastScore-deltaDown;
            int beta = lastScore+deltaUp;
            localSS.generator.initDangers(state);
            lastUsedNodes = localSS.nodes;
            smp_abort = false;
            for(int i=0; i<nbThreads-1; i++){
                helperThreads[i].launch(depth, alpha, beta, actDepth, limitWay);
            }
            if(abs(lastScore) > MAXIMUM-maxDepth) //is a mate score ?
                bestScore = negamax<PVNode, limitWay, true , true>(localSS, depth, state, alpha, beta, actDepth);
            else
                bestScore = negamax<PVNode, limitWay, false, true>(localSS, depth, state, alpha, beta, actDepth);
            smp_abort = true;
            for(int i=0; i<nbThreads-1; i++){
                helperThreads[i].wait_thread();
                localSS.nodes += helperThreads[i].local.nodes;
                localSS.nbFirstCutoff += helperThreads[i].local.nbFirstCutoff;
                localSS.nbCutoff += helperThreads[i].local.nbCutoff;
            }
            lastUsedNodes = localSS.nodes-lastUsedNodes;
            bestMove = bestScore != -INF ? localSS.rootBest : finalBestMove;
            string limit;
            if(bestScore <= alpha){
                deltaDown = max(deltaDown*2, lastScore-bestScore+1);
                limit = "upperbound";
            }else if(bestScore >= beta){
                deltaUp = max(deltaUp*2, bestScore-lastScore+1);
                finalBestMove = bestMove;
                limit = "lowerbound";
                ponderMove = nullMove;
            }else{
                finalBestMove = bestMove;
                PV = localSS.PVprint(localSS.PVlines[0]);
                if(localSS.PVlines[0].cmove > 1)ponderMove.moveInfo = localSS.PVlines[0].argMoves[1];
                else ponderMove = nullMove;
                break;
            }
            if(verbose && bestScore != -INF && getElapsedTime() >= chrono::milliseconds{10000}){
                sbig totNodes = localSS.nodes;
                double tcpu = getElapsedTime().count()/1'000'000'000.0;
                printf("info depth %d seldepth %d score %s %s nodes %" PRId64 " nps %d time %d pv %s\n", depth, localSS.seldepth-startRelDepth, scoreToStr(bestScore).c_str(), limit.c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), finalBestMove.to_str().c_str());
                fflush(stdout);
            }
        }while(running);
        bestMove = finalBestMove;
        if(bestScore != -INF)
            lastScore = bestScore;
        double tcpu = getElapsedTime().count()/1'000'000'000.0;
        sbig totNodes = localSS.nodes;
        double speed=0;
        if(tcpu != 0)speed = totNodes/tcpu;
        if(verbose && bestScore != -INF){
            printf("info depth %d seldepth %d score %s nodes %" PRId64 " nps %d time %d pv %s string branching factor %.3f first cutoff %.3f\n", depth, localSS.seldepth-startRelDepth, scoreToStr(bestScore).c_str(), totNodes, (int)(speed), (int)(tcpu*1000), PV.c_str(), pow(totNodes, 1.0/depth), (double)localSS.nbFirstCutoff/localSS.nbCutoff);
            fflush(stdout);
        }
        if(running)
            allInfos.push_back({localSS.nodes, (int)(tcpu*1000), (int)(speed), depth, localSS.seldepth-startRelDepth, bestScore});
        if(abs(bestScore) > MAXIMUM-maxDepth && mateHardBound){
            tm.softBound = hardBound;
            softBoundTime = hardBoundTime;
        }
        softBoundTime = chrono::milliseconds{tm.updateSoft(localSS.bestMoveNodes, lastUsedNodes)};
        this->hardBound = tm.hardBound;
        if(limitWay == 1 && localSS.nodes > tm.softBound)break;
        if(limitWay == 0 && getElapsedTime() > softBoundTime)break;
    }
    return make_tuple(bestMove, ponderMove, lastScore, allInfos);
}

template bestMoveResponse BestMoveFinder::bestMove<0>(GameState&, TM, vector<Move>, bool, bool);
template bestMoveResponse BestMoveFinder::bestMove<1>(GameState&, TM, vector<Move>, bool, bool);
template bestMoveResponse BestMoveFinder::bestMove<2>(GameState&, TM, vector<Move>, bool, bool);
template bestMoveResponse BestMoveFinder::goState<0>(GameState&, TM, bool, bool, int);
template bestMoveResponse BestMoveFinder::goState<1>(GameState&, TM, bool, bool, int);
template bestMoveResponse BestMoveFinder::goState<2>(GameState&, TM, bool, bool, int);
int BestMoveFinder::testQuiescenceSearch(GameState& state){
    localSS.reinit(state);
    clock_t start=clock();
    int score = quiescenceSearch<false, true, false>(localSS, state, -INF, INF, 0);
    clock_t end = clock();
    double tcpu = double(end-start)/CLOCKS_PER_SEC;
    printf("speed: %d; Qnodes:%" PRId64 " score %s\n\n", (int)(localSS.nodes/tcpu), localSS.nodes, scoreToStr(score).c_str());
    return 0;
}

void BestMoveFinder::clear(){
    transposition.clear();
    localSS.history.init();
    localSS.correctionHistory.reset();
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].local.history.init();
        helperThreads[i].local.correctionHistory.reset();
    }
}

void BestMoveFinder::reinit(size_t count){
    transposition.reinit(count);
}

Perft::Perft(){}
big Perft::_perft(GameState& state, ubyte depth){
    visitedNodes++;
    if(depth == 0)return 1;
    //big lastCall=tt.get_eval(state.zobristHash, depth);
    //if(lastCall != MAX_BIG)return lastCall;
    bool inCheck;
    big dangerPositions = 0;
    generator.initDangers(state);
    int nbMoves=generator.generateLegalMoves(state, inCheck, stack[depth], dangerPositions);
    if(depth == 1)return nbMoves;
    big count=0;
    for(int i=0; i<nbMoves; i++){
        state.playMove(stack[depth][i]);
        big nbNodes=_perft(state, depth-1);
        state.undoLastMove();
        count += nbNodes;
    }
    //tt.push({state.zobristHash, count, depth});
    return count;
}
big Perft::perft(GameState& state, ubyte depth, bool verbose){
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
        state.playMove(moves[i]);
        big nbNodes=_perft(state, depth-1);
        state.undoLastMove();
        clock_t end=clock();
        double tcpu = double(end-startMove)/CLOCKS_PER_SEC;
        if(verbose){
            printf("%s: %" PRId64 " (%d/%d %.2fs => %.0f n/s)\n", moves[i].to_str().c_str(), nbNodes, i+1, nbMoves, tcpu, (visitedNodes-startVisitedNodes)/tcpu);
            fflush(stdout);
        }
        count += nbNodes;
    }
    clock_t end=clock();
    double tcpu = double(end-start)/CLOCKS_PER_SEC;
    if(verbose){
        printf("%.3f : %.3f nps %" PRId64 " visited nodes\n", tcpu, visitedNodes/tcpu, visitedNodes);
        fflush(stdout);
    }
    return count;
}
