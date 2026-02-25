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

BestMoveFinder::usefull::usefull(const GameState& state, tunables& parameters):nodes(0), bestMoveNodes(0), seldepth(0), nbCutoff(0), nbFirstCutoff(0),
tbHits(0),
rootBest(nullMove), mainThread(true){
    eval.init(state);
    generator.initDangers(state);
    history.init(parameters);
    correctionHistory.reset();
}
BestMoveFinder::usefull::usefull():nodes(0), bestMoveNodes(0), seldepth(0), nbCutoff(0), nbFirstCutoff(0),
tbHits(0),
rootBest(nullMove), mainThread(true){}
void BestMoveFinder::usefull::reinit(const GameState& state){
    nodes = 0;
    bestMoveNodes = 0;
    seldepth = 0;
    nbCutoff = 0;
    nbFirstCutoff = 0;
    tbHits = 0;
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

BestMoveFinder::BestMoveFinder(int memory, bool mute):transposition(memory), helperThreads(0){
    book = load_book("book.bin", mute);
}
BestMoveFinder::BestMoveFinder():transposition(hashMul), helperThreads(0){
}

void BestMoveFinder::clear_helpers(){
    if(helperThreads.size() == 0)return;
    smp_end = true;
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].launch(-1, -1);
        helperThreads[i].t.join();
    }
}

BestMoveFinder::~BestMoveFinder(){
    clear_helpers();
}

void BestMoveFinder::setThreads(int nT){
    if(nT == nbThreads)return;
    if(nbThreads > 1){
        clear_helpers();
    }
    if(nT == 1){
        helperThreads.clear();
    }else{
        smp_end = false;
        helperThreads = vector<HelperThread>(nT-1);
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

void BestMoveFinder::HelperThread::launch(int _relDepth, int _limitWay){
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
    if(ss.eval.isInsufficientMaterial() || state.rule50_count() > 100)return 0;
    ss.nodes++;
    if(isPV && relDepth > ss.seldepth)ss.seldepth = relDepth;
    //dbyte hint;
    const int rootDist = relDepth-startRelDepth;
    if(rootDist >= maxDepth)return ss.eval.correctEval(ss.eval.getRaw(state.friendlyColor()), ss.correctionHistory, state);
    bool ttHit=false;
    infoScore& ttEntry = transposition.getEntry(state, ttHit);
    if(ttHit){
        if(!isPV){
            int lastEval=transposition.get_eval(ttEntry, alpha, beta, 0);
            if(lastEval != INVALID)
                return fromTT(lastEval, rootDist);
        }
        //hint = transposition.getMove(ttEntry);
    }
    // Tablebase probe in quiescence
    if (tbProbe.canProbe(state, ss.eval.getNbMan())) {
        int wdl = tbProbe.probeWDL(state);
        if (wdl != TB_RESULT_INVALID) {
            ss.tbHits++;
            return TablebaseProbe::wdlToScore(wdl, rootDist);
        }
    }
    int& staticEval = ss.stack[rootDist].static_score;
    int& raw_eval = ss.stack[rootDist].raw_eval;
    if(!isCalc){
        if(ttHit)
            raw_eval = ttEntry.raw_eval;
        else
            raw_eval = ss.eval.getRaw(state.friendlyColor());
        staticEval = ss.eval.correctEval(raw_eval, ss.correctionHistory, state);
    }
    int typeNode = UPPERBOUND;
    bool testCheck = ss.generator.initDangers(state);
    int bestEval = MINIMUM;
    if(!testCheck){
        if(staticEval >= beta){
            transposition.push(state, staticEval, LOWERBOUND, nullMove, 0, raw_eval, isPV);
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
        ss.eval.playMove(capture, !state.friendlyColor(), &state);
        int score = -quiescenceSearch<limitWay, isPV, false>(ss, state, -beta, -alpha, relDepth+1);
        ss.eval.undoMove(capture, !state.friendlyColor());
        state.undoLastMove();
        if(!running || smp_abort)return 0;
        if(score >= beta){
            ss.nbCutoff++;
            if(i == 0)ss.nbFirstCutoff++;
            transposition.push(state, absoluteScore(score, rootDist), LOWERBOUND, capture, 0, raw_eval, isPV);
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
    transposition.push(state, absoluteScore(bestEval, rootDist), typeNode, bestCapture, 0, raw_eval, isPV);
    return bestEval;
}

template<bool isPV, int limitWay>
inline int BestMoveFinder::Evaluate(usefull& ss, GameState& state, int alpha, int beta, int relDepth){
    int score = quiescenceSearch<limitWay, isPV, true>(ss, state, alpha, beta, relDepth);
    if constexpr(limitWay == 1)if(ss.nodes > hardBound)running=false;
    return score;
}

template <bool isPV, int limitWay, bool isRoot>
int BestMoveFinder::negamax(usefull& ss, int depth, GameState& state, int alpha, const int beta, const int relDepth, bool cutnode, const int16_t excludedMove){
    if(isPV)
        cutnode = false;
    bool allnode = !cutnode && !isPV;
    const int rootDist = relDepth-startRelDepth;
    if(rootDist >= maxDepth)return ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    if(isPV)ss.seldepth = max(ss.seldepth, relDepth);
    transposition.prefetch(state);
    if(MAXIMUM-rootDist <= alpha)return MAXIMUM-rootDist;
    if(MINIMUM+rootDist >= beta)return MINIMUM+rootDist;
    if constexpr(limitWay == 0)if((ss.nodes & 1023) == 0 && getElapsedTime() >= hardBoundTime)running=false;
    if(!running || smp_abort)return 0;
    if(state.rule50_count() >= 100 || ss.eval.isInsufficientMaterial()){
        if constexpr (isPV)ss.beginLine(rootDist);
        if(state.rule50_count() == 100){
            if(ss.generator.isCheck()){
                bool inCheck;
                Order& order = ss.stack[rootDist].order;
                order.nbMoves = ss.generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
                if(order.nbMoves == 0){
                    return MINIMUM+rootDist;
                }

            }
        }
        return 0;
    }
    int& static_eval = ss.stack[rootDist].static_score;
    int& raw_eval = ss.stack[rootDist].raw_eval;
    bool ttHit=false;
    infoScore& ttEntry = transposition.getEntry(state, ttHit);
    if(ttHit)
        raw_eval = ttEntry.raw_eval;
    else
        raw_eval = ss.eval.getRaw(state.friendlyColor());
    static_eval = ss.eval.correctEval(raw_eval, ss.correctionHistory, state);
    // Tablebase probe in search
    if (!isRoot && tbProbe.canProbe(state, ss.eval.getNbMan(), depth)) {
        int wdl = tbProbe.probeWDL(state);
        if (wdl != TB_RESULT_INVALID) {
            ss.tbHits++;
            int tbScore = TablebaseProbe::wdlToScore(wdl, rootDist);
            // For wins/losses, cut off immediately
            if (wdl == TB_RESULT_WIN || wdl == TB_RESULT_LOSS) {
                if constexpr(isPV) ss.beginLine(rootDist);
                return tbScore;
            }
            // Here we can only have DRAW, BLESSED_LOSS and CURSED_WIN. All are treated as a draw.
            // For draws at non-PV nodes, return immediately. At PV nodes, only use beta cutoff.
            else {
                if constexpr(!isPV) {
                    return tbScore;  // No PV concern at non-PV nodes
                }
                if (tbScore >= beta) {
                    ss.beginLine(rootDist);
                    return tbScore;
                }
            }
        }
    }
    if(depth == 0 || (!isRoot && depth == 1 && (static_eval+100 < alpha || static_eval > beta+100))){
        if constexpr(isPV)ss.beginLine(rootDist);
        return Evaluate<isPV, limitWay>(ss, state, alpha, beta, relDepth);
    }
    ss.nodes++;
    int16_t lastBest = nullMove.moveInfo;
    if(excludedMove == nullMove.moveInfo && ttHit){
        if constexpr(!isPV){
            int lastEval = transposition.get_eval(ttEntry, alpha, beta, depth);
            if(lastEval != INVALID)
                return fromTT(lastEval, rootDist);
        }
        lastBest = transposition.getMove(ttEntry);
    }
    ubyte typeNode = UPPERBOUND;
    Order& order = ss.stack[rootDist].order;
    bool inCheck=ss.generator.isCheck();
    bool improving = false;
    if((!ttHit || ttEntry.depth+parameters.iir_validity_depth < depth) && depth >= parameters.iir_min_depth && !allnode && excludedMove == nullMove.moveInfo)depth--;
    if(rootDist > 2)
        improving = ss.stack[rootDist-2].static_score < static_eval && excludedMove == nullMove.moveInfo;
    if constexpr(!isPV){
        if(!inCheck && excludedMove == nullMove.moveInfo && beta > MINIMUM+maxDepth){
            int margin;
            if(improving)
                margin = parameters.rfp_improving*depth;
            else
                margin = parameters.rfp_nimproving*depth;
            if(static_eval >= beta+margin)
                return static_eval;
            int r = (depth*parameters.nmp_red_depth_div+parameters.nmp_red_base)/1024;
            if(rootDist >= ss.min_nmp_ply && depth >= r && !ss.eval.isOnlyPawns() && static_eval >= beta){
                state.playNullMove();
                ss.generator.initDangers(state);
                int v = -negamax<false, limitWay>(ss, depth-r, state, -beta, -beta+1, relDepth+1, !cutnode);
                state.undoNullMove();
                if(v >= beta){
                    if(depth <= 10 || ss.min_nmp_ply != 0){
                        if(abs(v) > MAXIMUM-maxDepth)return beta;
                        return v;
                    }
#ifdef DEBUG_MACRO
                    if(cutnode)
                        nmpVerifCutNode++;
                    else
                        nmpVerifAllNode++;
#endif
                    ss.min_nmp_ply = rootDist+r;
                    ss.generator.initDangers(state);
                    v = negamax<false, limitWay>(ss, depth-r, state, beta-1, beta, relDepth, cutnode);
                    ss.min_nmp_ply = 0;
                    if(v >= beta){
#ifdef DEBUG_MACRO
                    if(cutnode)
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
    if(!isRoot && ttHit && ttEntry.depth + parameters.se_validity_depth >= depth && ttEntry.typeNode() != UPPERBOUND && depth >= parameters.se_min_depth && excludedMove == nullMove.moveInfo && abs(ttEntry.score) < MAXIMUM-maxDepth){
        int goal = ttEntry.score - depth;
        int score = negamax<false, limitWay>(ss, (depth-1)/2, state, goal-1, goal, relDepth, cutnode, ttEntry.bestMoveInfo);
        if(score < goal){
            firstMoveExtension++;
            if(!isPV && score <= goal-parameters.se_dext_margin)
                firstMoveExtension++;
        }else if(cutnode){
            firstMoveExtension--;
        }else if(ttEntry.score >= beta){
            firstMoveExtension--;
        }
        ss.generator.initDangers(state);
    }
    order.nbMoves = ss.generator.generateLegalMoves(state, inCheck, order.moves, order.dangerPositions);
    if constexpr(isRoot) {
        if (wdlFilterNb > 0) {
            int newNb = 0;
            for (int i = 0; i < order.nbMoves; i++) {
                for (int j = 0; j < wdlFilterNb; j++) {
                    if (order.moves[i].moveInfo == wdlFilterMoveInfos[j]) {
                        order.moves[newNb++] = order.moves[i];
                        break;
                    }
                }
            }
            if (newNb > 0)
                order.nbMoves = newNb;
        }
    }
    if(order.nbMoves == 0){
        int score;
        if(inCheck)
            score = MINIMUM+rootDist;
        else score = MIDDLE;
        if constexpr(isPV)ss.beginLine(rootDist);
        return score;
    }
    if(order.nbMoves == 1){
        if(isRoot)
            ss.rootBest = order.moves[0];
        state.playMove(order.moves[0]);
        if(state.twofold()){
            state.undoLastMove();
            if constexpr(isPV)ss.beginLineMove(rootDist, order.moves[0]);
            return MIDDLE;
        }
        ss.eval.playMove(order.moves[0], !state.friendlyColor(), &state);
        ss.generator.initDangers(state);
        int sc = -negamax<isPV, limitWay>(ss, depth, state, -beta, -alpha, relDepth+1, !cutnode);
        ss.eval.undoMove(order.moves[0], !state.friendlyColor());
        state.undoLastMove();
        if (sc > alpha && sc < beta && isPV)ss.transfer(rootDist, order.moves[0]);
        return sc;
    }
    order.init(state.friendlyColor(), lastBest, ss.history, rootDist, state);
    Move bestMove = nullMove;
    int bestScore = -INF;
    int triedMove = 0;
    for(int rankMove=0; rankMove<order.nbMoves; rankMove++){
        int flag;
        Move curMove = order.pop_max(flag);
        if(excludedMove == curMove.moveInfo)continue;
        sbig startNodes = ss.nodes;
        if(isRoot && verbose && ss.mainThread && DEBUG){
            printf("info depth %d currmove %s currmovenumber %d nodes %" PRId64 " string flag %d\n", depth, curMove.to_str().c_str(), rankMove+1, ss.nodes, flag);
            fflush(stdout);
        }
        int moveHistory;
        if(ss.history.isKiller(curMove, rootDist))
            moveHistory = maxHistory;
        else
            moveHistory = ss.history.getHistoryScore(curMove, state.friendlyColor());
        if(bestScore >= MINIMUM+maxDepth){
            if(!curMove.isTactical()){
                if(triedMove > depth*depth*parameters.lmp_mul+parameters.lmp_base)continue;
                if(moveHistory < -parameters.mhp_mul*depth && triedMove >= 1)
                    continue;
                int futilityValue = static_eval+parameters.fp_base+parameters.fp_mul*depth;
                if(!isPV && triedMove >= 1 && depth <= parameters.fp_max_depth && !inCheck && futilityValue <= alpha){
                    continue;
                }
            }else{
                if(!isPV && moveHistory < -parameters.mchp_mul*depth*depth && depth <= 4)
                    continue;
            }
        }
#ifdef DEBUG_MACRO
        if(curMove.isTactical()){
            capthistSum += moveHistory;
            capthistSquare += moveHistory*moveHistory;
            nbCaptHist++;
        }else{
            quiethistSum += moveHistory;
            quiethistSquare += moveHistory*moveHistory;
            nbquietHist++;
        }
#endif
        int score;
        state.playMove(curMove);
        bool isDraw = false;
        triedMove++;
        if(state.twofold()){
            score = MIDDLE;
            isDraw = true;
        }else{
            ss.eval.playMove(curMove, !state.friendlyColor(), &state);
            bool inCheckPos = ss.generator.initDangers(state);
            int reductionDepth = 1;
            if(inCheckPos && firstMoveExtension == 0){
                reductionDepth--;
            }
            if(rankMove > 0){
                int addRedDepth = 0;
                if(rankMove > 3 && depth > 3){
                    addRedDepth = static_cast<int>(parameters.lmr_base + log(depth) * log(rankMove) * parameters.lmr_div);
                    addRedDepth -= (moveHistory)*parameters.lmr_history/maxHistory;
                    addRedDepth /= 1024;
                    addRedDepth = max(addRedDepth, 0);
                }
                score = -negamax<false, limitWay>(ss, depth-reductionDepth-addRedDepth, state, -alpha-1, -alpha, relDepth+1, true);
                bool fullSearch = false;
                if((score > alpha && score < beta) || (isPV && score == beta && beta == alpha+1)){
                    fullSearch = true;
                }
                if(addRedDepth && score >= beta)
                    fullSearch = true;
                if(fullSearch){
                    ss.generator.initDangers(state);
                    score = -negamax<isPV, limitWay>(ss, depth-reductionDepth, state, -beta, -alpha, relDepth+1, !cutnode);
                }
            }else
                score = -negamax<isPV, limitWay>(ss, depth-reductionDepth+firstMoveExtension, state, -beta, -alpha, relDepth+1, !cutnode);
            ss.eval.undoMove(curMove, !state.friendlyColor());
        }
        state.undoLastMove();
        if(!running || smp_abort)return bestScore;
        if(score >= beta){ //no need to copy the pv, because it will fail low on the parent
            transposition.push(state, absoluteScore(score, rootDist), LOWERBOUND, curMove, depth, raw_eval, isPV);
            ss.nbCutoff++;
            if(isRoot)ss.rootBest=curMove;
            if(rankMove == 0)ss.nbFirstCutoff++;
            ss.history.addKiller(curMove, depth, rootDist, state.friendlyColor());
            ss.history.negUpdate(order.moves, rankMove, state.friendlyColor(), depth);
            if(!curMove.isTactical()){
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
            if constexpr(isPV){
                if(isDraw)ss.beginLineMove(rootDist, curMove);
                else ss.transfer(rootDist, curMove);
            }
        }
        if(score > bestScore)bestScore = score;
    }
    if(cutnode && bestScore == alpha)
        return bestScore;
    if((!isRoot || typeNode != UPPERBOUND) && excludedMove == nullMove.moveInfo){
        transposition.push(state, absoluteScore(bestScore, rootDist), typeNode, bestMove, depth, raw_eval, isPV);
    }
    if(!inCheck && (!bestMove.isTactical()) && abs(bestScore) < MAXIMUM-maxDepth &&
        (typeNode != UPPERBOUND || bestScore < static_eval)){
        ss.correctionHistory.update(state, bestScore-static_eval, depth);
    }
    return bestScore;
}

void BestMoveFinder::launchSMP(int idThread){
    /*HelperThread& ss = helperThreads[idThread-1];
    ss.local.reinit(ss.localState);
    ss.local.mainThread = false;
    negamax<PVNode, limitWay, int, true>(ss.local, depth, ss.localState, alpha, beta, relDepth);*/
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
        if(ss.limitWay == 0)iterativeDeepening<0>(ss.local, ss.localState, TM(0, 0), ss.relDepth);
        if(ss.limitWay == 1)iterativeDeepening<1>(ss.local, ss.localState, TM(0, 0), ss.relDepth);
        if(ss.limitWay == 2)iterativeDeepening<2>(ss.local, ss.localState, TM(0, 0), ss.relDepth);
        {
            lock_guard<mutex> lock(ss.mtx);
            ss.running = false;
            ss.cv.notify_one();
        }
    }
}

void BestMoveFinder::updatemainSS(usefull& ss, Record& oldss){
    ss.nodes -= oldss.nodes;
    ss.nbFirstCutoff -= oldss.nbFirstCutoff;
    ss.nbCutoff -= oldss.nbCutoff;
    ss.tbHits -= oldss.tbHits;
    oldss.nbFirstCutoff = oldss.nbCutoff = oldss.nodes = oldss.tbHits = 0;
    for(int i=0; i<nbThreads-1; i++){
        oldss.nodes += helperThreads[i].local.nodes;
        oldss.nbFirstCutoff += helperThreads[i].local.nbFirstCutoff;
        oldss.nbCutoff += helperThreads[i].local.nbCutoff;
        ss.seldepth = max(ss.seldepth, helperThreads[i].local.seldepth);
        oldss.tbHits += helperThreads[i].local.tbHits;
    }
    ss.nodes += oldss.nodes;
    ss.nbFirstCutoff += oldss.nbFirstCutoff;
    ss.nbCutoff += oldss.nbCutoff;
    ss.tbHits += oldss.tbHits;
}

template <int limitWay>
bestMoveResponse BestMoveFinder::bestMove(GameState& state, TM tm, vector<Move> movesFromRoot, bool _verbose){
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
    bestMoveResponse res=goState<limitWay>(state, tm, verbose, actDepth);
    for(unsigned long i=0; i<movesFromRoot.size(); i++)
        state.undoLastMove();
    return res;
}

template<int limitWay>
bestMoveResponse BestMoveFinder::iterativeDeepening(usefull& ss, GameState& state, TM tm, int actDepth){
    vector<depthInfo> allInfos;
    chrono::milliseconds softBoundTime{tm.softBound};
    Move bestMove=nullMove;
    int depthMax = maxDepth;
    if(ss.mainThread && limitWay == 2){
        depthMax = tm.hardBound;
    }
    Record rec{};
    int lastScore = ss.eval.getScore(state.friendlyColor(), ss.correctionHistory, state);
    Move ponderMove=nullMove;
    startRelDepth = actDepth-1;
    for(int depth=1; depth<=depthMax && running && !smp_abort; depth++){
        int deltaUp = parameters.aw_base;
        int deltaDown = parameters.aw_base;
        ss.seldepth = 0;
        if(abs(lastScore) > MAXIMUM-maxDepth)
            deltaDown = 1;
        int bestScore;
        Move finalBestMove=bestMove;
        sbig lastUsedNodes = 0;
        string PV;
        do{
            int alpha = lastScore-deltaDown;
            int beta = lastScore+deltaUp;
            ss.generator.initDangers(state);
            lastUsedNodes = ss.nodes;
            smp_abort = false;
            bestScore = negamax<true, limitWay, true>(ss, depth, state, alpha, beta, actDepth, false);
            lastUsedNodes = ss.nodes-lastUsedNodes;
            bestMove = bestScore != -INF ? ss.rootBest : finalBestMove;
            string limit;
            if(bestScore <= alpha){
                deltaDown = max<int>(deltaDown*parameters.aw_mul, lastScore-bestScore+1);
                limit = "upperbound";
            }else if(bestScore >= beta){
                deltaUp = max<int>(deltaUp*parameters.aw_mul, bestScore-lastScore+1);
                finalBestMove = bestMove;
                limit = "lowerbound";
                ponderMove = nullMove;
            }else{
                finalBestMove = bestMove;
                PV = ss.PVprint(ss.PVlines[0]);
                if(ss.PVlines[0].cmove > 1)ponderMove.moveInfo = ss.PVlines[0].argMoves[1];
                else ponderMove = nullMove;
                break;
            }
            if(ss.mainThread && verbose && bestScore != -INF && getElapsedTime() >= chrono::milliseconds{10000}){
                sbig totNodes = ss.nodes;
                double tcpu = getElapsedTime().count()/1'000'000'000.0;
                updatemainSS(ss, rec);
                printf("info depth %d seldepth %d score %s %s nodes %" PRId64 " nps %d time %d tbhits %" PRId64 " pv %s\n", depth, ss.seldepth-startRelDepth, scoreToStr(bestScore).c_str(), limit.c_str(), totNodes, (int)(totNodes/tcpu), (int)(tcpu*1000), ss.tbHits, finalBestMove.to_str().c_str());
                fflush(stdout);
            }
        }while(running && !smp_abort);
        bestMove = finalBestMove;
        if(bestScore != -INF)
            lastScore = bestScore;
        if(ss.mainThread){
            double tcpu = getElapsedTime().count()/1'000'000'000.0;
            sbig totNodes = ss.nodes;
            double speed=0;
            if(tcpu != 0)speed = totNodes/tcpu;
            if(verbose && bestScore != -INF){
                updatemainSS(ss, rec);
                printf("info depth %d seldepth %d score %s nodes %" PRId64 " nps %d time %d tbhits %" PRId64 " pv %s string branching factor %.3f first cutoff %.3f\n", depth, ss.seldepth-startRelDepth, scoreToStr(bestScore).c_str(), totNodes, (int)(speed), (int)(tcpu*1000), ss.tbHits, PV.c_str(), pow(totNodes, 1.0/depth), (double)ss.nbFirstCutoff/ss.nbCutoff);
                fflush(stdout);
            }
            if(running)
                allInfos.push_back({ss.nodes, (int)(tcpu*1000), (int)(speed), depth, ss.seldepth-startRelDepth, bestScore});
            softBoundTime = chrono::milliseconds{tm.updateSoft(ss.bestMoveNodes, lastUsedNodes, bestMove.moveInfo, parameters, verbose)};
            this->hardBound = tm.hardBound;
            hardBoundTime = chrono::milliseconds{tm.hardBound};
            if(limitWay == 1 && ss.nodes > tm.softBound)break;
            if(limitWay == 0 && getElapsedTime() > softBoundTime)break;
        }
    }
    return make_tuple(bestMove, ponderMove, lastScore, allInfos);
}

template<int limitWay>
bestMoveResponse BestMoveFinder::goState(GameState& state, TM tm, bool _verbose, int actDepth){
    verbose = _verbose;
    wdlFilterNb = 0;
    hardBoundTime = chrono::milliseconds{tm.hardBound*1000};
    startSearch = timeMesure::now();
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
                printf("info string Found book move for fen : %s\n",state.toFen().c_str());
            return make_tuple(bookMove, nullMove, INF, vector<depthInfo>());
        }else if(verbose){
            printf("info string bad move find in table %s (in %s)\n", bookMove.to_str().c_str(), state.toFen().c_str());
        }
    }
    if(order.nbMoves == 0){
        int score;
        if(inCheck)score = MINIMUM;
        else score = 0;
        if(verbose)
            printf("info depth 1 seldepth 0 score %s nodes 0\n", scoreToStr(score).c_str());
        return make_tuple(nullMove, nullMove, score, vector<depthInfo>());
    }
    running = true;
    this->hardBound = INT64_MAX;
    if(order.nbMoves == 1 && limitWay == 0){
        running = false;
        if(verbose)
            printf("info depth 1 seldepth 0 score %s nodes 0 nps 0 time 0\n", scoreToStr(localSS.eval.getRaw(state.friendlyColor())).c_str());
        return make_tuple(order.moves[0], nullMove, INF, vector<depthInfo>(0));
    }
    // Tablebase probe at root (always do this)
    Move tbMove = nullMove;
    int tbWdl = TB_RESULT_INVALID;
    if (tbProbe.canProbe(state, localSS.eval.getNbMan())) {
        tbWdl = tbProbe.probeRoot(state, tbMove);
        if (tbWdl != TB_RESULT_INVALID) {
            if (verbose) {
                printf("info string Tablebase hit: ");
                switch (tbWdl) {
                    case TB_RESULT_WIN: printf("Win"); break;
                    case TB_RESULT_CURSED_WIN: printf("Cursed Win"); break;
                    case TB_RESULT_DRAW: printf("Draw"); break;
                    case TB_RESULT_BLESSED_LOSS: printf("Blessed Loss"); break;
                    case TB_RESULT_LOSS: printf("Loss"); break;
                }
                printf("\n");
                fflush(stdout);
            }
            // In all positions return  perfect move from egtb
            if (tbWdl == TB_RESULT_WIN){
                running = false;
                return make_tuple(tbMove, nullMove, MAXIMUM, vector<depthInfo>());
            }
            else if (tbWdl == TB_RESULT_LOSS){
                running = false;
                return make_tuple(tbMove, nullMove, MINIMUM, vector<depthInfo>());
            }
            // Only possibilities are : TB_RESULT_DRAW, TB_RESULT_CURSED_WIN and TB_RESULT_BLESSED_LOSS. All are treated as draw
            else{
                running = false;
                return make_tuple(tbMove, nullMove, 0, vector<depthInfo>());
            }
        }
        // DTZ probe failed (no DTZ files) - try WDL-only fallback to filter root moves
        int wdlFallback = tbProbe.probeRootWDLFallback(state, order.moves, order.nbMoves);
        if (wdlFallback != TB_RESULT_INVALID) {
            wdlFilterNb = order.nbMoves;
            for (int i = 0; i < order.nbMoves; i++)
                wdlFilterMoveInfos[i] = order.moves[i].moveInfo;
            if (verbose) {
                printf("info string Tablebase WDL fallback: ");
                switch (wdlFallback) {
                    case TB_RESULT_WIN:          printf("Win"); break;
                    case TB_RESULT_CURSED_WIN:   printf("Cursed Win"); break;
                    case TB_RESULT_DRAW:         printf("Draw"); break;
                    case TB_RESULT_BLESSED_LOSS: printf("Blessed Loss"); break;
                    case TB_RESULT_LOSS:         printf("Loss"); break;
                }
                printf(" (%d moves kept)\n", order.nbMoves);
                fflush(stdout);
            }
        }
    }
    if(verbose){
        printf("info string use a tt of %" PRId64 " entries (%" PRId64 " MB) (%" PRId64 "B by entry)\n", transposition.modulo, (big)transposition.modulo*sizeof(infoScore)/hashMul, (big)sizeof(infoScore));
    }
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].launch(actDepth, limitWay);
    }
    auto res=iterativeDeepening<limitWay>(localSS, state, tm, actDepth);
    smp_abort = true;
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].wait_thread();
    }
    smp_abort = false;
    return res;
}

void BestMoveFinder::aging(){
    transposition.aging();
}

template bestMoveResponse BestMoveFinder::bestMove<0>(GameState&, TM, vector<Move>, bool);
template bestMoveResponse BestMoveFinder::bestMove<1>(GameState&, TM, vector<Move>, bool);
template bestMoveResponse BestMoveFinder::bestMove<2>(GameState&, TM, vector<Move>, bool);
template bestMoveResponse BestMoveFinder::goState<0>(GameState&, TM, bool, int);
template bestMoveResponse BestMoveFinder::goState<1>(GameState&, TM, bool, int);
template bestMoveResponse BestMoveFinder::goState<2>(GameState&, TM, bool, int);
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
    localSS.history.init(parameters);
    localSS.correctionHistory.reset();
    for(int i=0; i<nbThreads-1; i++){
        helperThreads[i].local.history.init(parameters);
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
    //if(depth == 1)return nbMoves;
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
        big startVisitedNodes = visitedNodes;
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
