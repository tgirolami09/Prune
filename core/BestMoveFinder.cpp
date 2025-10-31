#include "BestMoveFinder.hpp"

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
    history.init();
}
void BestMoveFinder::stop(){
    running = false;
}
chrono::nanoseconds BestMoveFinder::getElapsedTime(){
    return timeMesure::now()-startSearch;
}

string BestMoveFinder::PVprint(LINE pvLine){
    string resLine = "";
    for(int i=0; i<pvLine.cmove; i++){
        Move mv;
        mv.moveInfo = pvLine.argMoves[i];
        if(i != 0)resLine += " ";
        resLine += mv.to_str();
    }
    return resLine;
}
void BestMoveFinder::transferLastPV(){
    lastPV.cmove = PVlines[0].cmove;
    for(int i=0; i<PVlines[0].cmove; i++)
        lastPV.argMoves[i] = PVlines[0].argMoves[i];
}


void BestMoveFinder::transfer(int relDepth, Move move){
    PVlines[relDepth-1].argMoves[0] = move.moveInfo;
    memcpy(&PVlines[relDepth-1].argMoves[1], PVlines[relDepth].argMoves, PVlines[relDepth].cmove * sizeof(int16_t));
    PVlines[relDepth-1].cmove = PVlines[relDepth].cmove+1;
}
void BestMoveFinder::beginLine(int relDepth){
    PVlines[relDepth-1].cmove = 0;
}

void BestMoveFinder::beginLineMove(int relDepth, Move move){
    PVlines[relDepth-1].argMoves[0] = move.moveInfo;
    PVlines[relDepth-1].cmove = 1;
}

void BestMoveFinder::resetLines(){
    for(int i=0; i<maxDepth; i++){
        PVlines[i].cmove = 0;
    }
}

int16_t BestMoveFinder::getPVMove(int relDepth){
    if(lastPV.cmove < relDepth)
        return lastPV.argMoves[relDepth];
    return nullMove.moveInfo;
}

int BestMoveFinder::testQuiescenceSearch(GameState& state){
    nodes = 0;
    clock_t start=clock();
    int score = quiescenceSearch<false, true>(state, -INF, INF, 0);
    clock_t end = clock();
    double tcpu = double(end-start)/CLOCKS_PER_SEC;
    printf("speed: %d; Qnodes:%d score %s\n\n", (int)(nodes/tcpu), nodes, scoreToStr(score).c_str());
    return 0;
}

void BestMoveFinder::clear(){
    transposition.clear();
    history.init();
}

void BestMoveFinder::reinit(size_t count){
    transposition.reinit(count);
}

Perft::Perft(size_t _space):tt(0), space(_space){}
big Perft::_perft(GameState& state, ubyte depth){
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
        state.playMove(moves[i]);
        big nbNodes=_perft(state, depth-1);
        state.undoLastMove();
        count += nbNodes;
    }
    tt.push({state.zobristHash, count, depth});
    return count;
}
big Perft::perft(GameState& state, ubyte depth){
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
        state.playMove(moves[i]);
        big nbNodes=_perft(state, depth-1);
        state.undoLastMove();
        clock_t end=clock();
        double tcpu = double(end-startMove)/CLOCKS_PER_SEC;
        printf("%s: %" PRId64 " (%d/%d %.2fs => %.0f n/s)\n", moves[i].to_str().c_str(), nbNodes, i+1, nbMoves, tcpu, (visitedNodes-startVisitedNodes)/tcpu);
        fflush(stdout);
        count += nbNodes;
    }
    tt.push({state.zobristHash, count, depth});
    clock_t end=clock();
    double tcpu = double(end-start)/CLOCKS_PER_SEC;
    printf("%.3f : %.3f nps %" PRId64 " visited nodes\n", tcpu, visitedNodes/tcpu, visitedNodes);
    fflush(stdout);
    tt.clearMem();
    return count;
}
void Perft::reinit(size_t count){
    space = count;
}