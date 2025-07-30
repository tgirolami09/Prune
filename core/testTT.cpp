#include <iostream>
#include "Move.hpp"
#include "TranspositionTable.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Evaluator.hpp"
const int sizeMB = 64*1000*1000;
LegalMoveGenerator generator;
transpositionTable tt(sizeMB);
GameState state;
Evaluator eval;
TTperft ptt(tt.modulo*sizeof(perftMem)); //assure that thay have the same count
const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
Move test;
big leef = 0;
big recursiveTest(int depth){
    if(depth <= 0){
        return 1;
    }
    int lastEval = ptt.get_eval(state.zobristHash, depth);
    int scorePos = eval.positionEvaluator(state);
    if(lastEval == -1){
        tt.push(state, scorePos, -INF, INF, nullMove, depth);
        assert(tt.get_eval(state, -INF, INF, depth, test) == scorePos);
    }else{
        assert(tt.get_eval(state, -INF, INF, depth, test) == scorePos);
        return lastEval;
    }
    Move moves[maxMoves];
    bool inCheck;
    big count = 0;
    int nbMoves = generator.generateLegalMoves(state, inCheck, moves);
    if(nbMoves == 0)return 0;
    for(int i=0; i<nbMoves; i++){
        state.playMove<false, false>(moves[i]);
        count += recursiveTest(depth-1);
        state.undoLastMove<false>();
    }
    ptt.push({state.zobristHash, count, (ubyte)depth});
    int score = tt.get_eval(state, -INF, INF, depth, test);
    assert(score == INVALID || score == scorePos);
    return count;
}

int main(){
    state.fromFen(startpos);
    int depth;
    cout << ptt.modulo << " " << tt.modulo << '\n';
    cin >> depth;
    cout << depth << '\n';
    leef = recursiveTest(depth);
    printf("%lld\n", leef);
}