#include "GameState.hpp"
#include "Evaluator.hpp"
#include "NNUE.hpp"
#include <fstream>
#include <cstdio>
using namespace std;
int main(int argc, char** argv){
    ifstream file(argv[1]);
    IncrementalEvaluator eval;
    if(argc > 2)
        globnnue = NNUE(argv[2]);
    GameState state;
    string fen;
    big sum_eval = 0;
    int count = 0;
    while(getline(file, fen)){
        state.fromFen(fen),
        eval.init(state);
        sum_eval += abs(eval.getRaw(state.friendlyColor()));
        count++;
    }
    printf("%lf\n", ((double)sum_eval)/count);
}