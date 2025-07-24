#include "GameState.hpp"
#include "Evaluator.hpp"
#include "Move.hpp"
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv){
    string move;
    Evaluator eval;
    vector<Move> moves;
    ifstream file(argv[1]);
    int indice = argc > 2?atoi(argv[2]):1;
    string game;
    for(int i=0; i<indice; i++)getline(file, game);
    istringstream movesStr(game);
    while(movesStr >> move){
        Move mv;
        mv.from_uci(move);
        moves.push_back(mv);
    }
    GameState state;
    state.fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if(argc <= 3)state.print();
    for(Move mv:moves){
        state.playPartialMove(mv);
        if(argc <= 3){
            state.print();
            printf("cp: %d\n", eval.positionEvaluator(state));
        }
    }
    if(argc <= 3)printf("%s\n", string(50, '#').c_str());
    for(int i=0; i<moves.size(); i++){
        state.undoLastMove();
        if(argc <= 3)state.print();
    }
    if(argc > 3){
        int nbTour=atoi(argv[3]);
        clock_t start=clock();
        int sumScore=0;
        for(int i=0; i<nbTour; i++){
            for(Move mv:moves){
                state.playPartialMove(mv);
                sumScore += eval.positionEvaluator(state);
            }
            for(int i=0; i<moves.size(); i++){
                state.undoLastMove();
                sumScore += eval.positionEvaluator(state);
            }
        }
        clock_t end=clock();
        double tcpu=((double)end-start)/CLOCKS_PER_SEC;
        printf("%.3f\n", nbTour*moves.size()/tcpu);
        printf("%.3f\n", ((double)sumScore)/(nbTour*moves.size()));
    }
}