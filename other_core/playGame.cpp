#include "GameState.hpp"
#include "Move.hpp"
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv){
    string move;
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
    state.print();
    for(Move mv:moves){
        state.playPartialMove(mv);
        state.print();
    }
    printf("%s\n", string(50, '#').c_str());
    for(int i=0; i<moves.size(); i++){
        state.undoLastMove();
        state.print();
    }
}