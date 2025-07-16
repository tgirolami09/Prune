#include "GameState.hpp"
#include "Move.hpp"
#include <fstream>
#include <vector>

int main(int argc, char** argv){
    string move;
    vector<Move> moves;
    ifstream file(argv[1]);
    while(file >> move){
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