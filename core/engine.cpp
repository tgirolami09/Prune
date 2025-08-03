#include <sstream>
#include <string>
#include <strings.h>
// #include "util_magic.cpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "BestMoveFinder.hpp"
#include "TranspositionTable.hpp"

#include <iostream>

// big* table[128]; // [bishop, rook]
// info constants[128];

using namespace std;
//Main class for the game
class Chess{
    public : GameState currentGame;

    //Get it from i/o
    int w_time;
    int b_time;
    int binc=0;
    int winc=0;
};
int moveOverhead = 10;
const int alloted_space=64*1000*1000;
BestMoveFinder bestMoveFinder(alloted_space);
Perft doPerft(alloted_space);
Move getBotMove(GameState gameState, int alloted_time){
    Move moveToPlay = bestMoveFinder.bestMove(gameState,alloted_time);
    return moveToPlay;
}

Move getOpponentMove(){
    //TODO : Some logic to get it from i/o
    return {};
}

int computeAllotedTime(Chess& state){
    int time = state.currentGame.friendlyColor() == WHITE?state.w_time:state.b_time;
    int inc = state.currentGame.friendlyColor() == WHITE?state.winc:state.binc;
    return time/20+inc/2-moveOverhead;
}

void doUCI(string UCI_instruction, Chess& state){
    istringstream stream(UCI_instruction);
    string command;
    stream >> command;
    map<string, int> args;
    string arg;
    int precision;
    if(command == "setoption"){
        string inter;
        stream >> inter; //remove name
        while(stream >> arg){
            stream >> inter; // remove "value"
            if(arg == "Clear" && inter == "Hash"){
                bestMoveFinder.clear();
            }else if(arg == "Move" && inter == "Overhead"){
                stream >> inter >> precision;
                moveOverhead = precision;
            }else{
                stream >> precision;
                if(arg == "Hash"){
                    bestMoveFinder.reinit(precision*1000*1000); // size in MB
                    doPerft.reinit(precision*1000*1000);
                }
            }
        }
        return;
    }else if(command != "position"){
        while(stream >> arg){
            stream >> precision;
            args[arg] = precision;
        }
    }
    if(command == "go"){
        if(args.count("perft")){
            printf("Nodes searched: %lld\n", doPerft.perft(state.currentGame, args["perft"]));
        }else{
            Move move;
            if(args.count("btime") && args.count("wtime")){
                state.b_time = args["btime"];
                state.w_time = args["wtime"];
                state.winc = args["winc"];
                state.binc = args["binc"];
                move=getBotMove(state.currentGame, computeAllotedTime(state));
            }else{
                move = getBotMove(state.currentGame, args["movetime"]);
            }
            printf("bestmove %s\n", move.to_str().c_str());
        }
    }else if(command == "uci"){
        printf("id name pruningBot\nid author tgirolami09 & jbienvenue\n");
        printf("option name Hash type spin default 64 min 1 max 512\n");
        printf("option name Move Overhead type spin default 10 min 0 max 5000\n");
        printf("option name Clear Hash type button\n");
        printf("uciok\n");
    }else if(command == "position"){
        string arg;
        stream >> arg;
        string fen;
        if(arg == "fen"){
            fen = "";
            for(int i=0; i<6; i++){
                string field;
                stream >> field;
                fen += field+" ";
            }
        }else if(arg == "kiwipete"){
            fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
        }else{
            assert(arg == "startpos");
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
        state.currentGame.fromFen(fen);
        string moves;
        while(stream >> moves){
            if(moves == "moves")continue;
            Move move;
            move.from_uci(moves);
            state.currentGame.playPartialMove(move);
        }
    }else if(command == "isready"){
        printf("readyok\n");
    }else if(command == "d"){
        state.currentGame.print();
    }else if(command == "runQ"){
        bestMoveFinder.testQuiescenceSearch(state.currentGame);
    }else if(command == "eval"){
        printf("static evaluation: %d cp\n", bestMoveFinder.eval.positionEvaluator(state.currentGame));
    }else if(command == "stop"){
        bestMoveFinder.stop();
    }
    //Implement actual logic for UCI management
}

int main(int argc, char** argv){
    string UCI_instruction = "programStart";
    Chess state;
    state.currentGame.fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if(argc > 1){
        for(int i=1; i<argc; i++)
            doUCI(argv[i], state);
        return 0;
    }
    while (UCI_instruction != "quit"){
        doUCI(UCI_instruction, state);
        fflush(stdout);
        if(!getline(cin,UCI_instruction)){
            printf("quit\n");
            break;
        }
    }
}