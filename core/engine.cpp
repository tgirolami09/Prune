#include <sstream>
#include <string>
#include <strings.h>
// #include "util_magic.cpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "BestMoveFinder.hpp"

#include <iostream>

// big* table[128]; // [bishop, rook]
// info constants[128];

using namespace std;
//Main class for the game
class Chess{
public :
    GameState root;
    vector<Move> movesFromRoot;

    //Get it from i/o
    int w_time;
    int b_time;
    int binc=0;
    int winc=0;
};
int moveOverhead = 100;
const int alloted_space=64*1000*1000;
BestMoveFinder bestMoveFinder(alloted_space);
Perft doPerft(alloted_space);
template<bool isTimeLimit=true>
Move getBotMove(Chess state, int alloted_time){
    Move moveToPlay = bestMoveFinder.bestMove<isTimeLimit>(state.root, alloted_time, state.movesFromRoot);
    return moveToPlay;
}

Move getOpponentMove(){
    //TODO : Some logic to get it from i/o
    return {};
}

int computeAllotedTime(Chess& state){
    bool color = state.root.friendlyColor()^(state.movesFromRoot.size()&1);
    int time = color == WHITE?state.w_time:state.b_time;
    int inc = color == WHITE?state.winc:state.binc;
    int maxTime = time/20+inc/2;
    //maxTime = min(maxTime, time/10);
    return maxTime-moveOverhead;
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
            printf("Nodes searched: %ld\n", doPerft.perft(state.root, args["perft"]));
        }else{
            Move move;
            if(args.count("btime") && args.count("wtime")){
                state.b_time = args["btime"];
                state.w_time = args["wtime"];
                state.winc = args["winc"];
                state.binc = args["binc"];
                move=getBotMove(state, computeAllotedTime(state));
            }else if(args.count("movetime")){
                move = getBotMove(state, args["movetime"]);
            }else{
                move = getBotMove<false>(state, args["depth"]);
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
        state.root.fromFen(fen);
        string moves;
        state.movesFromRoot.clear();
        while(stream >> moves){
            if(moves == "moves")continue;
            Move move;
            move.from_uci(moves);
            state.movesFromRoot.push_back(move);
        }
    }else if(command == "isready"){
        printf("readyok\n");
    }else if(command == "d"){
        for(Move move:state.movesFromRoot){
            state.root.playPartialMove(move);
        }
        state.root.print();
        for(Move move:state.movesFromRoot)
            state.root.undoLastMove();
    }else if(command == "runQ"){
        bestMoveFinder.testQuiescenceSearch(state.root);
    }else if(command == "eval"){
        bestMoveFinder.eval.init(state.root);
        printf("static evaluation: %d cp\n", bestMoveFinder.eval.getScore(state.root.friendlyColor(), state.root.getPawnStruct()  ));
    }else if(command == "stop"){
        bestMoveFinder.stop();
    }else if(command == "ucinewgame"){
        bestMoveFinder.clear();
    }
    //Implement actual logic for UCI management
}

int main(int argc, char** argv){
    string UCI_instruction = "programStart";
    Chess state;
    state.root.fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
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