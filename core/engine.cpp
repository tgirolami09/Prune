#include <sstream>
#include <string>
#include <cstring>
#include <future>
// #include "util_magic.cpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "BestMoveFinder.hpp"

#include <cmath>
#include <iostream>

#ifdef _WIN32
#include <conio.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#endif
// big* table[128]; // [bishop, rook]
// info constants[128];

using namespace std;
//Main class for the game
const vector<string> benches = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpN3/3N2B1/4P3/7P/PPPQ1PP1/2KR3R b - - 0 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4PpP1/1BNP4/PPP2P1P/3R1RK1 b - g3 0 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 4 3",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
    "7k/7P/6K1/8/3B4/8/8/8 b - - 0 1"
};

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

void stop_calculations() {
    std::string inputBuffer;
    while (bestMoveFinder.running) {
#ifdef _WIN32
        if (_kbhit()) {
            char ch = _getch();
            if (ch == '\n' || ch == '\r') {
                if (inputBuffer == "stop") {
                    bestMoveFinder.running = false; // Signal the worker to stop
                }
                inputBuffer.clear(); // Clear buffer after newline
            } else {
                inputBuffer += ch; // Build the input string
            }
        }
#else
        // Check if input is available
        fd_set readfds;
        struct timeval tv;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms wait for input

        if (select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv) > 0) {
            char ch;
            read(STDIN_FILENO, &ch, 1);
            if (ch == '\n') {
                if (inputBuffer == "stop") {
                    bestMoveFinder.running = false; // Signal the worker to stop
                }
                inputBuffer.clear(); // Clear buffer after newline
            } else {
                inputBuffer += ch; // Build the input string
            }
        }
#endif
    }
}

template<int limitWay=0>
Move getBotMove(Chess& state, int softBound, int hardBound){
    bestMoveFinder.running = true;
#ifdef STOP_POSS
    thread t(&stop_calculations);
#endif
    Move moveToPlay = get<0>(bestMoveFinder.bestMove<limitWay>(state.root, softBound, hardBound, state.movesFromRoot));
    printf("bestmove %s\n", moveToPlay.to_str().c_str());
#ifdef STOP_POSS
    t.join();
#endif
    return moveToPlay;
}

Move getOpponentMove(){
    //TODO : Some logic to get it from i/o
    return {};
}

pair<int, int> computeAllotedTime(Chess& state){
    bool color = state.root.friendlyColor()^(state.movesFromRoot.size()&1);
    int time = color == WHITE?state.w_time:state.b_time;
    int inc = color == WHITE?state.winc:state.binc;
    int hardBound = time/20+inc/2-moveOverhead;
    int softBound = hardBound/2;
    //maxTime = min(maxTime, time/10);
    return {softBound, hardBound};
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
            }else if(arg == "nnueFile"){
                stream >> inter;
                bestMoveFinder.eval.nnue = NNUE(inter);
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
            printf("Nodes searched: %" PRId64 "\n", doPerft.perft(state.root, args["perft"]));
        }else{
            Move move;
            if(args.count("btime") && args.count("wtime")){
                state.b_time = args["btime"];
                state.w_time = args["wtime"];
                state.winc = args["winc"];
                state.binc = args["binc"];
                auto [softBound, hardBound] = computeAllotedTime(state);
                move=getBotMove(state, softBound, hardBound);
            }else if(args.count("movetime")){
                move = getBotMove(state, args["movetime"], args["movetime"]);
            }else if(args.count("nodes")){
                move = getBotMove<1>(state, args["nodes"], args["nodes"]);
            }else if(args.count("depth")){
                move = getBotMove<2>(state, args["depth"], args["depth"]);
            }else{
                move = getBotMove<2>(state, 200, 200);
            }
            return;
        }
    }else if(command == "uci"){
        printf("id name pruningBot\nid author tgirolami09 & jbienvenue\n");
        printf("option name Hash type spin default 64 min 1 max 512\n");
        printf("option name Move Overhead type spin default 10 min 0 max 5000\n");
        printf("option name Clear Hash type button\n");
        printf("option name nnueFile type string default model.bin\n");
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
        for(unsigned long i=0; i<state.movesFromRoot.size(); i++)
            state.root.undoLastMove();
    }else if(command == "runQ"){
        bestMoveFinder.testQuiescenceSearch(state.root);
    }else if(command == "eval"){
        for(Move move:state.movesFromRoot)
            state.root.playPartialMove(move);
        bestMoveFinder.eval.init(state.root);
        int overall_eval = bestMoveFinder.eval.getScore(state.root.friendlyColor());
        for(int r=7; r >= 0; r--){
            pair<char, int> evals[8];
            for(int c=0; c < 8; c++){
                int square = (r << 3) | c;
                int piece = state.root.getfullPiece(square);
                if(type(piece) != SPACE){
                    bestMoveFinder.eval.nnue.change2<-1>(piece, square);
                    char repr=id_to_piece[type(piece)];
                    int derived = overall_eval-bestMoveFinder.eval.getScore(state.root.friendlyColor());
                    if(color(piece) == WHITE)repr = toupper(repr);
                    evals[7-c] = {repr, derived};
                    bestMoveFinder.eval.nnue.change2<1>(piece, square);
                }else evals[7-c] = {' ', 0};
            }
            for(int i=0; i<8; i++)
                printf("+-------");
            printf("+\n");
            for(int i=0; i<8; i++)
                printf("|   %c   ", evals[i].first);
            printf("|\n");
            for(int i=0; i<8; i++){
                if(evals[i].first == ' ')
                    printf("|       ");
                else{
                    if(abs(evals[i].second) > 10*100)
                        printf("| %+2.1f ", evals[i].second/100.0);
                    else
                        printf("| %+1.2f ", evals[i].second/100.0);
                }
            }
            printf("|\n");
        }
        for(int i=0; i<8; i++)
            printf("+-------");
        printf("+\n");

        for(unsigned long i=0; i<state.movesFromRoot.size(); i++)
            state.root.undoLastMove();
        printf("static evaluation: %d cp\n", overall_eval);
    }else if(command == "stop"){
        bestMoveFinder.stop();
    }else if(command == "ucinewgame"){
        bestMoveFinder.clear();
    }else if(command == "version"){
#ifdef COMMIT
        printf("version: %s\n", COMMIT);
#else
        printf("version: test\n");
#endif
#ifdef STOP_POSS
        printf("stop possibility version\n");
#endif
    }else if(command == "bench"){
        int sumNodes[maxDepth+1];
        int sumNPS = 0;
        int sumTime = 0;
        int histDepth[maxDepth+1];
        int sumSelDepth[maxDepth+1];
        for(int i=0; i<=maxDepth; i++){
            sumNodes[i] = 0;
            histDepth[i] = 0;
            sumSelDepth[i] = 0;
        }
        int maxDepthAttain = 0;
        int moveTime = args.count("movetime")?args["movetime"]:100;
        vector<pair<int, int>> Scores;
        for(unsigned idFen=0; idFen<benches.size(); idFen++){
            printf("\rposition %d/%d", idFen, (int)benches.size());
            fflush(stdout);
            GameState pos;
            pos.fromFen(benches[idFen]);
            bestMoveFinder.clear();
            vector<depthInfo> infos = get<2>(bestMoveFinder.bestMove<0>(pos, moveTime, moveTime, {}, false));
            for(depthInfo info:infos){
                sumNodes[info.depth] += info.node;
                histDepth[info.depth]++;
                sumSelDepth[info.depth] += info.seldepth;
                if(info.depth == maxDepth){
                    printf("\n%s\n", benches[idFen].c_str());
                }
                maxDepthAttain = max(maxDepthAttain, info.depth);
            }
            if(infos.size()){
                depthInfo lastInfo = infos[infos.size()-1];
                sumNPS += lastInfo.node;
                sumTime += lastInfo.time;
                Scores.push_back({lastInfo.depth, lastInfo.node});
            }
        }
        printf("\rposition %" PRId64 "/%" PRId64 "\n", benches.size(), benches.size());
        printf("depth\t");
        for(int i=0; i<=maxDepthAttain; i++)
            printf("\t%d", i);
        printf("\nnodes\t");
        for(int i=0; i<=maxDepthAttain; i++){
            if(histDepth[i])
                printf("\t%d", sumNodes[i]/histDepth[i]);
            else printf("\t0");
        }
        printf("\nseldepth");
        for(int i=0; i<=maxDepthAttain; i++){
            if(histDepth[i])
                printf("\t%d", sumSelDepth[i]/histDepth[i]);
            else printf("\t0");
        }
        printf("\n%.0fnps\n", sumNPS*1000.0/sumTime);
        sort(Scores.begin(), Scores.end());
        int size = Scores.size();
        pair<double, double> scoreThird = {0.0, 0.0}, scoreAll={0.0, 0.0};
        for(int i=0; i<size; i++){
            pair<double, double> locScore = {Scores[i].first, Scores[i].second};
            scoreAll.first += locScore.first;
            scoreAll.second += locScore.second;
            if(i >= size/3 && i < size*2/3){
                scoreThird.first += locScore.first;
                scoreThird.second += locScore.second;
            }
        }
        printf("search score: (%.0f %.0f) (%.0f %.0f)", scoreThird.first, scoreThird.second, scoreAll.first, scoreAll.second);
    }else if(command == "arch"){
#ifdef __AVX512F__
        printf("arch: AVX512");
#elif defined(__AVX2__)
        printf("arch: AVX2\n");
#elif defined(__AVX__)
        printf("arch: AVX\n");
#elif defined(__SSE2__)
        printf("arch: SSE2\n");
#else
        printf("arch: unknow\n");
#endif
    }
    //Implement actual logic for UCI management
    return;
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