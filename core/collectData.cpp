int nbThreads = 1;
#include "BestMoveFinder.hpp"
#include "Evaluator.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include <fstream>
#include <vector>
#include <filesystem>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <cassert>
#include <omp.h>
//#define DEBUG
using namespace std;
const int alloted_space = 2*1000*1000;

string secondsToStr(big s){
    string res="";
    if(s >= 60){
        big m=s/60;
        s %= 60;
        if(m >= 60){
            big h=m/60;
            m %= 60;
            if(h >= 24){
                big d=h/24;
                h %= 24;
                res += to_string(d)+"d ";
            }
            res += to_string(h)+"h ";
        }
        res += to_string(m)+"m ";
    }
    res += to_string(s)+"s ";
    return res;
}

template<typename T> 
void fastWrite(T data, FILE* file){
    fwrite(reinterpret_cast<const char*>(&data), sizeof(data), 1, file);
}

class MoveInfo{
public:
    Move move;
    int score;
    MoveInfo(){
        move = nullMove;
        score = 0;
    }
    void dump(FILE* datafile){
        uint16_t mv = move.getMovePart();
        int type = 0;
        if(move.promotion() > 0){
            mv |= (move.promotion()-1) << 12;
            type = 3;
        }else if(move.capture == -1)
            type = 1;
        else if(move.piece == KING && abs(move.from()-move.to()) == 2){
            type = 2;
            if(move.from() > move.to())
                mv &= ~7;
            else
                mv |= 7;
        }
        mv ^= 0x0707;
        mv |= type << 14;
        fastWrite(mv, datafile);
        fastWrite<int16_t>(score, datafile);
    }
    static const int size = 4;
};
class GamePlayed{
public:
    vector<MoveInfo> game;
    GameState startPos;
    ubyte result;
    static const int headerSize = 8+16+8;

    void dump(FILE* datafile){
        big occupied = 0;
        for(int i=0; i<6; i++)
            for(int j=0; j<2; j++)
                occupied |= startPos.boardRepresentation[j][i];
        fastWrite(reverse_col(occupied), datafile);
        int8_t entry = 0x00;
        bool isSec = false;
        int nbEntry = 0;
        big castle = startPos.castlingMask();
        for(int i=0; i<64; i++){
            if(((1ULL << (i^0x07)) & occupied)){
                int8_t piece = startPos.getfullPiece(i^0x07);
                int _c = color(piece);
                piece = type(piece);
                if(piece == ROOK && ((1ULL << i)&castle))
                    piece = 6;
                entry = (entry << 4) | (_c << 3) | piece;
                if(isSec)
                    fastWrite(entry, datafile);
                isSec ^= 1;
                nbEntry += 1;
            }
        }
        for(int i=nbEntry; i<32; i++){
            entry = entry << 4 | 0;
            if(isSec){
                fastWrite(entry, datafile);
            }
            isSec ^= 1;
        }
        uint8_t info = startPos.lastDoublePawnPush == -1 ? 64 : startPos.lastDoublePawnPush^0x07;
        info |= startPos.friendlyColor() << 7;
        fastWrite(info, datafile);
        fastWrite<uint8_t>(0, datafile);  // halfmove clock (for 50 move rule)
        fastWrite<uint16_t>(0, datafile); // full move
        fastWrite<uint16_t>(0, datafile); //score of the position
        fastWrite<uint8_t>(result, datafile); // result
        fastWrite<uint8_t>(0, datafile); //unused extra byte
        for(MoveInfo moves:game){
            moves.dump(datafile);
        }
        fastWrite<uint32_t>(0, datafile);
    }
    void clear(){
        game.clear();
    }
};

class threadHelper{
public:
    BestMoveFinder player0;
    BestMoveFinder player1;
    IncrementalEvaluator eval;
    LegalMoveGenerator generator;
    GameState state;
    GamePlayed game;
    Move legalMoves[maxMoves];
    threadHelper():player0(alloted_space, true), player1(alloted_space, true){}
    void init(string fen){
        player0.clear();
        player1.clear();
        game.startPos.fromFen(fen);
        state.fromFen(fen);
        eval.init(state);
        game.clear();
    }
    BestMoveFinder& getPlayer(){
        if(state.friendlyColor() == WHITE)
            return player0;
        else return player1;
    }
};

int main(int argc, char** argv){
    PrecomputeKnightMoveData();
    init_lines();
    precomputePawnsAttack();
    precomputeCastlingMasks();
    precomputeNormlaKingMoves();
    precomputeDirections();
    init_zobrs();
    load_table();
    ifstream file(argv[1]);
    vector<string> fens;
    string curFen;
    int limitNodes;
    if(argc > 3)
        limitNodes = atoi(argv[3]);
    else
        limitNodes = 200000;
    while(getline(file, curFen))
        fens.push_back(curFen);
    int sizeGame = fens.size();
    if(argc > 4)
        sizeGame = atoi(argv[4]);
    big gamesMade = 0;
    auto start=chrono::high_resolution_clock::now();
    int lastGamesMade=0;
    int realThread;
#ifndef DEBUG
    #pragma omp parallel
    #pragma omp single
#endif
    realThread = min(omp_get_num_threads(), sizeGame);
    if(argc > 5)realThread = atoi(argv[5]);
    globnnue = NNUE(argv[2]);
    big nodesSearched = 0;
#ifndef DEBUG
    #pragma omp parallel for shared(gamesMade, lastGamesMade, nodesSearched)
#endif
    for(int idThread=0; idThread<realThread; idThread++){
        int startReg=sizeGame*idThread/realThread;
        string nameDataFile = string("data")+to_string(idThread)+string(".out");
        ifstream infile(nameDataFile);
        if(infile.is_open()){
            int pointer=0;
            int fileSize = filesystem::file_size(nameDataFile);
            int nbGames=0;
            int totalMoves = 0;
            while(pointer < fileSize){
                pointer += GamePlayed::headerSize;
                nbGames++;
                assert(pointer < fileSize);
                uint32_t bytes = 0;
                infile.seekg(pointer);
                do{
                    infile.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
                    totalMoves += bytes != 0;
                    pointer += 4;
                }while(bytes);
            }
            printf("file %d finding %d games (%d moves in total) delta %d\n", idThread, nbGames, totalMoves, pointer-fileSize);
            startReg += nbGames;
#ifndef DEBUG
            #pragma omp atomic update
#endif
            lastGamesMade += nbGames;
        }
        int endReg = sizeGame*(idThread+1)/realThread;
        threadHelper* state = new threadHelper;
        FILE* fptr;
        fptr = fopen(nameDataFile.c_str(), "ab");
        for(int i=startReg; i<endReg; i++){
            //printf("begin thread %d loop %d\n", omp_get_thread_num(), i);
            state->init(fens[i]);
            int result = 1; //0 black win 1 draw 2 white win
            big dngpos;
            big localNodes = 0;
            do{
                bestMoveResponse res;
                TM tm(limitNodes, limitNodes*1000);
                res = state->getPlayer().goState<1>(state->state, tm, false, false, state->game.game.size());
                vector<depthInfo> infos = get<3>(res);
                if(!infos.empty())
                    localNodes += infos.back().node;
                int score = get<2>(res);
                Move curMove = get<0>(res);
                assert(curMove.moveInfo != nullMove.moveInfo);
                /*if(abs(score) > MAXIMUM-maxDepth){
                    result = (score > 0)*2;
                    if(state->state.friendlyColor() == BLACK)
                        result = 2-result;
                    break;
                }*/
                MoveInfo curProc;
                curProc.move = curMove;
                curProc.score = score;
                state->eval.playNoBack(curMove, state->state.friendlyColor());
                state->state.playMove(curMove);
                if(state->state.threefold()){
                    result = 1;
                    break;
                }
                state->game.game.push_back(curProc);
                bool inCheck;
                state->generator.initDangers(state->state);
                int nbMoves = state->generator.generateLegalMoves(state->state, inCheck, state->legalMoves, dngpos, false);
                if(nbMoves == 0){
                    if(inCheck){
                        if(score == 0)printf("\n%s\n", state->state.toFen().c_str());
                        result = (state->state.enemyColor() == WHITE)*2;
                    }else
                        result = 1;
                    break;
                }
                if(state->eval.isInsufficientMaterial())break;
            }while(state->state.rule50_count() < 100);
            state->game.result = result;
            state->game.dump(fptr);
            fflush(fptr);
#ifndef DEBUG
            #pragma omp atomic update
#endif
            gamesMade++;
#ifndef DEBUG
            #pragma omp atomic update
#endif
            nodesSearched += localNodes;
            int totGamesMade = lastGamesMade+gamesMade;
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            auto end=chrono::high_resolution_clock::now();
            big duration = chrono::duration_cast<chrono::milliseconds>(end-start).count();
            string unit;
            int speed;
            big nps = nodesSearched*1000/duration/realThread;
            if(duration > gamesMade*1000){
                unit = "s/g";
                speed = duration*100/(gamesMade*1000);
            }else{
                speed = gamesMade*1000*100/duration;
                unit = "g/s";
            }
            string remaindTime = secondsToStr(duration*(sizeGame-totGamesMade)/(gamesMade*1000)); // in seconds
            int percent = 1000*totGamesMade/sizeGame;
            string printed = to_string(percent/10)+string(".")+to_string(percent%10);
            printed += "% (";
            printed += to_string(totGamesMade)+string("/")+to_string(sizeGame)+" in ";
            printed += to_string(duration/1000.0)+"s) "+remaindTime+" ";
            printed += to_string(speed/100);
            if(speed%100 >= 10)
                printed += string(".")+to_string(speed%100);
            else 
                printed += string(".0")+to_string(speed%100);
            printed += unit + " " + to_string(nps)+"npst [";
            int percentWind = (w.ws_col-printed.size()-1)*totGamesMade*10/sizeGame;
            printed += string(percentWind/10, '#');
            if(totGamesMade != sizeGame){
                printed += to_string(percentWind%10);
                printed += string((w.ws_col-printed.size()-1), ' ');
            }
            printed += "]";
#ifndef DEBUG
            #pragma omp critical
#endif
            {
                printf("%s\r", printed.c_str());
                fflush(stdout);
            }
        }
        fclose(fptr);
        delete state;
    }
    printf("\n");
    clear_table();
}