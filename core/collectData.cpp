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
void fastWrite(T& data, ofstream& file){
    file.write(reinterpret_cast<const char*>(&data), sizeof(data));
}

class MoveInfo{
public:
    int16_t moveInfo;
    int score;
    int staticScore;
    bool isVoid;
    MoveInfo(){
        moveInfo = 0;
        score = 0;
        staticScore = 0;
        isVoid = false;
    }
    void dump(ofstream& datafile){
        fastWrite(isVoid, datafile);
        fastWrite(moveInfo, datafile);
        fastWrite(score, datafile);
        fastWrite(staticScore, datafile);
    }
    static const int size = sizeof(isVoid)+sizeof(moveInfo)+sizeof(score)+sizeof(staticScore);
};
class GamePlayed{
public:
    vector<MoveInfo> game;
    GameState startPos;
    ubyte result;
    static const int headerSize = sizeof(GameState::boardRepresentation)+sizeof(ubyte)+sizeof(dbyte);

    void dump(ofstream& datafile){
        for(int i=0; i<6; i++)
            for(int j=0; j<2; j++)
                fastWrite(startPos.boardRepresentation[j][i], datafile);
        ubyte gameInfo = result*2+startPos.friendlyColor(); //the result (0-1-2) and the color of the starting player
        fastWrite(gameInfo, datafile);
        dbyte sizeGame = game.size();
        fastWrite(sizeGame, datafile);//length of the game
        for(MoveInfo moves:game){
            moves.dump(datafile);
        }
    }
    void clear(){
        game.clear();
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
    LegalMoveGenerator generator;
    int lastGamesMade=0;
    int realThread;
    #pragma omp parallel
    #pragma omp single
    realThread = min(omp_get_num_threads(), sizeGame);
    globnnue = NNUE(argv[2]);
    big nodesSearched = 0;
    #pragma omp parallel for shared(gamesMade, lastGamesMade) private(generator)
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
                pointer += GamePlayed::headerSize-sizeof(dbyte);
                nbGames++;
                dbyte nbMoves=0;
                assert(pointer < fileSize);
                infile.seekg(pointer);
                infile.read(reinterpret_cast<char*>(&nbMoves), sizeof(nbMoves));
                pointer += MoveInfo::size*nbMoves+sizeof(dbyte);
                totalMoves += nbMoves;
            }
            printf("file %d finding %d games (%d moves in total) delta %d\n", idThread, nbGames, totalMoves, pointer-fileSize);
            startReg += nbGames;
            #pragma omp atomic update
            lastGamesMade += nbGames;
        }
        int endReg = sizeGame*(idThread+1)/realThread;
        IncrementalEvaluator* eval = new IncrementalEvaluator;
        BestMoveFinder* players[2];
        players[0] = new BestMoveFinder(alloted_space, true);
        players[1] = new BestMoveFinder(alloted_space, true);
        GamePlayed* Game = new GamePlayed;
        GameState* current = new GameState;
        Move LegalMoves[maxMoves];
        for(int i=startReg; i<endReg; i++){
            //printf("begin thread %d loop %d\n", omp_get_thread_num(), i);
            players[0]->clear();
            players[1]->clear();
            current->fromFen(fens[i]);
            vector<Move> moves;
            int result = 1; //0 black win 1 draw 2 white win
            big dngpos;
            int countMoves = 0;
            Game->startPos.fromFen(fens[i]);
            Game->clear();
            big localNodes = 0;
            do{
                bool isWhite = current->friendlyColor() == WHITE;
                bestMoveResponse res;
                TM tm(limitNodes, limitNodes*1000);
                res = players[!isWhite]->goState<1>(*current, tm, false, false, moves.size());
                vector<depthInfo> infos = get<3>(res);
                if(!infos.empty())
                    localNodes += infos.back().node;
                int score = get<2>(res);
                Move curMove = get<0>(res);
                if(abs(score) > MAXIMUM-maxDepth){
                    result = (score > 0)*2;
                    if(current->friendlyColor() == BLACK)
                        result = 2-result;
                    break;
                }
                if(current->playMove(curMove) == 3){
                    result = 1;
                    break;
                }
                MoveInfo curProc;
                curProc.moveInfo = curMove.moveInfo;
                if(score != INF){
                    curProc.score = score;
                    eval->init(*current);
                    curProc.staticScore = eval->getRaw(current->friendlyColor());
                    curProc.isVoid = false;
                }else{
                    curProc.isVoid = true;
                }
                Game->game.push_back(curProc);
                countMoves++;
                if(curMove.isTactical())
                    countMoves = 0;
                bool inCheck;
                generator.initDangers(*current);
                int nbMoves = generator.generateLegalMoves(*current, inCheck, LegalMoves, dngpos, false);
                if(nbMoves == 0){
                    if(inCheck) result = (current->enemyColor() == WHITE)*2;
                    else result = 1;
                    break;
                }
                moves.push_back(curMove);
            }while(countMoves < 100);
            Game->result = result;
            ofstream datafile(nameDataFile, ios::app);
            Game->dump(datafile);
            datafile.close();
            #pragma omp critical
            {
                #pragma omp atomic update
                gamesMade++;
                #pragma omp atomic update
                nodesSearched += localNodes;
                int totGamesMade = lastGamesMade+gamesMade;
                struct winsize w;
                ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
                auto end=chrono::high_resolution_clock::now();
                big duration = chrono::duration_cast<chrono::milliseconds>(end-start).count();
                string unit;
                int speed;
                big nps = nodesSearched*1000/duration;
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
                printed += unit + " " + to_string(nps)+"nps [";
                int percentWind = (w.ws_col-printed.size()-1)*totGamesMade*10/sizeGame;
                printed += string(percentWind/10, '#');
                if(totGamesMade != sizeGame){
                    printed += to_string(percentWind%10);
                    printed += string((w.ws_col-printed.size()-1), ' ');
                }
                printed += "]";
                printf("%s\r", printed.c_str());
                fflush(stdout);
            }
        }
        delete current;
        delete players[0];
        delete players[1];
        delete eval;
    }
    printf("\n");
}