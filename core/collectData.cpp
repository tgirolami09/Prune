#include "BestMoveFinder.hpp"
#include <fstream>
#include <vector>
#include <filesystem>
#include <omp.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
const int alloted_space = 64*1000*1000;
//int omp_get_thread_num(){return 0;}
//#define DEBUG
#define NUM_THREADS 70
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
};

int main(int argc, char** argv){
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
    int realThread = min(NUM_THREADS, sizeGame);
#ifndef DEBUG
    #pragma omp parallel for shared(gamesMade, lastGamesMade) private(generator) num_threads(NUM_THREADS)
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
        for(int i=startReg; i<endReg; i++){
            //printf("begin thread %d loop %d\n", omp_get_thread_num(), i);
            BestMoveFinder player(alloted_space, true);
            BestMoveFinder opponent(alloted_space, true);
            player.eval.nnue = NNUE(argv[2]);
            opponent.eval.nnue = NNUE(argv[2]);
            GameState root;
            root.fromFen(fens[i]);
            GameState current;
            current.fromFen(fens[i]);
            vector<Move> moves;
            int result = 1; //0 black win 1 draw 2 white win
            Move LegalMoves[maxMoves];
            big dngpos;
            int countMoves = 0;
            GamePlayed Game;
            Game.startPos.fromFen(fens[i]);
            do{
                bool isWhite = current.friendlyColor() == WHITE;
                root.fromFen(fens[i]);
                bestMoveResponseres;
                if(isWhite)res = player.bestMove<1>(root, limitNodes, limitNodes*1000, moves, false, false);
                else res = opponent.bestMove<1>(root, limitNodes, limitNodes*1000, moves, false, false);
                int score = get<2>(res);
                Move curMove = get<1>(res);
                if(abs(score) > MAXIMUM-maxDepth){
                    result = (score > 0)*2;
                    if(current.friendlyColor() == BLACK)
                        result = 2-result;
                    break;
                }
                if(current.playMove<false>(curMove) == 3){
                    result = 1;
                    break;
                }
                MoveInfo curProc;
                curProc.moveInfo = curMove.moveInfo;
                if(score != INF){
                    curProc.score = score;
                    player.eval.init(current);
                    curProc.staticScore = player.eval.getScore(current.friendlyColor());
                    curProc.isVoid = false;
                }else{
                    curProc.isVoid = true;
                }
                Game.game.push_back(curProc);
                countMoves++;
                if(curMove.isTactical())
                    countMoves = 0;
                bool inCheck;
                generator.initDangers(current);
                int nbMoves = generator.generateLegalMoves(current, inCheck, LegalMoves, dngpos, false);
                if(nbMoves == 0){
                    if(inCheck) result = (current.enemyColor() == WHITE)*2;
                    else result = 1;
                    break;
                }
                moves.push_back(curMove);
            }while(countMoves < 100);
            Game.result = result;
            ofstream datafile(nameDataFile, ios::app);
            Game.dump(datafile);
            #pragma omp critical
            {
                #pragma omp atomic update
                gamesMade++;
                int totGamesMade = lastGamesMade+gamesMade;
                struct winsize w;
                ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
                auto end=chrono::high_resolution_clock::now();
                big duration = chrono::duration_cast<chrono::milliseconds>(end-start).count();
                string unit;
                int speed;
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
                printed += unit+" [";
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
    }
    printf("\n");
}