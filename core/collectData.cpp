#include "BestMoveFinder.hpp"
#include <fstream>
#include <vector>
#include <omp.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>
const int alloted_space = 64*1000*1000;
//int omp_get_thread_num(){return 0;}
//#define DEBUG
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
    int gamesMade = 0;
    auto start=chrono::high_resolution_clock::now();
    LegalMoveGenerator generator;
#ifndef DEBUG
    #pragma omp parallel for shared(gamesMade) private(generator) num_threads(70)
#endif
    for(int i=0; i<fens.size(); i++){
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
        int result; //0 black win 1 draw 2 white win
        Move LegalMoves[maxMoves];
        big dngpos;
        int countMoves = 0;
        vector<int> scores[2];
        vector<int> statics[2];
        vector<string> movesPlayed[2];
        vector<string> playedFens[2];
        do{
            bool isWhite = current.friendlyColor() == WHITE;
            root.fromFen(fens[i]);
            pair<Move, int> res;
            if(isWhite)res = player.bestMove<1>(root, limitNodes, moves, false);
            else res = opponent.bestMove<1>(root, limitNodes, moves, false);
#ifdef DEBUG
            //printf("%s\n", current.toFen().c_str());
#endif
            moves.push_back(res.first);
            if(abs(res.second) == INF+1){
                result = (res.second > 0)*2;
                break;
            }
            if(res.second != INF){
                scores[isWhite].push_back(res.second);
                player.eval.init(current);
                statics[isWhite].push_back(player.eval.getScore(current.friendlyColor(), current.getPawnStruct()));
                playedFens[isWhite].push_back(current.toFen());
                movesPlayed[isWhite].push_back(res.first.to_str());
            }
            if(current.playMove<false>(res.first) == 3){
                result = 1;
                break;
            }
            countMoves++;
            if(res.first.isTactical())
                countMoves = 0;
            bool inCheck;
            int nbMoves = generator.generateLegalMoves(current, inCheck, LegalMoves, dngpos, false);
            if(nbMoves == 0){
                if(inCheck) result = (current.enemyColor() == WHITE)*2;
                else result = 1;
                break;
            }
        }while(countMoves < 100);
        string nameDataFile = string("data")+to_string(omp_get_thread_num())+string(".out");
        ofstream datafile(nameDataFile, ios::app);
        for(int i=0; i<2; i++){
            for(int j=0; j<scores[i].size(); j++){
                datafile << playedFens[i][j] << "|" << scores[i][j] << "|" << statics[i][j] << "|" << movesPlayed[i][j] << "|" << result/2.0 << '\n';
            }
            result = 1-result;
        }
        #pragma omp critical
        {
            gamesMade++;
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            auto end=chrono::high_resolution_clock::now();
            int duration = chrono::duration_cast<chrono::milliseconds>(end-start).count();
            string unit;
            int speed;
            if(duration > gamesMade*1000){
                unit = "s/g";
                speed = duration*100/(gamesMade*1000);
            }else{
                speed = gamesMade*1000*100/duration;
                unit = "g/s";
            }
            int percent = 1000*gamesMade/fens.size();
            string printed = to_string(percent/10)+string(".")+to_string(percent%10);
            printed += "% (";
            printed += to_string(gamesMade)+string("/")+to_string(fens.size())+") ";
            printed += to_string(speed/100);
            if(speed%100 >= 10)
                printed += string(".")+to_string(speed%100);
            else 
                printed += string(".0")+to_string(speed%100);
            printed += unit+" [";
            int percentWind = (w.ws_col-printed.size()-1)*gamesMade*10/fens.size();
            printed += string(percentWind/10, '#');
            printed += to_string(percentWind%10);
            printed += string((w.ws_col-printed.size()-1), ' ');
            printed += "]";
            printf("%s\r", printed.c_str());
            fflush(stdout);
        }
    }
}