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
    big gamesMade = 0;
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
        int result = 1; //0 black win 1 draw 2 white win
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
            tuple<Move, int, vector<depthInfo>> res;
            if(isWhite)res = player.bestMove<1>(root, limitNodes, limitNodes*1000, moves, false, false);
            else res = opponent.bestMove<1>(root, limitNodes, limitNodes*1000, moves, false, false);
#ifdef DEBUG
            //printf("%s\n", current.toFen().c_str());
#endif
            moves.push_back(get<0>(res));
            if(abs(get<1>(res)) == INF+1){
                result = (get<1>(res) > 0)*2;
                break;
            }
            if(get<1>(res) != INF){
                scores[isWhite].push_back(get<1>(res));
                player.eval.init(current);
                statics[isWhite].push_back(player.eval.getScore(current.friendlyColor()));
                playedFens[isWhite].push_back(current.toFen());
                movesPlayed[isWhite].push_back(get<0>(res).to_str());
            }
            if(current.playMove<false>(get<0>(res)) == 3){
                result = 1;
                break;
            }
            countMoves++;
            if(get<0>(res).isTactical())
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
            result = 2-result;
        }
        #pragma omp critical
        {
            gamesMade++;
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
            string remaindTime = secondsToStr(duration*(fens.size()-gamesMade)/(gamesMade*1000)); // in seconds
            int percent = 1000*gamesMade/fens.size();
            string printed = to_string(percent/10)+string(".")+to_string(percent%10);
            printed += "% (";
            printed += to_string(gamesMade)+string("/")+to_string(fens.size())+" in ";
            printed += to_string(duration/1000.0)+"s) "+remaindTime+" ";
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