#include "BestMoveFinder.hpp"
#include "tunables.hpp"
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include <deque>
#include <random>
using namespace std;

vector<BestMoveFinder> players;

vector<string> fens;
int idFen;
int nbGamesPerIter;
int nbIters;
int nbThreads;

class stateIter{
public:
    int penta[5];
    tunables parameters;
    int idSPSA;
    int idPair;
    int nbFinished;
    vector<pair<int, string>> pairs;
    map<int, int> M;
    void init(int id, tunables& globParams){
        idPair = 0;
        idSPSA = id;
        memset(penta, 0, sizeof(penta));
        parameters = globParams;
        pairs.resize(nbGamesPerIter/2);
        nbFinished = 0;
    }

    string init_players(int id){
        M[id] = idPair;
        string fen;
        if(pairs[idPair].first&1){
            idPair++;
            fen = pairs[idPair].second;
        }else{
            pairs[idPair].first ^= 1;
            fen = pairs[idPair].second = fens[idFen];
        }
        id *= 2;
        for(int idPlayer=0; idPlayer<2; idPlayer++){
            int sign = idPlayer?-1:1;
            std::mt19937 gen(idSPSA);
            std::normal_distribution<double> d(0, 1);
            vector<int*> Vs = players[id+idPlayer].parameters.to_tune_int();
            vector<int*> VBs = parameters.to_tune_int();
            for(int i = 0; i<VBs.size(); i++){
                *Vs[i] = *VBs[i]+*VBs[i]*d(gen)*sign/100;
            }
            vector<float*> Vfs = players[id+idPlayer].parameters.to_tune_float();
            vector<float*> VBfs = parameters.to_tune_float();
            for(int i = 0; i<VBfs.size(); i++){
                *Vfs[i] = *VBfs[i]+*VBfs[i]*d(gen)*sign/100;
            }
        }
        return fen;
    }

    bool add_result(int result, int id){
        int i = M[id];
        if(pairs[i].first/2){
            penta[pairs[i].first/2+result-1]++;
        }else{
            pairs[i].first += (result+1)*2;
        }
        return (++nbFinished) == nbGamesPerIter;
    }
};

int main(int argc, char** argv){
    if(argc == 1){
        printf("usage: %s <book> <nbGames per Iter> <nbIter> <nbThreads> (optional:) <memory> <baseTime> <increment>\n", argv[0]);
        return 0;
    }
    string book(argv[1]);
    nbGamesPerIter = atoi(argv[2]);
    nbIters = atoi(argv[3]);
    nbThreads = atoi(argv[4]);
    int memory = 16;
    int baseTime = 10000, increment = 100;
    if(argc > 5){
        memory = atoi(argv[5]);
        baseTime = atoi(argv[6]);
        increment = atoi(argv[7]);
    }
    memory *= hashMul;
    players = vector<BestMoveFinder>(nbThreads*2, BestMoveFinder(memory));
    tunables state;
    deque<int> Q;
    int idSPSA = 0;
    for(int i=0; i<(nbThreads-1)/nbGamesPerIter+1; i++){
        Q.push_back(idSPSA++);
    }

}
