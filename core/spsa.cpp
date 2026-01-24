bool DEBUG = false;
#include "BestMoveFinder.hpp"
#include "Const.hpp"
#include "Evaluator.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Move.hpp"
#include "tunables.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <random>
#include <cmath>
using namespace std;

BestMoveFinder* players;

vector<string> fens;
int idFen;
int nbGamesPerIter;
int nbIters;
int nbThreads;
int baseTime, increment;
int moveOverhead = 100;

class stateIter{
public:
    int penta[5];
    tunables parameters;
    int idSPSA;
    int idPair;
    int nbFinished;
    int nbLaunched;
    vector<pair<int, string>> pairs;
    vector<float> randoms;
    map<int, int> M;
    void init(int id, tunables& globParams){
        randoms.clear();
        M.clear();
        pairs.clear();
        idPair = 0;
        idSPSA = id;
        memset(penta, 0, sizeof(penta));
        parameters = globParams;
        pairs.resize(nbGamesPerIter/2);
        nbFinished = 0;
        std::mt19937 gen(idSPSA);
        std::normal_distribution<double> d(0, 1);
        for(int i=0; i<(int)parameters.to_tune_int().size(); i++)
            randoms.push_back(d(gen));
        for(int i=0; i<(int)parameters.to_tune_float().size(); i++)
            randoms.push_back(d(gen));
    }

    string init_players(int id, bool& lastGame){
        M[id] = idPair;
        nbLaunched++;
        lastGame = nbLaunched == nbGamesPerIter;
        string fen;
        if(pairs[idPair].first&1){
            idPair++;
            fen = pairs[idPair].second;
        }else{
            pairs[idPair].first ^= 1;
            fen = pairs[idPair].second = fens[idFen++];
        }
        id *= 2;
        for(int idPlayer=0; idPlayer<2; idPlayer++){
            int sign = idPlayer?-1:1;
            vector<int*> Vs = players[id+idPlayer].parameters.to_tune_int();
            vector<int*> VBs = parameters.to_tune_int();
            for(int i = 0; i<(int)VBs.size(); i++){
                *Vs[i] = *VBs[i]+*VBs[i]*randoms[i]*sign/100;
            }
            vector<float*> Vfs = players[id+idPlayer].parameters.to_tune_float();
            vector<float*> VBfs = parameters.to_tune_float();
            for(int i = 0; i<(int)VBfs.size(); i++){
                *Vfs[i] = *VBfs[i]+*VBfs[i]*randoms[i+VBs.size()]*sign/100;
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

    double diffloss(){
        //code used from fastchess
        const int pairs_ = penta[0]+penta[1]+penta[1]+penta[1]+penta[4];
        const double WW       = double(penta[4]) / pairs_;
        const double WD       = double(penta[3]) / pairs_;
        const double WLDD     = double(penta[2]) / pairs_;
        const double LD       = double(penta[1]) / pairs_;
        const double LL       = double(penta[0]) / pairs_;
        double score = WW + WD*0.75 + WLDD*0.5 + LD*0.25;
        const double WW_dev   = WW   * std::pow((1    - score), 2);
        const double WD_dev   = WD   * std::pow((0.75 - score), 2);
        const double WLDD_dev = WLDD * std::pow((0.5  - score), 2);
        const double LD_dev   = LD   * std::pow((0.25 - score), 2);
        const double LL_dev   = LL   * std::pow((0    - score), 2);
        const double variance = WW_dev + WD_dev + WLDD_dev + LD_dev + LL_dev;
        return (score-0.5)/sqrt(2*variance)*(800/log(10));
    }

    void apply(tunables& globals){
        double loss = diffloss();
        int ind=0;
        for(int* v:globals.to_tune_int())
            *v += loss*randoms[ind++]/100;
        for(float* v:globals.to_tune_float())
            *v += loss*randoms[ind++]/100;
    }
};


class HelperThread{
public:
    int id;
    thread t;
    string fen;
    bool running;
    mutex mtx;
    condition_variable cv;
    int ans;
    bool zombie;
    void init(){
        zombie = false;
        running = false;
    }
    void launch(string _pos){ 
        ans = 0;
        fen = _pos;
        {
            lock_guard<mutex> lock(mtx);
            running = true;
        }
        cv.notify_one();
    }
    void wait_notification(){
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this]{return running;});
    }
    void set_result(int result){
        ans = result;
        {
            lock_guard<mutex> lock(mtx);
            running = false;
            cv.notify_one();
        }
    }
    bool check_finished(){
        lock_guard<mutex> lock(mtx);
        return running || zombie;
    }
    void bezombie(){
        zombie = true;
    }
};

vector<HelperThread> threads;
void play_games(int id){
    HelperThread& ss=threads[id];
    GameState* state = new GameState;
    IncrementalEvaluator* eval = new IncrementalEvaluator;
    Move legalMoves[maxMoves];
    LegalMoveGenerator generator;
    bool inCheck;
    big dngpos;
    while(1){
        ss.wait_notification();
        if(ss.fen == "-")return;
        int times[2] = {baseTime, baseTime};
        int result = 1;
        state->fromFen(ss.fen);
        eval->init(*state);
        int player = (state->friendlyColor() == BLACK);
        int ply = 0;
        while(1){
            auto start = chrono::steady_clock::now();
            auto res=players[id*2+player].goState(*state, TM(moveOverhead, times[0], times[1], increment, increment, state->friendlyColor()), false, true, ply);
            auto end = start.time_since_epoch();
            times[player] -= chrono::duration_cast<chrono::milliseconds>(end).count();
            if(times[player] < 0){
                result = (state->enemyColor() == WHITE)*2;
                break;
            }
            times[player] += increment;
            Move bm=get<0>(res);
            state->playMove(bm);
            if(state->threefold() || state->rule50_count() >= 100){
                break;
            }
            eval->playNoBack(bm, state->enemyColor());
            if(eval->isInsufficientMaterial())break;
            generator.initDangers(*state);
            int nbMoves = generator.generateLegalMoves(*state, inCheck, legalMoves, dngpos, false);
            if(nbMoves == 0){
                if(inCheck){
                    result = (state->enemyColor() == WHITE)*2;
                }else
                    result = 1;
                break;
            }
            player ^= 1;
        }
        ss.set_result(result);
    }
}

int main(int argc, char** argv){
    if(argc == 1){
        printf("usage: %s <book> <nbGames per Iter> <nbIter> <nbThreads> (optional:) <memory> <baseTime> <increment>\n", argv[0]);
        return 0;
    }
    string book(argv[1]);
    ifstream file(book);
    idFen = 0;
    string curFen;
    while(getline(file, curFen))
        fens.push_back(curFen);
    nbGamesPerIter = atoi(argv[2]);
    nbIters = atoi(argv[3]);
    nbThreads = atoi(argv[4]);
    int memory = 16;
    baseTime = 10000, increment = 100;
    if(argc > 5){
        memory = atoi(argv[5]);
        baseTime = atoi(argv[6]);
        increment = atoi(argv[7]);
    }
    memory *= hashMul;
    players = (BestMoveFinder*)calloc(nbThreads*2, sizeof(BestMoveFinder));
    for(int i=0; i<nbThreads*2; i++){
        players[i].reinit(memory);
    }
    tunables state;
    vector<stateIter> Q;
    int idSPSA = 0;
    stateIter S;
    S.init(idSPSA++, state);
    Q.push_back(S);
    vector<stateIter*> games(nbThreads, NULL);
    printf("start tuning with %ld parameters %d threads tc=%.1f+%.1f memory=%dB %d iters %d games per iter\n", state.to_tune_int().size()+state.to_tune_float().size(), nbThreads, baseTime/1000.0, increment/1000.0, memory, nbIters, nbGamesPerIter);
    for(int i=0; i<nbThreads; i++){
        threads[i].t = thread(play_games, i);
        bool islast;
        string fen = Q.back().init_players(i, islast);
        threads[i].launch(fen);
        games[i] = &Q.back();
        if(islast){
            stateIter nS;
            nS.init(idSPSA++, state);
            Q.push_back(nS);
        }
    }
    int threadUp = nbThreads;
    while(threadUp){
        for(int i=0; i<nbThreads; i++){
            if(threads[i].check_finished()){
                if(games[i]->add_result(threads[i].ans, i)){
                    games[i]->apply(state);
                }
                bool islast;
                if(Q.back().nbLaunched < nbGamesPerIter){
                    string fen = Q.back().init_players(i, islast);
                    threads[i].launch(fen);
                    games[i] = &Q.back();
                }
                if(islast){
                    if(idSPSA >= nbIters){
                        threadUp--;
                        threads[i].bezombie();
                        continue;
                    }
                    stateIter nS;
                    nS.init(idSPSA++, state);
                    Q.push_back(nS);
                }
            }
        }
    }
}
