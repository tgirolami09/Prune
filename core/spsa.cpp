bool DEBUG = false;
int nbThreads=1;
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
#include <vector>
#include <cassert>
using namespace std;
using namespace std::chrono;

vector<string> fens;
int idFen;
int nbGamesPerIter;
int nbIters;
int nbThreadsSPSA;
int baseTime, increment;
int memory;
int moveOverhead = 10;

class internalState{
public:
    vector<float> state;
    internalState(int nbParams):state(nbParams, 1){
    }
    internalState(tunables tun){
        state.resize(tun.to_tune_int().size()+tun.to_tune_float().size());
        int i = 0;
        for(int *j:tun.to_tune_int()){
            state[i++] = *j;
        }
        for(float *j:tun.to_tune_float()){
            state[i++] = *j;
        }
    }
    float getParam(int idx, int precision=1){
        return round(state[idx]*precision)/precision;
    }
    float getUpdate(int idx, float evolution, int precision=1){
        return round(state[idx]*(1+evolution)*precision)/precision;
    }
    void updateParam(int idx, float evolution){
        state[idx] *= (1+evolution);
        state[idx] = max(state[idx], 1.0f);
    }
    void print(ofstream& file){
        for(float i:state){
            file << i << "\t";
        }
        file << "\n";
        file.flush();
    }
};


class HelperThread{
public:
    int id;
    thread t;
    string fen;
    bool running;
    int ply;
    mutex mtx;
    condition_variable cv;
    BestMoveFinder player0, player1;
    GameState state;
    IncrementalEvaluator eval;
    HelperThread():player0(memory, true), player1(memory, true){}
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
    BestMoveFinder& getPlayer(int idPlayer){
        return idPlayer?player0:player1;
    }
    bestMoveResponse getEval(int idPlayer, TM tm){
        return getPlayer(idPlayer).goState<0>(state, tm, false, true, ply);
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
        return !running || zombie;
    }
    void bezombie(){
        zombie = true;
    }
};

HelperThread *threads;
class stateIter{
public:
    int penta[5];
    internalState* parameters;
    int idSPSA;
    int idPair;
    int nbFinished;
    int nbLaunched;
    vector<pair<int, string>> pairs;
    vector<float> randoms;
    map<int, int> M;
    void init(int id, internalState* globParams){
        randoms.clear();
        M.clear();
        pairs.clear();
        idPair = 0;
        idSPSA = id;
        memset(penta, 0, sizeof(penta));
        parameters = globParams;
        pairs.resize(nbGamesPerIter/2);
        nbLaunched = nbFinished = 0;
        std::mt19937 gen(idSPSA);
        std::normal_distribution<double> d(0, 1);
        for(int i=0; i<(int)parameters->state.size(); i++)
            randoms.push_back(d(gen));
    }

    string init_players(int id, bool& lastGame){
        M[id] = nbLaunched;
        nbLaunched++;
        lastGame = nbLaunched == nbGamesPerIter;
        string fen;
        bool isfirst = false;
        if(pairs[idPair].first&1){
            fen = pairs[idPair].second;
            idPair++;
        }else{
            pairs[idPair].first ^= 1;
            isfirst = true;
            fen = pairs[idPair].second = fens[idFen++];
        }
        int idPlayer = !isfirst;
        for(int _=0; _<2; _++){
            int sign = idPlayer?-1:1;
            vector<int*> Vs = threads[id].getPlayer(idPlayer).parameters.to_tune_int();
            for(int i = 0; i<(int)Vs.size(); i++){
                *Vs[i] = parameters->getUpdate(i, randoms[i]*sign/100);
            }
            vector<float*> Vfs = threads[id].getPlayer(idPlayer).parameters.to_tune_float();
            int offset = Vs.size();
            for(int i = 0; i<(int)Vfs.size(); i++){
                *Vfs[i] = parameters->getUpdate(i+offset, randoms[i+offset]*sign/100, 1000);
            }
            idPlayer ^= 1;
        }
        return fen;
    }

    bool add_result(int result, int id){
        int curPair = M[id]/2;
        int side = M[id]&1;
        if(side)result = 2-result;
        if(pairs[curPair].first/2){
            penta[pairs[curPair].first/2+result-1]++;
        }else{
            pairs[curPair].first += (result+1)*2;
        }
        return (++nbFinished) == nbGamesPerIter;
    }

    double diffloss(){
        //code used from fastchess
        const int pairs_ = penta[0]+penta[1]+penta[2]+penta[3]+penta[4];
        const double WW       = double(penta[4]) / pairs_;
        const double WD       = double(penta[3]) / pairs_;
        const double WLDD     = double(penta[2]) / pairs_;
        const double LD       = double(penta[1]) / pairs_;
        double score = WW + WD*0.75 + WLDD*0.5 + LD*0.25;
        return (score-0.5)*2;
    }

    void apply(float lr){
        double loss = diffloss();
        for(int i=0; i<(int)parameters->state.size(); i++)
            parameters->updateParam(i, loss*randoms[i]*lr/10);
    }
};

void play_games(int id){
    HelperThread& ss=threads[id];
    Move legalMoves[maxMoves];
    LegalMoveGenerator generator;
    bool inCheck;
    big dngpos;
    while(1){
        ss.wait_notification();
        if(ss.fen == "-")return;
        int times[2] = {baseTime, baseTime};
        int result = 1;
        ss.state.fromFen(ss.fen);
        ss.eval.init(ss.state);
        int player = (ss.state.friendlyColor() == BLACK);
        ss.ply = 0;
        while(1){
            auto start = high_resolution_clock::now();
            auto res=ss.getEval(player, TM(moveOverhead, times[0], times[1], increment, increment, ss.state.friendlyColor()));
            auto end = high_resolution_clock::now();
            int used_time = duration_cast<milliseconds>(end - start).count();
            Move bm=get<0>(res);
            if(bm.moveInfo == nullMove.moveInfo){
                printf("score: %d fen: %s ply: %d times: %d %d\n", get<2>(res), ss.state.toFen().c_str(), ss.ply, times[player], times[player^1]);
                assert(false);
            }
            times[player] -= used_time;
            if(times[player] < 0){
                printf("loss on time on thread %d, last time used = %d, time now remains=%d\n", id, used_time, times[player]);
                result = (ss.state.enemyColor() == WHITE)*2;
                break;
            }
            times[player] += increment;
            ss.eval.playNoBack(bm, ss.state.friendlyColor());
            ss.state.playMove(bm);
            ss.ply++;
            if(ss.state.threefold()){
                break;
            }
            if(ss.eval.isInsufficientMaterial()){
                break;
            }
            generator.initDangers(ss.state);
            int nbMoves = generator.generateLegalMoves(ss.state, inCheck, legalMoves, dngpos, false);
            if(nbMoves == 0){
                if(inCheck){
                    result = (ss.state.enemyColor() == WHITE)*2;
                }else{
                    result = 1;
                }
                break;
            }
            if(ss.state.rule50_count() >= 100){
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
    nbThreadsSPSA = atoi(argv[4]);
    memory = 16;
    baseTime = 10000, increment = 100;
    if(argc > 5){
        memory = atoi(argv[5]);
        baseTime = atoi(argv[6]);
        increment = atoi(argv[7]);
    }
    memory *= hashMul;
    internalState state((tunables()));
    int idSPSA = 0;
    stateIter S;
    S.init(idSPSA++, &state);
    vector<stateIter> Qiters;
    Qiters.push_back(S);
    vector<int> games(nbThreadsSPSA, -1);
    threads = new HelperThread[nbThreadsSPSA];
    float lr_min = 0.0001;
    float lr_max = 1;
    int nbPassedIters = 0;
    printf("start tuning with %ld parameters %d threads tc=%.1f+%.1f memory=%dB %d iters %d games per iter\n", state.state.size(), nbThreadsSPSA, baseTime/1000.0, increment/1000.0, memory, nbIters, nbGamesPerIter);
    for(int i=0; i<nbThreadsSPSA; i++){
        threads[i].t = thread(play_games, i);
        bool islast = false;
        string fen = Qiters.back().init_players(i, islast);
        threads[i].launch(fen);
        games[i] = Qiters.size()-1;
        if(islast){
            stateIter nS;
            nS.init(idSPSA++, &state);
            Qiters.push_back(nS);
        }
    }
    printf("all threads has been launched\n");
    int threadUp = nbThreadsSPSA;
    ofstream logFile("spsaOut.log");
    while(threadUp){
        for(int i=0; i<nbThreadsSPSA; i++){
            if(threads[i].check_finished()){
                if(Qiters[games[i]].add_result(threads[i].ans, i)){
                    float lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(M_PI * nbPassedIters / nbIters));
                    Qiters[games[i]].apply(lr);
                    logFile << Qiters[games[i]].diffloss() << '\n';
                    nbPassedIters++;
                    printf("\r%d/%d iters", nbPassedIters, nbIters);
                    fflush(stdout);
                    state.print(logFile);
                }
                bool islast = false;
                if(Qiters.back().nbLaunched < nbGamesPerIter){
                    string fen = Qiters.back().init_players(i, islast);
                    threads[i].launch(fen);
                    games[i] = Qiters.size()-1;
                }
                if(islast){
                    if(idSPSA >= nbIters){
                        threadUp--;
                        threads[i].bezombie();
                        continue;
                    }
                    stateIter nS;
                    nS.init(idSPSA++, &state);
                    Qiters.push_back(nS);
                }
            }
        }
    }
}
