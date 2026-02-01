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

vector<string> fens;
int idFen;
int nbGamesPerIter;
int nbIters;
int nbThreadsSPSA;
int baseTime, increment;
int memory;
int moveOverhead = 10;
float alpha = 0.602, _gamma = 0.101, lr=0.01, grain = 0.05;
class internalState{
public:
    vector<float> state;
    vector<float> firstState;
    internalState(int nbParams):state(nbParams, 1), firstState(nbParams, 1){
    }
    internalState(tunables tun){
        int nbParams = tun.to_tune_int().size()+tun.to_tune_float().size();
        state.resize(nbParams);
        firstState.resize(nbParams);
        int i = 0;
        for(int *j:tun.to_tune_int()){
            firstState[i] = state[i] = *j;
            i++;
        }
        for(float *j:tun.to_tune_float()){
            firstState[i] = state[i] = *j;
            i++;
        }
    }
    float getParam(int idx, int precision=1){
        return round(state[idx]*precision)/precision;
    }
    float getUpdate(int idx, float evolution, int precision=1){
        return max(1.0f, round((state[idx]+evolution*firstState[idx])*precision)/precision);
    }
    void updateParam(int idx, float evolution){
        state[idx] += evolution*firstState[idx];
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
    internalState* parameters;
    internalState frozenParams;
    int idSPSA;
    int nbFinished;
    int nbLaunched;
    double ak, ck;
    vector<string> pairs;
    vector<float> randoms;
    map<int, int> M;
    int score;
    stateIter():frozenParams(0){} // to have a default constructor
    void init(int id, internalState* globParams){
        randoms.clear();
        M.clear();
        pairs.clear();
        idSPSA = id;
        frozenParams = *globParams;
        score = 0;
        parameters = globParams;
        pairs.resize(nbGamesPerIter/2);
        nbLaunched = nbFinished = 0;
        std::mt19937 gen(idSPSA);
        std::uniform_int_distribution d(0, 1);
        const int n = nbIters*nbGamesPerIter;
        const double A = 0.1*n;
        const double a = pow(n+A, alpha)*lr*grain*grain;
        const double c = grain*pow(n, _gamma);
        const int k=idSPSA*nbGamesPerIter;
        ak = a / pow(k + 1 + A, alpha);
        ck = c / pow(k + 1, _gamma);
        for(int i=0; i<(int)parameters->state.size(); i++)
            randoms.push_back((d(gen)*2-1)*ck);
    }

    string init_players(int id, bool& lastGame){
        M[id] = nbLaunched;
        string fen;
        if(nbLaunched&1){
            fen = pairs[nbLaunched/2]; //recup the same fen as the other game of the pair
        }else{
            frozenParams = *parameters; // we can now update the frozen ones, this is just to assure the base parameters are the sames between the pairs games
            fen = pairs[nbLaunched/2] = fens[idFen++];
        }
        int idPlayer = nbLaunched&1; //second game of the pair we switch the players
        for(int x=0; x<2; x++){
            int sign = idPlayer?-1:1;
            vector<int*> Vs = threads[id].getPlayer(idPlayer).parameters.to_tune_int();
            for(int i = 0; i<(int)Vs.size(); i++){
                *Vs[i] = frozenParams.getUpdate(i, randoms[i]*sign);
            }
            vector<float*> Vfs = threads[id].getPlayer(idPlayer).parameters.to_tune_float();
            int offset = Vs.size();
            for(int i = 0; i<(int)Vfs.size(); i++){
                *Vfs[i] = frozenParams.getUpdate(i+offset, randoms[i+offset]*sign, 1000);
            }
            idPlayer ^= 1;
        }
        nbLaunched++;
        lastGame = nbLaunched == nbGamesPerIter;
        return fen;
    }

    bool add_result(int result, int id){
        int side = M[id]&1;
        if(side)result = 2-result; //because the first player is theta-delta
        score += result-1;
        return (++nbFinished) == nbGamesPerIter;
    }

    double diffloss(){
        return (double)score/nbGamesPerIter;
    }

    void apply(){
        double loss = diffloss();
        for(int i=0; i<(int)parameters->state.size(); i++)
            parameters->updateParam(i, loss*ak/(randoms[i]*ck));
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
        ss.ply = 0;
        while(1){
            auto start = high_resolution_clock::now();
            int player = ss.state.friendlyColor() == BLACK;
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
                }
                break;
            }
            if(ss.state.rule50_count() >= 100){
                break;
            }
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
    auto start = high_resolution_clock::now();
    ofstream logFile("spsaOut.log");
    int nbFinishedGames = 0;
    while(threadUp){
        for(int i=0; i<nbThreadsSPSA; i++){
            if(threads[i].check_finished()){
                nbFinishedGames++;
                if(Qiters[games[i]].add_result(threads[i].ans, i)){
                    Qiters[games[i]].apply();
                    logFile << Qiters[games[i]].diffloss() << '\n';
                    nbPassedIters++;
                    state.print(logFile);
                }
                bool islast = false;
                if(Qiters.back().nbLaunched < nbGamesPerIter){
                    string fen = Qiters.back().init_players(i, islast);
                    threads[i].launch(fen);
                    games[i] = Qiters.size()-1;
                }
                auto end = high_resolution_clock::now();
                int passedTime = duration_cast<milliseconds>(end-start).count();
                printf("\r%d/%d iters %d/%d games, speed: %.2fg/s time remaining: %s      ", nbPassedIters, nbIters, nbFinishedGames, nbIters*nbGamesPerIter, (nbFinishedGames*1000.0)/passedTime, secondsToStr(((long)nbIters*nbGamesPerIter-nbFinishedGames)*passedTime/(nbFinishedGames*1000)).c_str());
                fflush(stdout);
                if(islast || Qiters.back().nbLaunched >= nbGamesPerIter){
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
