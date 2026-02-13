bool DEBUG = false;
int nbThreads=1;
#include "BestMoveFinder.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Move.hpp"
#include "tunables.hpp"
#include "Functions.hpp"
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
#include <sstream>
#include <set>
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
    void print(){
        for(float i:state){
            printf("%.4f\t", i);
        }
        printf("\n");
        fflush(stdout);
    }
    void moveToFirst(){
        firstState = state;
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
        return getPlayer(idPlayer).goState<0>(state, tm, false, ply);
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
        int phase = countbit(
            ss.state.boardRepresentation[WHITE][PAWN] |
            ss.state.boardRepresentation[BLACK][PAWN] |
            ss.state.boardRepresentation[WHITE][ROOK] |
            ss.state.boardRepresentation[BLACK][ROOK] |
            ss.state.boardRepresentation[WHITE][QUEEN] |
            ss.state.boardRepresentation[BLACK][QUEEN]
        )*2;
        phase += countbit(
            ss.state.boardRepresentation[WHITE][BISHOP] |
            ss.state.boardRepresentation[BLACK][BISHOP] |
            ss.state.boardRepresentation[WHITE][KNIGHT] |
            ss.state.boardRepresentation[BLACK][KNIGHT]
        );
        ss.ply = 0;
        while(1){
            auto start = high_resolution_clock::now();
            int player = ss.state.friendlyColor() == BLACK;
            auto res=ss.getEval(player, TM(moveOverhead, times[0], times[1], increment, increment, ss.state.friendlyColor()));
            auto end = high_resolution_clock::now();
            int used_time = duration_cast<milliseconds>(end - start).count();
            Move bm=get<0>(res);
            if(bm.moveInfo == nullMove.moveInfo){
                int score = get<2>(res);
                if(score == 0)break;
                if(score == -INF){
                    result = (ss.state.enemyColor() == WHITE)*2;
                    break;
                }
                printf("score: %d fen: %s ply: %d times: %d %d\n", get<2>(res), ss.state.toFen().c_str(), ss.ply, times[player], times[player^1]);
                assert(false);
            }
            times[player] -= used_time;
            if(times[player] < 0){
                printf("loss on time on thread %d, last time used = %d, time now remains=%d\n", id, used_time, times[player]);
                result = (ss.state.enemyColor() == WHITE)*2;
                break;
            }
            if(bm.capture != -2){
                phase -= (bm.capture != BISHOP && bm.capture != KNIGHT)+1;
            }
            times[player] += increment;
            ss.state.playMove(bm);
            ss.ply++;
            if(ss.state.threefold()){
                break;
            }
            if(phase <= 1){
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
        printf("usage: %s <book> <nbGames per Iter> <nbIter> <nbThreads> (optional:)( <memory> <baseTime> <increment>) (another optional)<logFile> <added params>(to continue)\n", argv[0]);
        printf("book: a file name that contains a list of fens\n");
        printf("nbGames Per Iter: number of games per SPSA iters\n");
        printf("nbIter: number of SPSA iters\n");
        printf("nbThreads: number of threads (each thread will run one game at the time)\n");
        printf("memory: memory per player in MB\n");
        printf("baseTime: base time on the clock in milliseconds\n");
        printf("increment: increment per move in milliseconds\n");
        printf("logFile: to continue from a stopped spsa\n");
        printf("added params: in case of restarted spsa, take in account in the reading of the file that paramaters were added in place x1 x2 etc. all the x1... should be between \" \" (like \"1 2 3\")\n");
        return 0;
    }
    string book(argv[1]);
    ifstream file(book);
    if(!file.is_open()){
        printf("book doesn't open\n");
        return 1;
    }
    idFen = 0;
    string curFen;
    while(getline(file, curFen))
        fens.push_back(curFen);
    nbGamesPerIter = atoi(argv[2]);
    nbIters = atoi(argv[3]);
    nbThreadsSPSA = atoi(argv[4]);
    memory = 16;
    baseTime = 10000, increment = 100;
    int idSPSA = 0;
    internalState state((tunables()));
    int nbPassedIters = 0;
    int nbFinishedGames = 0;
    if(argc > 5){
        memory = atoi(argv[5]);
        baseTime = atoi(argv[6]);
        increment = atoi(argv[7]);
        if(argc > 8){
            string line;
            ifstream oldLog(argv[8]);
            set<int> S;
            if(argc > 9){
                stringstream addedparams(argv[9]);
                int x;
                while(addedparams >> x)S.insert(x);
            }
            bool iseven = true;
            while(getline(oldLog, line)){
                if(!iseven){
                    stringstream ss(line);
                    int idParam = 0;
                    while(S.count(idParam))idParam++;
                    while(ss >> state.state[idParam++]){
                        while(S.count(idParam))idParam++;
                    }
                    idSPSA++;
                }
                iseven = !iseven;
            }
            state.print();
            nbPassedIters = idSPSA;
            nbFinishedGames = idSPSA*nbGamesPerIter;
            idSPSA++;
            printf("%d\n", idSPSA);
        }
    }
    int alreadyMadeGames = nbFinishedGames;
    memory *= hashMul;
    stateIter S;
    S.init(idSPSA++, &state);
    vector<stateIter> Qiters;
    Qiters.push_back(S);
    vector<int> games(nbThreadsSPSA, -1);
    threads = new HelperThread[nbThreadsSPSA];
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
                printf("\r%d/%d iters %d/%d games, speed: %.2fg/s time remaining: %s      ", nbPassedIters, nbIters, nbFinishedGames, nbIters*nbGamesPerIter, ((nbFinishedGames-alreadyMadeGames)*1000.0)/passedTime, secondsToStr(((long)nbIters*nbGamesPerIter-nbFinishedGames+alreadyMadeGames)*passedTime/(nbFinishedGames*1000)).c_str());
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
