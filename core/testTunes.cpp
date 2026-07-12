#include <algorithm>
#include <fstream>
bool DEBUG = false;
bool isdfrc=true;
#define TUNE
int nbThreads=1;
#include "BestMoveFinder.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include "tunables.hpp"
#include <mutex>
#include <vector>
#include <cmath>
#include <vector>
#include <cassert>
#include <sstream>
#include "viriformatUtil.hpp"
#include <array>
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
int nbGames;
int nbThreadsFG;
int baseTime;
int memory;
const int moveOverhead = 10;
const float alpha = 0.602, _gamma = 0.101, a_ratio=0.1;
class internalState{
public:
    vector<TunableFloat> state;
    vector<TunableFloat> firstState;
    int nbInts;
    internalState(int nbParams, int _nbInts):state(nbParams, 1), firstState(nbParams, 1), nbInts(_nbInts){
    }
    internalState(tunables tun){
        int nbParams = tun.to_tune_int().size()+tun.to_tune_float().size();
        nbInts = tun.to_tune_int().size();
        state.resize(nbParams, 1);
        firstState.resize(nbParams, 1);
        int i = 0;
        for(TunableInt *j:tun.to_tune_int()){
            TunableFloat x(j->value, j->minimum, j->maximum, j->c_end, j->r_end);
            firstState[i] = state[i] = x;
            i++;
        }
        for(TunableFloat* j:tun.to_tune_float()){
            firstState[i] = state[i] = *j;
            i++;
        }
    }
    float getParam(int idx, int precision=1){
        return round(state[idx]*precision)/precision;
    }
    float getUpdate(int idx, float evolution, int precision=1){
        return clamp(round((state[idx]+evolution)*precision)/precision, state[idx].minimum, state[idx].maximum);
    }
    void updateParam(int idx, float evolution){
        state[idx].value += evolution;
        state[idx].value = clamp(state[idx].value, state[idx].minimum, state[idx].maximum);
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
    HelperThread():player0(memory), player1(memory){}
    int ans;
    bool zombie;
    void init(){
        ans = -1;
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
        return getPlayer(idPlayer).goState<1>(state, tm, false, ply);
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
        return !running && !zombie;
    }
    void bezombie(){
        zombie = true;
    }
};

HelperThread *threads;
void play_pairs(int id){
    HelperThread& ss=threads[id];
    Move legalMoves[maxMoves];
    LegalMoveGenerator generator;
    bool inCheck;
    big dngpos;
    while(1){
        ss.wait_notification();
        if(ss.fen == "-")break;
        int overallresult = 0;
        for(int side=0; side<2; side++){
            int result = 1;
            ss.state.fromFen(ss.fen);
            int phase = countbit(
                ss.state.board.pieces[PAWN] |
                ss.state.board.pieces[ROOK] |
                ss.state.board.pieces[QUEEN]
            )*2;
            phase += countbit(
                ss.state.board.pieces[BISHOP] |
                ss.state.board.pieces[KNIGHT]
            );
            ss.ply = 0;
            while(1){
                int player = (ss.state.friendlyColor() == WHITE)^side;
                auto res=ss.getEval(player, TM(baseTime, baseTime));
                Move bm=get<0>(res);
                if(bm.moveInfo == nullMove.moveInfo){
                    int score = get<2>(res);
                    if(score == 0)break;
                    if(score == -INF){
                        result = (ss.state.enemyColor() == WHITE)*2;
                        break;
                    }
                    printf("score: %d fen: %s ply: %d\n", get<2>(res), ss.state.toFen().c_str(), ss.ply);
                    assert(false);
                }
                int capture = ss.state.board.getCapture(bm);
                if(capture != SPACE){
                    phase -= (capture != BISHOP && capture != KNIGHT)+1;
                }
                if(bm.promotion() == KNIGHT || bm.promotion() == BISHOP)
                    phase -= 1;
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
            if(side)
                result = 2-result;
            overallresult += result;
        }
        ss.set_result(overallresult);
    }
}

string getNextFen(){
    return fens[(idFen++)%fens.size()];
}

int main(int argc, char** argv){
    if(argc == 1){
        printf("usage: %s <book> <nbGames> <nbThreads> <memory> <baseTime> <logFile>\n", argv[0]);
        printf("book: a file name that contains a list of fens\n");
        printf("nbGames: number of games in the tune\n");
        printf("nbThreads: number of threads (each thread will run one game at the time)\n");
        printf("memory: memory per player in MB\n");
        printf("baseTime: base time in nodes\n");
        printf("logFile: to continue from a stopped spsa\n");
        printf("every: every how much iters you want a fixed game test of nbGames\n");
        return 0;
    }
    assert(argc > 7);
    string book(argv[1]);
    nbGames = atoi(argv[2]);
    nbThreadsFG = atoi(argv[3]);
    memory = atoi(argv[4]);
    baseTime = atoi(argv[5]);
    string logname = argv[6];
    int every = atoi(argv[7]);
    assert(nbGames >= nbThreadsFG);
    ifstream file(book);
    if(!file.is_open()){
        printf("book doesn't open\n");
        return 1;
    }
    idFen = 0;
    string curFen;
    while(getline(file, curFen))
        fens.push_back(curFen);
    string line;
    ifstream logs(logname);
    bool iseven = true;
    int idSPSA = 0;
    internalState state((tunables()));
    memory *= hashMul;
    threads = new HelperThread[nbThreadsFG];
    for(int i=0; i<nbThreadsFG; i++)
        threads[i].t = thread(play_pairs, i);
    printf("start reading\n");
    while(getline(logs, line)){
        if(!iseven){
            stringstream ss(line);
            int idParam = 0;
            while(ss >> state.state[idParam++].value);
            idSPSA++;
            if(idSPSA%every == 0){
                array<int, 5> pairs{};
                int nbFinishedGame = 0;
                int nbStartedGame = 0;
                for(int i=0; i<nbThreadsFG; i++){
                    threads[i].init();
                    int curParam = 0;
                    for(auto x:threads[i].player0.parameters.to_tune_int()){
                        x->value = state.getParam(curParam++);
                    }
                    for(auto x:threads[i].player0.parameters.to_tune_float()){
                        x->value = state.getParam(curParam++, 1000);
                    }
                }
                while(nbFinishedGame < nbGames){
                    for(int i=0; i<nbThreadsFG; i++){
                        if(threads[i].check_finished()){
                            if(threads[i].ans != -1){
                                pairs[threads[i].ans]++;
                                nbFinishedGame++;
                                printf("\r%d/%d", nbFinishedGame, nbGames);
                                fflush(stdout);
                            }
                            if(nbStartedGame < nbGames){
                                nbStartedGame++;
                                threads[i].launch(getNextFen());
                            }else{
                                threads[i].bezombie();
                            }
                        }
                    }
                }
                printf("\r%d : [%d, %d, %d, %d, %d]\n", idSPSA, pairs[0], pairs[1], pairs[2], pairs[3], pairs[4]);
            }
        }
        iseven = !iseven;
    }
}