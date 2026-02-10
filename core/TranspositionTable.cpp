#include "TranspositionTable.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include "Move.hpp"
#include <cstring>
#include <thread>
#include <cmath>

transpositionTable::transpositionTable(size_t count){
    count /= sizeof(Cluster);
    table = (Cluster*)calloc(count, sizeof(Cluster));
    modulo=count;
    age = 0;
}

int LOG[maxDepth+1];
__attribute__((constructor(101))) void init_log_table(){
    for(int i=0; i <= maxDepth; i++){
        LOG[i] = log(i+1)*10;
    }
}

int infoScore::typeNode() const{
    return 3-(flag&0b11);
}

int infoScore::age() const{
    return flag >> 2;
}

infoScore& Cluster::probe(resHash hash, bool& ttHit){
    for(int i=0; i<clusterSize; i++){
        if(entries[i].typeNode() != 3 && entries[i].hash == hash){
            ttHit = true;
            return entries[i];
        }
    }
    ttHit = false;
    return entries[0];
}

int absEntryScore(const infoScore& entry, int curAge){
    return entry.depth-((curAge-entry.age())&maxAge);
}

int preference(const infoScore& newentry, const infoScore& oldentry, int curAge){
    return absEntryScore(newentry, curAge)-absEntryScore(oldentry, curAge);
}

void Cluster::push(infoScore& entry, int curAge){
    int bestID=0;
    int bestScore = -INT_MAX;
    for(int i=0; i<clusterSize; i++){
        if(entries[i].typeNode() == 3 || entries[i].hash == entry.hash){
            bestID = i;
            break;
        }
        int score = preference(entry, entries[i], curAge);
        //if(entries[i].hash == entry.hash && score < 0)return; //if we already have a better entry for the same hash, do not put the new one in the cluster
        if(score > bestScore){
            bestScore = score;
            bestID = i;
        }
    }
    entries[bestID] = entry;
}

pair<big, resHash> getIndex(const GameState& state, big modulo){
    __uint128_t tHash = ((__uint128_t)state.zobristHash)*modulo;
    static const int dec = 8*sizeof(resHash);
    tHash >>= 64-dec;
    return {tHash >> dec, tHash&((1ULL << dec)-1)};
}

inline int transpositionTable::storedScore(int alpha, int beta, int depth, const infoScore& entry) const{
    if(entry.depth >= depth){//if we have evaluated it with more depth remaining, we can just return this evaluation since it's a better evaluation
        if(entry.typeNode() == EXACT)
            return entry.score;
        if(entry.score >= beta && entry.typeNode() == LOWERBOUND)
            return entry.score;
        if(entry.score <= alpha && entry.typeNode() == UPPERBOUND)
            return entry.score;
    }
    return INVALID;
}

int transpositionTable::get_eval(const infoScore& entry, int alpha, int beta, ubyte depth) const{
    int score = storedScore(alpha, beta, depth, entry);
    if(score != INVALID)return score;
    return INVALID;
}

int16_t transpositionTable::getMove(const infoScore& entry) const{
    return entry.bestMoveInfo; //probably a good move
}

infoScore& transpositionTable::getEntry(const GameState& state, bool& ttHit){
    auto [index, hash] = getIndex(state, modulo);
    return table[index].probe(hash, ttHit);
}

void transpositionTable::push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth, int16_t raw_eval){
    //if(score == 0)return; //because of the repetition
    infoScore info;
    auto [index, hash] = getIndex(state, modulo);
    info.raw_eval = raw_eval;
    info.score = score;
    info.hash = hash;
    info.bestMoveInfo = move.moveInfo;
    info.depth = depth;
    info.flag = (3-typeNode)|(age << 2);
    //if(table[index].hash != info.hash && table[index].depth >= info.depth)return;
    table[index].push(info, age);
}

void transpositionTable::prefetch(const GameState& state){
    __builtin_prefetch(&table[getIndex(state, modulo).first]);
}

void transpositionTable::clearRange(big start, big end){
    memset(table+start, 0, (end-start)*sizeof(Cluster));
}

void transpositionTable::clear(){
    age = 0;
    if(nbThreads == 1){
        clearRange(0, modulo);
    }else{
        thread* threads = (thread*)calloc(nbThreads, sizeof(Cluster));
        for(int i=0; i<nbThreads; i++){
            big start = modulo*i/nbThreads;
            big end = modulo*(i+1)/nbThreads;
            threads[i] = thread(&transpositionTable::clearRange, this, start, end);
        }
        for(int i=0; i<nbThreads; i++){
            if(threads[i].joinable())
                threads[i].join();
        }
        free(threads);
    }
}
void transpositionTable::reinit(size_t count){
    count /= sizeof(Cluster);
    table = (Cluster*)realloc(table, sizeof(Cluster)*count);
    modulo = count;
    clear();
    place = 0;
    rewrite = 0;
    age=0;
}
void transpositionTable::aging(){
    age++;
    age &= maxAge;
}

TTperft::TTperft(int alloted_mem):mem(alloted_mem/sizeof(perftMem)), modulo(alloted_mem/sizeof(perftMem)){}
void TTperft::push(perftMem eval){
    int index = eval.hash%modulo;
    mem[index] = eval;
}
int TTperft::get_eval(big hash, int depth){
    int index = (hash*256+depth)%modulo;
    if(mem[index].depth == depth && mem[index].hash == hash)
        return mem[index].leefs;
    return -1;
}
void TTperft::clear(){
    mem.clear();
}
void TTperft::reinit(int count){
    count /= sizeof(perftMem);
    mem.resize(count);
    modulo = count;
}
void TTperft::clearMem(){
    mem = vector<perftMem>(0);
    modulo = 0;
}