#include "TranspositionTable.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include "Move.hpp"
#include <cstring>
#include <thread>

transpositionTable::transpositionTable(size_t count){
    count /= sizeof(infoScore);
    table = (infoScore*)calloc(count, sizeof(infoScore));
    modulo=count;
}

pair<big, uint32_t> getIndex(const GameState& state, big modulo){
    __uint128_t tHash = ((__uint128_t)state.zobristHash)*modulo;
    return {tHash >> 64, tHash&~(uint32_t)0};
}

inline int transpositionTable::storedScore(int alpha, int beta, int depth, const infoScore& entry) const{
    if(entry.depth >= depth){//if we have evaluated it with more depth remaining, we can just return this evaluation since it's a better evaluation
        if(entry.typeNode == EXACT)
            return entry.score;
        if(entry.score >= beta && entry.typeNode == LOWERBOUND)
            return entry.score;
        if(entry.score <= alpha && entry.typeNode == UPPERBOUND)
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
    ttHit = false;
    auto [index, resHash] = getIndex(state, modulo);
    if(table[index].hash == resHash)
        ttHit = true;
    return table[index];

}

void transpositionTable::push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth, int16_t raw_eval){
    //if(score == 0)return; //because of the repetition
    infoScore info;
    auto [index, resHash] = getIndex(state, modulo);
    info.raw_eval = raw_eval;
    info.score = score;
    info.hash = resHash;
    info.bestMoveInfo = move.moveInfo;
    info.depth = depth;
    info.typeNode = typeNode;
    //if(table[index].hash != info.hash && table[index].depth >= info.depth)return;
    if(info.depth >= table[index].depth || info.typeNode < table[index].typeNode)
        table[index] = info;
}

void transpositionTable::prefetch(const GameState& state){
    __builtin_prefetch(&table[getIndex(state, modulo).first]);
}

void transpositionTable::clearRange(big start, big end){
    memset(table+start, 0, (end-start)*sizeof(infoScore));
}

void transpositionTable::clear(){
    if(nbThreads == 1){
        clearRange(0, modulo);
    }else{
        thread* threads = (thread*)calloc(nbThreads, sizeof(infoScore));
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
    count /= sizeof(infoScore);
    table = (infoScore*)realloc(table, sizeof(infoScore)*count);
    modulo = count;
    clear();
    place = 0;
    rewrite = 0;
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