#include "TranspositionTable.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include <cstring>

transpositionTable::transpositionTable(size_t count){
    count /= sizeof(infoScore);
    table = (infoScore*)calloc(count, sizeof(infoScore));
    modulo=count;
}

inline int transpositionTable::storedScore(int alpha, int beta, int depth, const infoScore& entry){
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

int transpositionTable::get_eval(const GameState& state, int alpha, int beta, ubyte depth, int16_t& best){
    int index=state.zobristHash%modulo;
    uint32_t resHash = state.zobristHash/modulo;
    if(table[index].hash == resHash){
        int score = storedScore(alpha, beta, depth, table[index]);
        if(score != INVALID)return score;
        best = table[index].bestMoveInfo; //probably a good move
    }
    return INVALID;
}

int16_t transpositionTable::getMove(const GameState& state){
    int index=state.zobristHash%modulo;
    uint32_t resHash = state.zobristHash/modulo;
    if(table[index].hash == resHash)
        return table[index].bestMoveInfo; //probably a good move
    return nullMove.moveInfo;
}

infoScore transpositionTable::getEntry(const GameState& state, bool& ttHit){
    ttHit = false;
    int index=state.zobristHash%modulo;
    uint32_t resHash = state.zobristHash/modulo;
    if(table[index].hash == resHash)
        ttHit = true;
    return table[index];

}

void transpositionTable::push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth){
    //if(score == 0)return; //because of the repetition
    infoScore info;
    info.score = score;
    info.hash = state.zobristHash/modulo;
    info.bestMoveInfo = move.moveInfo;
    info.depth = depth;
    info.typeNode = typeNode;
    int index = state.zobristHash%modulo;
    //if(table[index].hash != info.hash && table[index].depth >= info.depth)return;
    if(info.depth >= table[index].depth || (info.typeNode != table[index].typeNode && (info.typeNode == EXACT || table[index].typeNode == UPPERBOUND)))
        table[index] = info;
}
void transpositionTable::clear(){
    memset(table, 0, modulo*sizeof(infoScore));
}
void transpositionTable::reinit(int count){
    count /= sizeof(infoScore);
    table = (infoScore*)realloc(table, sizeof(infoScore)*count);
    modulo = count;
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