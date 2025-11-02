#include "TranspositionTable.hpp"

transpositionTable::transpositionTable(size_t count){
        count /= sizeof(infoScore);
        byDepth = vector<infoScore>(count);
        always  = vector<infoScore>(count);
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
    if(byDepth[index].hash == state.zobristHash){
        int score = storedScore(alpha, beta, depth, byDepth[index]);
        if(score != INVALID)return score;
        best = byDepth[index].bestMoveInfo; //probably a good move
    }else if(always[index].hash == state.zobristHash){
        int score = storedScore(alpha, beta, depth, always[index]);
        if(score != INVALID)return score;
        best = always[index].bestMoveInfo; //probably a good move
    }
    return INVALID;
}

int16_t transpositionTable::getMove(const GameState& state){
    int index=state.zobristHash%modulo;
    if(byDepth[index].hash == state.zobristHash)
        return byDepth[index].bestMoveInfo; //probably a good move
    else if(always[index].hash == state.zobristHash)
        return always[index].bestMoveInfo; //probably a good move
    return nullMove.moveInfo;
}

void transpositionTable::push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth){
    //if(score == 0)return; //because of the repetition
    if(score <= -INF || score >= INF)return;
    infoScore info;
    info.score = score;
    info.hash = state.zobristHash;
    info.bestMoveInfo = move.moveInfo;
    info.depth = depth;
    info.typeNode = typeNode;
    int index = info.hash%modulo;
    //if(table[index].hash != info.hash && table[index].depth >= info.depth)return;
    if(info.depth >= byDepth[index].depth)
        byDepth[index] = info;
    else
        always[index] = info;
}
void transpositionTable::clear(){
    byDepth = vector<infoScore>(modulo);
    always = vector<infoScore>(modulo);
}
void transpositionTable::reinit(int count){
    count /= sizeof(infoScore);
    count /= 2;
    byDepth.resize(count);
    always.resize(count);
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