#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include <climits>

class __attribute__((packed)) infoScore{
public:
    int score;
    ubyte typeNode;
    int16_t bestMoveInfo;
    ubyte depth;
    big hash;
};
const int INVALID = INT_MAX;
class transpositionTable{
public:
    vector<infoScore> byDepth;
    vector<infoScore> always;
    int modulo;
    int rewrite=0;
    int place=0;
    transpositionTable(size_t count){
        count /= sizeof(infoScore);
        byDepth = vector<infoScore>(count);
        always  = vector<infoScore>(count);
        modulo=count;
    }

    inline int storedScore(int alpha, int beta, int depth, const infoScore& entry){
        if(entry.depth >= depth){//if we have evaluated it with more depth remaining, we can just return this evaluation since it's a better evaluation
            if(entry.typeNode == EXACT)
                return entry.score;
            if(entry.score >= beta && entry.typeNode == LOWERBOUND)
                return entry.score;
            if(entry.score < alpha && entry.typeNode == UPPERBOUND)
                return entry.score;
        }
        return INVALID;
    }

    int get_eval(const GameState& state, int alpha, int beta, ubyte depth, int16_t& best){
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
    void push(GameState& state, int score, ubyte typeNode, Move move, ubyte depth){
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
    void clear(){
        byDepth = vector<infoScore>(modulo);
        always = vector<infoScore>(modulo);
    }
    void reinit(int count){
        count /= sizeof(infoScore);
        byDepth.resize(count);
        always.resize(count);
        modulo = count;
        place = 0;
        rewrite = 0;
    }
};

class __attribute__((packed)) infoQ{
public:
    int score;
    ubyte typeNode;
    big hash;
};
class QuiescenceTT{
public:
    vector<infoQ> table;
    int modulo;
    QuiescenceTT(size_t count){
        count /= sizeof(infoQ);
        table = vector<infoQ>(count);
        modulo=count;
    }
    int get_eval(const GameState& state, int alpha, int beta){
        int index=state.zobristHash%modulo;
        if(table[index].hash == state.zobristHash){
            if(table[index].typeNode == EXACT)
                return table[index].score;
            if(table[index].score >= beta  && table[index].typeNode == LOWERBOUND)
                return table[index].score;
            if(table[index].score <= alpha && table[index].typeNode == UPPERBOUND)
                return table[index].score;
        }
        return INVALID;
    }
    void push(GameState& state, int score, int typeNode){
        infoQ info;
        info.score = score;
        info.hash = state.zobristHash;
        info.typeNode = typeNode;
        int index = info.hash%modulo;
        table[index] = info;
    }
    void clear(){
        table = vector<infoQ>(modulo);
    }
    void reinit(int count){
        count /= sizeof(infoQ);
        table.resize(count);
        modulo = count;
    }
};

class perftMem{
public:
    big hash;
    big leefs;
    ubyte depth;
};
class TTperft{
public:
    vector<perftMem> mem;
    int modulo;
    TTperft(int alloted_mem):mem(alloted_mem/sizeof(perftMem)), modulo(alloted_mem/sizeof(perftMem)){}
    void push(perftMem eval){
        int index = eval.hash%modulo;
        mem[index] = eval;
    }
    int get_eval(big hash, int depth){
        int index = (hash*256+depth)%modulo;
        if(mem[index].depth == depth && mem[index].hash == hash)
            return mem[index].leefs;
        return -1;
    }
    void clear(){
        mem.clear();
    }
    void reinit(int count){
        count /= sizeof(perftMem);
        mem.resize(count);
        modulo = count;
    }
    void clearMem(){
        mem = vector<perftMem>(0);
        modulo = 0;
    }
};

#endif