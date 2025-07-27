#ifndef TRANSPOSITION_TABLE_HPP
#define TRANSPOSITION_TABLE_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "Evaluator.hpp"
#include <climits>
const int EXACT = 0;
const int LOWERBOUND = 1;
const int UPPERBOUND = 2;

class infoScore{
public:
    int score;
    int typeNode;
    Move bestMove;
    int depth;
    big hash;
};
const int INVALID = INT_MAX;
class transpositionTable{
public:
    vector<infoScore> table;
    int modulo;
    int rewrite=0;
    int place=0;
    transpositionTable(int count){
        table = vector<infoScore>(count);
        modulo=count;
    }
    int get_eval(const GameState& state, int alpha, int beta, int depth, Move& best){
        int index=state.zobristHash%modulo;
        if(table[index].hash == state.zobristHash){
            if(table[index].depth == depth){//if we have evaluated it with more depth remaining, we can just return this evaluation since it's a better evaluation
                if(table[index].typeNode == EXACT)
                    return table[index].score;
                if(table[index].score >= beta  && table[index].typeNode == LOWERBOUND)
                    return beta;
                if(table[index].score <= alpha && table[index].typeNode == UPPERBOUND)
                    return alpha;
            }
            best = table[index].bestMove; //probably a good move
        }
        return INVALID;
    }
    void push(GameState& state, int score, int alpha, int beta, Move move, int depth){
        infoScore info;
        info.score = score;
        info.hash = state.zobristHash;
        info.bestMove = move;
        info.depth = depth;
        if(score >= beta) //cutted by beta, so can be higher
            info.typeNode = LOWERBOUND;
        else if(score < alpha) // the score has surely been cuted one depth after, so can be even lower
            info.typeNode = UPPERBOUND;
        else //result unaffected by borns
            info.typeNode = EXACT;
        int index = info.hash%modulo;
        //if(table[index].hash != info.hash && table[index].depth >= info.depth)return;
        table[index] = info;
    }
    void clear(){
        table = vector<infoScore>(modulo);
    }
    void reinit(int count){
        count /= sizeof(infoScore);
        table.resize(count);
        modulo = count;
        place = 0;
        rewrite = 0;
    }
};

class infoQ{
public:
    int score, typeNode;
    big hash;
};
class QuiescenceTT{
public:
    vector<infoQ> table;
    int modulo;
    QuiescenceTT(int count){
        table = vector<infoQ>(count);
        modulo=count;
    }
    int get_eval(const GameState& state, int alpha, int beta){
        int index=state.zobristHash%modulo;
        if(table[index].hash == state.zobristHash){
            if(table[index].typeNode == EXACT)
                return table[index].score;
            if(table[index].score >= beta  && table[index].typeNode == LOWERBOUND)
                return beta;
            if(table[index].score <= alpha && table[index].typeNode == UPPERBOUND)
                return alpha;
        }
        return INVALID;
    }
    void push(GameState& state, int score, int alpha, int beta){
        infoQ info;
        info.score = score;
        info.hash = state.zobristHash;
        if(score >= beta)
            info.typeNode = LOWERBOUND;
        else if(score < alpha)
            info.typeNode = UPPERBOUND;
        else info.typeNode = EXACT;
        int index = info.hash%modulo;
        table[index] = info;
    }
    void clear(){
        table = vector<infoQ>(modulo);
    }
    void reinit(int count){
        count /= sizeof(infoScore);
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
        int index = (eval.hash*256+eval.depth)%modulo;
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
};

#endif