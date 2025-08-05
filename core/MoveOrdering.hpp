#include "Evaluator.hpp"
#include "Move.hpp"
#include <cstring>

class HelpOrdering{
    Move killers[maxDepth][2];
    int history[2][64][64];
    int& getIndex(Move move, bool c){
        return history[c][move.start_pos][move.end_pos];
    }
public:
    void init(){
        for(int i=0; i<maxDepth; i++){
            killers[i][0] = nullMove;
            killers[i][1] = nullMove;
        }
        memset(history, 0, sizeof(history));
    }
    void addKiller(Move move, int depth, bool c){
        if(move.capture == -2){
            if(move.start_pos != killers[depth][0].start_pos || move.end_pos != killers[depth][0].end_pos){
                killers[depth][1] = killers[depth][0];
                killers[depth][0] = move;
            }
            getIndex(move, c) += depth*depth;
        }
    }
    bool isKiller(Move move, int depth) const{
        return move == killers[depth][0] || move == killers[depth][1];
    }
    int getHistoryScore(Move move, bool c) const{
        return history[c][move.start_pos][move.end_pos];
    }
};

template<int maxMoves>
class Order{
public:
    Move moves[maxMoves];
    int nbMoves;
    int scores[maxMoves];
    bool isPriority;
    int pointer;
    big dangerPositions;
    Order():dangerPositions(0){
    }
    void init(Evaluator& eval, bool c, Move priorityMove, const HelpOrdering& history, ubyte depth=-1){
        isPriority=false;
        pointer = 0;
        for(int i=0; i<nbMoves; i++){
            scores[i] = eval.score_move(moves[i], c, dangerPositions, history.isKiller(moves[i], depth), history.getHistoryScore(moves[i], c));
            if(priorityMove.start_pos == moves[i].start_pos && priorityMove.end_pos == moves[i].end_pos){
                swap(scores[i], scores[0]);
                swap(moves[i], moves[0]);
                isPriority = true;
            }
        }
    }
    Move pop_max(){
        if(isPriority && pointer == 0){
            pointer++;
            return moves[0];
        }else{
            int bPointer=pointer;
            for(int i=pointer+1; i<nbMoves; i++){
                if(scores[i] > scores[bPointer])
                    bPointer = i;
            }
            swap(moves[pointer], moves[bPointer]);
            swap(scores[pointer], scores[bPointer]);
            pointer++;
            return moves[pointer-1];
        }
    }
};