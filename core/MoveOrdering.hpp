#include "Evaluator.hpp"
#include "Move.hpp"

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
    void init(Evaluator& eval, bool c, Move priorityMove){
        isPriority=false;
        pointer = 0;
        for(int i=0; i<nbMoves; i++){
            scores[i] = eval.score_move(moves[i], c, dangerPositions);
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