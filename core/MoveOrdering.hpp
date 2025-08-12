#include "Evaluator.hpp"
#include "Move.hpp"
#include <cstring>

class HelpOrdering{
    Move killers[maxDepth][2];
    int history[2][64][64];
    int& getIndex(Move move, bool c){
        return history[c][move.from()][move.to()];
    }
    bool fastEq(Move a, Move b) const{
        return (a.from() == b.from()) && (a.to() == b.to()) && (a.promotion() == b.promotion());
    }
public:
    void init(){
        for(int i=0; i<maxDepth; i++){
            killers[i][0] = nullMove;
            killers[i][1] = nullMove;
        }
        memset(history, 0, sizeof(history));
    }
    void addKiller(Move move, int depth, int relDepth, bool c){
        if(move.capture == -2){
            if(!fastEq(move, killers[relDepth][0])){
                killers[relDepth][1] = killers[relDepth][0];
                killers[relDepth][0] = move;
            }
            getIndex(move, c) += depth*depth;
            if(getIndex(move, c) > KILLER_ADVANTAGE-PAWN){
                for(int a=0; a<2; a++){
                    for(int from=0; from<64; from++){
                        for(int to=0; to<64; to++){
                            history[a][from][to] /= 2;
                        }
                    }
                }
            }
        }
    }

    bool isKiller(Move move, int relDepth) const{
        if(relDepth == (ubyte)-1)return false;
        return fastEq(move, killers[relDepth][0]) || fastEq(move, killers[relDepth][1]);
    }
    int getHistoryScore(Move move, bool c) const{
        return history[c][move.from()][move.to()];
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
    bool sorted = false;
    Order():dangerPositions(0){
    }

    bool isChanger(Move move) const{
        return move.capture != -2 || move.promotion() != -1;
    }

    void init(bool c, int16_t moveInfoPriority, const HelpOrdering& history, ubyte relDepth=-1){
        isPriority=false;
        pointer = 0;
        for(int i=0; i<nbMoves; i++){
            scores[i] = score_move(moves[i], c, dangerPositions, history.isKiller(moves[i], relDepth), history.getHistoryScore(moves[i], c));
            if(moveInfoPriority == moves[i].moveInfo){
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
                if(isChanger(moves[bPointer])){
                    if(isChanger(moves[i]) && scores[i] > scores[bPointer])
                        bPointer = i;
                }else if(isChanger(moves[i]) || scores[i] > scores[bPointer])
                    bPointer = i;
            }
            swap(moves[pointer], moves[bPointer]);
            swap(scores[pointer], scores[bPointer]);
            pointer++;
            return moves[pointer-1];
        }
    }
};
