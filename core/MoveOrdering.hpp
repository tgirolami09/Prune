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
            if(getIndex(move, c) > maxHistory){
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
    ubyte flags[maxMoves]; // winning captures, non-losing quiet move, losing captures, losing quiet moves
    big dangerPositions;
    bool sorted = false;
    Order():dangerPositions(0){
    }

    void swap(int idMove1, int idMove2){
        std::swap(moves[idMove1], moves[idMove2]);
        std::swap(scores[idMove1], scores[idMove2]);
        std::swap(flags[idMove1], flags[idMove2]);
    }

    void init(bool c, int16_t moveInfoPriority, const HelpOrdering& history, ubyte relDepth, GameState& state, bool useSEE=true){
        isPriority=false;
        pointer = 0;
        for(int i=0; i<nbMoves; i++){
            int SEEscore = 0;
            if(useSEE){
                state.playMove<false, false>(moves[i]);
                SEEscore = -SEE(moves[i].to(), state);
                if(moves[i].capture != -2)
                    SEEscore += value_pieces[moves[i].capture == -1?0:moves[i].capture];
                state.undoLastMove<false>();
            }else if(moves[i].isTactical()){
                int cap = moves[i].capture;
                if(cap == -1)cap = 0;
                if(cap != -2)
                    SEEscore = value_pieces[cap]*10;
                if((1ULL << moves[i].to())&dangerPositions)
                    SEEscore -= value_pieces[moves[i].piece];
            }
            scores[i] = score_move(moves[i], c, dangerPositions, history.isKiller(moves[i], relDepth), history.getHistoryScore(moves[i], c), SEEscore);
            if(SEEscore > 0)
                flags[i] = 2;
            else flags[i] = 0;
            if(moves[i].isTactical())
                flags[i]++;
            if(moveInfoPriority == moves[i].moveInfo){
                this->swap(i, 0);
                isPriority = true;
            }
        }
    }

    inline bool compareMove(int idMove1, int idMove2){
        if(flags[idMove1] != flags[idMove2])return flags[idMove2] > flags[idMove1];
        return scores[idMove2] > scores[idMove1];
    }

    Move pop_max(){
        if(isPriority && pointer == 0){
            pointer = 1;
            return moves[0];
        }else{
            int bPointer=pointer;
            for(int i=pointer+1; i<nbMoves; i++){
                if(compareMove(bPointer, i))
                    bPointer = i;
            }
            this->swap(bPointer, pointer);
            pointer++;
            return moves[pointer-1];
        }
    }
};

class RootOrder{
    int nodeUsed[maxMoves];
    int scores[maxMoves];
    int pointer;
    bool isPriority;
    bool compareScore(int idMove1, int idMove2) const{
        if(moves[idMove1].isTactical() != moves[idMove2].isTactical())return moves[idMove2].isTactical() > moves[idMove1].isTactical();
        return scores[idMove2] > scores[idMove1];
    }

    bool compareMoves(int idMove1, int idMove2) const{
        if(nodeUsed[idMove1] != nodeUsed[idMove2])return nodeUsed[idMove2] > nodeUsed[idMove1];
        return compareScore(idMove1, idMove2);
    }
public:
    int nbMoves;
    big dangerPositions;
    Move moves[maxMoves];
    RootOrder():dangerPositions(0){}

    void init(bool c, const HelpOrdering& history, GameState& state){
        isPriority = false;
        pointer = 0;
        for(int i=0; i<nbMoves; i++){
            int SEEscore = 0;
            state.playMove<false, false>(moves[i]);
            SEEscore = SEE(moves[i].to(), state);
            state.undoLastMove<false>();
            scores[i] = score_move(moves[i], c, dangerPositions, false, history.getHistoryScore(moves[i], c), SEEscore);
            nodeUsed[i] = 0;
        }
    }

    void reinit(int16_t priorityMove){
        isPriority = false;
        for(int i=0; i<nbMoves; i++){
            if(moves[i].moveInfo == priorityMove){
                this->swap(i, 0);
                isPriority = true;
                break;
            }
        }
        pointer = 0;
    }

    void pushNodeUsed(int usedNodes){
        nodeUsed[pointer-1] = usedNodes;
    }

    void cutoff(){
        for(int i=pointer; i<nbMoves; i++){
            nodeUsed[pointer] = 0;
        }
    }

    void swap(int idMove1, int idMove2){
        if(idMove1 != idMove2){
            std::swap(moves[idMove1], moves[idMove2]);
            std::swap(scores[idMove1], scores[idMove2]);
            std::swap(nodeUsed[idMove1], nodeUsed[idMove2]);
        }
    }

    Move pop_max(){
        if(isPriority){
            isPriority = false;
            pointer++;
            return moves[0];
        }else{
            int bestPointer = pointer;
            for(int i=pointer+1; i<nbMoves; i++){
                if(compareMoves(bestPointer, i))
                    bestPointer = i;
            }
            this->swap(bestPointer, pointer);
            return moves[pointer++];
        }
    }
};