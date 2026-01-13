#include <cstring>
#include "MoveOrdering.hpp"
#include "Evaluator.hpp"
#include "Const.hpp"
//#define COUNTER

#ifdef DEBUG_MACRO
int histSum=0;
double histSquare=0;
int nbHist=0;
#endif

int getrand(big& state){
    big z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int& HelpOrdering::getIndex(Move move, bool c){
    return history[c][move.from()][move.to()];
}
bool HelpOrdering::fastEq(Move a, Move b) const{
    return a.moveInfo == b.moveInfo;
}
void HelpOrdering::init(){
    for(int i=0; i<maxDepth; i++){
        killers[i][0] = nullMove;
        killers[i][1] = nullMove;
    }
    memset(history, 0, sizeof(history));
}
void HelpOrdering::updateHistory(Move move, bool c, int bonus){
    int& index = getIndex(move, c);
    bonus = min(max(bonus, -maxHistory), maxHistory);
    int correction = bonus - index*abs(bonus)/maxHistory;
    index += correction;
}

void HelpOrdering::negUpdate(Move moves[maxMoves], int upto, bool c, int depth){
    for(int i=0; i<upto; i++){
        if(!moves[i].isTactical())
            updateHistory(moves[i], c, -depth*3);
    }
}

void HelpOrdering::addKiller(Move move, int depth, int relDepth, bool c){
    if(move.capture == -2){
        if(!fastEq(move, killers[relDepth][0])){
            killers[relDepth][1] = killers[relDepth][0];
            killers[relDepth][0] = move;
        }
        updateHistory(move, c, depth*depth);
    }
}

bool HelpOrdering::isKiller(Move move, int relDepth) const{
    if(relDepth == (ubyte)-1)return false;
    return fastEq(move, killers[relDepth][0]) || fastEq(move, killers[relDepth][1]);
}
int HelpOrdering::getHistoryScore(Move move, bool c) const{
    return history[c][move.from()][move.to()];
}

int HelpOrdering::getMoveScore(Move move, bool c, int relDepth) const{
    int score = 0;
    if(isKiller(move, relDepth))
        score = KILLER_ADVANTAGE;
    return score+getHistoryScore(move, c);
}

Order::Order():dangerPositions(0){
}
void Order::swap(int idMove1, int idMove2){
    std::swap(moves[idMove1], moves[idMove2]);
    std::swap(scores[idMove1], scores[idMove2]);
}

void Order::init(bool c, int16_t moveInfoPriority, const HelpOrdering& history, ubyte relDepth, const GameState& state){
    nbPriority = 0;
    pointer = 0;
    SEE_BB bb(state);
    for(int i=0; i<nbMoves; i++){
        if(moveInfoPriority == moves[i].moveInfo){
            this->swap(i, 0);
            if(nbPriority)
                this->swap(i, 1);
            nbPriority++;
        }else{
            scores[i] = score_move(moves[i], history.getMoveScore(moves[i], c, relDepth), bb, state);
        }
    }
}

void Order::reinit(int16_t priorityMove){
    nbPriority = 0;
    for(int i=0; i<nbMoves; i++){
        if(moves[i].moveInfo == priorityMove){
            this->swap(i, 0);
            nbPriority = 1;
            break;
        }
    }
    pointer = 0;
}
inline bool Order::compareMove(int idMove1, int idMove2){
    return scores[idMove2] > scores[idMove1];
}

Move Order::pop_max(int& flag){
    if(pointer < nbPriority){
        pointer++;
        flag = 5;
        return moves[pointer-1];
    }else{
        int bPointer=pointer;
        for(int i=pointer+1; i<nbMoves; i++){
            if(compareMove(bPointer, i))
                bPointer = i;
        }
        this->swap(bPointer, pointer);
        flag = scores[pointer] >> 28;
        pointer++;
        return moves[pointer-1];
    }
}