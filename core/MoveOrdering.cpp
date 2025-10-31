#include <cstring>
#include "MoveOrdering.hpp"
#include "Const.hpp"
//#define COUNTER

int& HelpOrdering::getIndex(Move move, bool c){
    return history[c][move.from()][move.to()];
}
bool HelpOrdering::fastEq(Move a, Move b) const{
    return (a.from() == b.from()) && (a.to() == b.to()) && (a.promotion() == b.promotion());
}
void HelpOrdering::init(){
    for(int i=0; i<maxDepth; i++){
        killers[i][0] = nullMove;
        killers[i][1] = nullMove;
    }
    memset(history, 0, sizeof(history));
#ifdef COUNTER
    for(int f=0; f<64*64; f++)
        counterMove[f] = nullMove.moveInfo;
#endif
}
void HelpOrdering::addKiller(Move move, int depth, int relDepth, bool c, Move lastMove){
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
#ifdef COUNTER
        counterMove[lastMove.getMovePart()] = move.moveInfo;
#endif
    }
}

bool HelpOrdering::isKiller(Move move, int relDepth) const{
    if(relDepth == (ubyte)-1)return false;
    return fastEq(move, killers[relDepth][0]) || fastEq(move, killers[relDepth][1]);
}
int HelpOrdering::getHistoryScore(Move move, bool c) const{
    return history[c][move.from()][move.to()];
}
#ifdef COUNTER
bool HelpOrdering::isCounter(Move move, Move lastMove) const{
    return move.moveInfo == counterMove[lastMove.getMovePart()];
}
#endif

int HelpOrdering::getMoveScore(Move move, bool c, int relDepth, Move lastMove) const{
    int score = 0;
    if(isKiller(move, relDepth))
        score = KILLER_ADVANTAGE;
#ifdef COUNTER
    if(isCounter(move, lastMove))
        score += KILLER_ADVANTAGE/2;
#endif
    return score+getHistoryScore(move, c);
}

template<int maxMoves>
Order<maxMoves>::Order():dangerPositions(0){
}
template<int maxMoves>
void Order<maxMoves>::swap(int idMove1, int idMove2){
    std::swap(moves[idMove1], moves[idMove2]);
    std::swap(scores[idMove1], scores[idMove2]);
    std::swap(flags[idMove1], flags[idMove2]);
}
template<int maxMoves>
void Order<maxMoves>::init(bool c, int16_t moveInfoPriority, int16_t PVMove, const HelpOrdering& history, ubyte relDepth, GameState& state, LegalMoveGenerator& generator, bool useSEE){
    nbPriority = 0;
    pointer = 0;
    for(int i=0; i<nbMoves; i++){
        if(moveInfoPriority == moves[i].moveInfo){
            this->swap(i, 0);
            if(nbPriority)
                this->swap(i, 1);
            nbPriority++;
        }else if(PVMove == moves[i].moveInfo){
            if(nbPriority)
                this->swap(i, 1);
            else 
                this->swap(i, 0);
            nbPriority++;
        }else{
            scores[i] = score_move(moves[i], dangerPositions, history.getMoveScore(moves[i], c, relDepth, state.getLastMove()), useSEE, state, flags[i], generator);
            if(moves[i].isTactical())
                flags[i]++;
        }
    }
}
template<int maxMoves>
void Order<maxMoves>::reinit(int16_t priorityMove){
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
template<int maxMoves>
inline bool Order<maxMoves>::compareMove(int idMove1, int idMove2){
    if(flags[idMove1] != flags[idMove2])return flags[idMove2] > flags[idMove1];
    return scores[idMove2] > scores[idMove1];
}
template<int maxMoves>
Move Order<maxMoves>::pop_max(){
    if(pointer < nbPriority){
        pointer++;
        return moves[pointer-1];
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

template class Order<maxMoves>;
template class Order<maxCaptures>;