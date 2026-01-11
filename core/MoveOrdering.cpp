#include <cstring>
#include "MoveOrdering.hpp"
#include "Evaluator.hpp"
#include "Const.hpp"
//#define COUNTER
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
#ifdef COUNTER
    for(int f=0; f<64*64; f++)
        counterMove[f] = nullMove.moveInfo;
#endif
}
void HelpOrdering::updateHistory(Move move, bool c, int bonus){
    bonus = min(max(bonus, -maxHistory), maxHistory);
    getIndex(move, c) += bonus - getIndex(move, c)*abs(bonus)/maxHistory;
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
#ifdef COUNTER
bool HelpOrdering::isCounter(Move move, Move lastMove) const{
    return move.moveInfo == counterMove[lastMove.getMovePart()];
}
#endif

int HelpOrdering::getMoveScore(Move move, bool c, int relDepth) const{
    int score = 0;
    if(isKiller(move, relDepth))
        score = KILLER_ADVANTAGE;
#ifdef COUNTER
    if(isCounter(move, lastMove))
        score += KILLER_ADVANTAGE/2;
#endif
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
        // Utilise maxIdx au lieu de bPointer
        int maxIdx = pointer;
        int maxScore = scores[pointer];
        
        // SIMD-accelerated max finding
        int remaining = nbMoves - pointer - 1;
        int i = pointer + 1;
        
        if(remaining >= 8) {
            // Ajouter le max partout
            __m256i vMaxScore = _mm256_set1_epi32(maxScore);
            __m256i vMaxIdx = _mm256_set1_epi32(maxIdx);
            
            // On s'occupe de 8 scores à la fois
            for(; i + 7 < nbMoves; i += 8) {
                // Charge 8 scores
                __m256i vScores = _mm256_loadu_si256((__m256i*)&scores[i]);
                
                // On crée une liste avec des indices i -> i+7
                __m256i vIndices = _mm256_add_epi32(
                    _mm256_set1_epi32(i),
                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
                );
                
                // On compare et mask = vScores > vMaxScore
                __m256i mask = _mm256_cmpgt_epi32(vScores, vMaxScore);
                
                // On update max score et idx quand mask est vrai
                vMaxScore = _mm256_blendv_epi8(vMaxScore, vScores, mask);
                vMaxIdx = _mm256_blendv_epi8(vMaxIdx, vIndices, mask);
            }
            
            // Reduction horizontale pour le max (partie que je comprend le moins)
            int temp[8];
            int tempIdx[8];
            _mm256_storeu_si256((__m256i*)temp, vMaxScore);
            _mm256_storeu_si256((__m256i*)tempIdx, vMaxIdx);
            
            for(int j = 0; j < 8; j++) {
                if(temp[j] > maxScore) {
                    maxScore = temp[j];
                    maxIdx = tempIdx[j];
                }
            }
        }
    
    // Même logique qu'avant
    // Ce qui reste (pas besoin de redef i)
    for(; i < nbMoves; i++) {
        if(scores[i] > maxScore) {
            maxScore = scores[i];
            maxIdx = i;
        }
    }
    
    this->swap(maxIdx, pointer);
    flag = scores[pointer] >> 28;
    return moves[pointer++];
    }
}