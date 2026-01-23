#include <cstring>
#include <climits>
#include "MoveOrdering.hpp"
#include "Evaluator.hpp"
#include "Const.hpp"
#include "tunables.hpp"
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
void HelpOrdering::init(tunables& Parameters){
    for(int i=0; i<maxDepth; i++){
        killers[i][0] = nullMove;
        killers[i][1] = nullMove;
    }
    this->parameters = Parameters;
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
            updateHistory(moves[i], c, -depth*parameters.mo_mul_malus);
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
    const int value_pieces[7] = {history.parameters.pvalue, history.parameters.nvalue, history.parameters.bvalue, history.parameters.rvalue, history.parameters.qvalue, 100000, 0};
    for(int i=0; i<nbMoves; i++){
        if(moveInfoPriority == moves[i].moveInfo){
            this->swap(i, 0);
            if(nbPriority)
                this->swap(i, 1);
            nbPriority++;
        }else{
            scores[i] = score_move(moves[i], history.getMoveScore(moves[i], c, relDepth), bb, state, value_pieces);
        }
    }
    #if defined(__AVX2__)
    __m256i vIntMin = _mm256_set1_epi32(INT_MIN);
    _mm256_storeu_si256((__m256i*)&scores[nbMoves], vIntMin);
    #endif
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
        // AVX2 accelerated max finding
#if defined(__AVX2__)
        int bPointer = pointer;
        int maxScore = scores[pointer];
        
        __m256i vMaxScore = _mm256_set1_epi32(maxScore);
        __m256i vMaxIdx = _mm256_set1_epi32(bPointer);

        for(int i = pointer + 1; i < nbMoves; i += 8) {
            __m256i vScores = _mm256_loadu_si256((__m256i*)&scores[i]);
            __m256i vIndices = _mm256_add_epi32(
                _mm256_set1_epi32(i),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
            );
            
            __m256i mask = _mm256_cmpgt_epi32(vScores, vMaxScore);
            vMaxScore = _mm256_blendv_epi8(vMaxScore, vScores, mask);
            vMaxIdx = _mm256_blendv_epi8(vMaxIdx, vIndices, mask);
        }
        
        // Horizontal reduction magic
        __m256i permScore = _mm256_permute2x128_si256(vMaxScore, vMaxScore, 1);
        __m256i permIdx = _mm256_permute2x128_si256(vMaxIdx, vMaxIdx, 1);
        __m256i mask1 = _mm256_cmpgt_epi32(permScore, vMaxScore);
        vMaxScore = _mm256_blendv_epi8(vMaxScore, permScore, mask1);
        vMaxIdx = _mm256_blendv_epi8(vMaxIdx, permIdx, mask1);
        
        permScore = _mm256_shuffle_epi32(vMaxScore, _MM_SHUFFLE(1, 0, 3, 2));
        permIdx = _mm256_shuffle_epi32(vMaxIdx, _MM_SHUFFLE(1, 0, 3, 2));
        __m256i mask2 = _mm256_cmpgt_epi32(permScore, vMaxScore);
        vMaxScore = _mm256_blendv_epi8(vMaxScore, permScore, mask2);
        vMaxIdx = _mm256_blendv_epi8(vMaxIdx, permIdx, mask2);
        
        permScore = _mm256_shuffle_epi32(vMaxScore, _MM_SHUFFLE(2, 3, 0, 1));
        permIdx = _mm256_shuffle_epi32(vMaxIdx, _MM_SHUFFLE(2, 3, 0, 1));
        __m256i mask3 = _mm256_cmpgt_epi32(permScore, vMaxScore);
        vMaxScore = _mm256_blendv_epi8(vMaxScore, permScore, mask3);
        vMaxIdx = _mm256_blendv_epi8(vMaxIdx, permIdx, mask3);
        
        maxScore = _mm256_extract_epi32(vMaxScore, 0);
        bPointer = _mm256_extract_epi32(vMaxIdx, 0);
#else
        int bPointer=pointer;
        for(int i=pointer+1; i<nbMoves; i++){
            if(compareMove(bPointer, i))
                bPointer = i;
        }
#endif
        this->swap(bPointer, pointer);
        flag = scores[pointer] >> 28;
        pointer++;
        return moves[pointer-1];
    }
}