#include <cstring>
#include <climits>
#include "MoveOrdering.hpp"
#include "Evaluator.hpp"
#include "Const.hpp"
#include "GameState.hpp"
#include "tunables.hpp"

#ifdef DEBUG_MACRO
StatVar<sbig, maxHistory*2, -maxHistory*2> quiethistPreStat;
StatVar<sbig, maxHistory, -maxHistory> capthistPreStat;
#endif

//#define COUNTER
int getrand(big& state){
    big z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int& HelpOrdering::getTactIndex(Move move, bool c){
    if(move.promotion() == -1)
        return captHist[c][move.piece][max<int8_t>(move.capture, 0)][move.to()];
    else
        return captHist[c][move.promotion()-KNIGHT+nbPieces][max<int>(move.capture, 0)][move.to()];

}
bool HelpOrdering::fastEq(Move a, Move b) const{
    return a.moveInfo == b.moveInfo;
}
void HelpOrdering::init(const tunables& Parameters){
    for(int i=0; i<maxDepth; i++){
        killers[i][0] = nullMove;
        killers[i][1] = nullMove;
    }
    this->parameters = Parameters;
    memset(history, 0, sizeof(history));
    memset(conthist, 0, sizeof(conthist));
    memset(captHist, 0, sizeof(captHist));
}
void HelpOrdering::updateHistory(int bonus, int& hist){
    bonus = min(max(bonus, -maxHistory), maxHistory);
    hist += bonus - hist*abs(bonus)/maxHistory;
}

void HelpOrdering::bonusMove(int depth, Move move, bool c, const GameState& state){
    if(move.isTactical()){
        updateHistory(depth*parameters.capthist_mul_bonus, getTactIndex(move, c));
    }else{
        updateHistory(depth*parameters.mainHist.mul_bonus, history[c][move.from()][move.to()]);
        Move lastmove = state.getLastMove();
        Move contmove = state.getContMove();
        updateHistory(depth*parameters.prevHist.mul_bonus, conthist[0][c][lastmove.piece][lastmove.to()][move.piece][move.to()]);
        updateHistory(depth*parameters.contHist.mul_bonus, conthist[1][c][contmove.piece][contmove.to()][move.piece][move.to()]);
    }
}

void HelpOrdering::malusMove(int depth, Move move, bool c, const GameState& state){
    if(move.isTactical()){
        updateHistory(-depth*parameters.capthist_mul_malus, getTactIndex(move, c));
    }else{
        updateHistory(-depth*parameters.mainHist.mul_malus, history[c][move.from()][move.to()]);
        Move lastmove = state.getLastMove();
        Move contmove = state.getContMove();
        updateHistory(-depth*parameters.prevHist.mul_malus, conthist[0][c][lastmove.piece][lastmove.to()][move.piece][move.to()]);
        updateHistory(-depth*parameters.contHist.mul_malus, conthist[1][c][contmove.piece][contmove.to()][move.piece][move.to()]);
    }
}

void HelpOrdering::negUpdate(Move moves[maxMoves], int upto, bool c, int depth, const GameState& state){
    for(int i=0; i<upto; i++){
        if(moves[i].isTactical() >= moves[upto].isTactical())
            malusMove(depth, moves[i], c, state);
    }
}

void HelpOrdering::addKiller(Move move, int depth, int relDepth, bool c, const GameState& state){
    if(move.capture == -2){
        if(!fastEq(move, killers[relDepth][0])){
            killers[relDepth][1] = killers[relDepth][0];
            killers[relDepth][0] = move;
        }
    }
    bonusMove(depth, move, c, state);
}

bool HelpOrdering::isKiller(Move move, int relDepth) const{
    if(relDepth == (ubyte)-1)return false;
    return fastEq(move, killers[relDepth][0]) || fastEq(move, killers[relDepth][1]);
}

template<int id>
int HelpOrdering::getQuietScore(Move move, bool c, const GameState& state) const{
    Move lastmove = state.getLastMove();
    Move contmove = state.getContMove();
    int hist = 0;
    hist += history[c][move.from()][move.to()]*parameters.mainHist.getByID<id>();
    hist += conthist[0][c][lastmove.piece][lastmove.to()][move.piece][move.to()]*parameters.prevHist.getByID<id>();
    hist += conthist[0][c][contmove.piece][contmove.to()][move.piece][move.to()]*parameters.contHist.getByID<id>();
    return hist/1024;
}
template int HelpOrdering::getQuietScore<0>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<1>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<2>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<3>(Move, bool, const GameState&) const;
template<int id>
int HelpOrdering::getHistoryScore(Move move, bool c, const GameState& state) const{
    if(!move.isTactical()){
        return getQuietScore<id>(move, c, state);
    }else if(move.promotion() == -1)
        return captHist[c][move.piece][max<int8_t>(move.capture, 0)][move.to()];
    else
        return captHist[c][move.promotion()-KNIGHT+nbPieces][max<int>(move.capture, 0)][move.to()];
}
template int HelpOrdering::getHistoryScore<0>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<1>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<2>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<3>(Move, bool, const GameState&) const;

int HelpOrdering::getMoveScore(Move move, bool c, int relDepth, const GameState& state) const{
    int score = 0;
    if(move.capture == -2 && isKiller(move, relDepth))
        score = KILLER_ADVANTAGE;
    return score+getHistoryScore<TunableHistory::ORDER>(move, c, state);
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
    const int value_pieces[7] = {history.parameters.pvalue, history.parameters.nvalue, history.parameters.bvalue, history.parameters.rvalue, history.parameters.qvalue, 100000, 0};
    for(int i=0; i<nbMoves; i++){
        if(moveInfoPriority == moves[i].moveInfo){
            this->swap(i, 0);
            if(nbPriority)
                this->swap(i, 1);
            nbPriority++;
        }else{
#ifdef DEBUG_MACRO
            int moveHistory = history.getHistoryScore<TunableHistory::ORDER>(moves[i], state.friendlyColor(), state);
            if(moveHistory != maxHistory){
                if(moves[i].isTactical()){
                    capthistPreStat.update(moveHistory);
                }else{
                    quiethistPreStat.update(moveHistory);
                }
            }
#endif
            scores[i] = score_move(moves[i], history.getMoveScore(moves[i], c, relDepth, state), state, value_pieces);
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