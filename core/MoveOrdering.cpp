#include <cstring>
#include <climits>
#include "MoveOrdering.hpp"
#include "Evaluator.hpp"
#include "Const.hpp"
#include "Move.hpp"
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

int& HelpOrdering::getTactIndex(const GameState& state, Move move, bool c){
    int piece = state.getPiece(move.from());
    int capture = state.board.getCapture(move);
    if(move.getFlag() != Move::fpromo)
        return captHist[c][piece][capture][move.to()];
    else
        return captHist[c][move.promotion()-KNIGHT+nbPieces][capture-1][move.to()];

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
    bonus = min(max(bonus/fracDepth, -maxHistory), maxHistory);
    hist += bonus - hist*abs(bonus)/maxHistory;
}

void HelpOrdering::bonusMove(int depth, Move move, bool c, const GameState& state){
    if(state.board.isTactical(move)){
        updateHistory(depth*parameters.capthist_mul_bonus, getTactIndex(state, move, c));
    }else{
        updateHistory(depth*parameters.mainHist.bonus, history[c][move.from()][move.to()]);
        ExpendedMove lastmove = state.getLastMove();
        ExpendedMove contmove = state.getContMove();
        updateHistory(depth*parameters.prevHist.bonus, conthist[0][c][lastmove.piece][lastmove.move.to()][state.getPiece(move.from())][move.to()]);
        updateHistory(depth*parameters.contHist.bonus, conthist[1][c][contmove.piece][contmove.move.to()][state.getPiece(move.from())][move.to()]);
    }
}

void HelpOrdering::malusMove(int depth, Move move, bool c, const GameState& state){
    if(state.board.isTactical(move)){
        updateHistory(-depth*parameters.capthist_mul_malus, getTactIndex(state, move, c));
    }else{
        updateHistory(-depth*parameters.mainHist.malus, history[c][move.from()][move.to()]);
        ExpendedMove lastmove = state.getLastMove();
        ExpendedMove contmove = state.getContMove();
        updateHistory(-depth*parameters.prevHist.malus, conthist[0][c][lastmove.piece][lastmove.move.to()][state.getPiece(move.from())][move.to()]);
        updateHistory(-depth*parameters.contHist.malus, conthist[1][c][contmove.piece][contmove.move.to()][state.getPiece(move.from())][move.to()]);
    }
}

void HelpOrdering::negUpdate(Move moves[maxMoves], int upto, bool c, int depth, const GameState& state){
    for(int i=0; i<upto; i++){
        if(state.board.isTactical(moves[i]) >= state.board.isTactical(moves[upto]))
            malusMove(depth, moves[i], c, state);
    }
}

void HelpOrdering::addKiller(Move move, int depth, int relDepth, bool c, const GameState& state){
    if(state.getPiece(move.to()) == SPACE || move.getFlag() == Move::fcastle){
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


int HelpOrdering::getCaptScore(Move move, bool c, const GameState& state) const{
    int capture = state.board.getCapture(move);
    int piece = state.getPiece(move.from());
    if(move.getFlag() != Move::fpromo)
        return captHist[c][piece][capture][move.to()];
    else
        return captHist[c][move.promotion()-KNIGHT+nbPieces][capture-1][move.to()];
}

template<int id>
int HelpOrdering::getQuietScore(Move move, bool c, const GameState& state) const{
    int score = 0;
    ExpendedMove lastmove = state.getLastMove();
    ExpendedMove contmove = state.getContMove();
    score += history[c][move.from()][move.to()]*parameters.mainHist.getParam<id>();
    score += conthist[0][c][lastmove.piece][lastmove.move.to()][state.getPiece(move.from())][move.to()]*parameters.prevHist.getParam<id>();
    score += conthist[1][c][contmove.piece][contmove.move.to()][state.getPiece(move.from())][move.to()]*parameters.contHist.getParam<id>();
    if constexpr(id == TunableHist::ORDER){
        return score*2/(3*1024);
    }else {
        return score/1024;
    }
}

template<int id>
int HelpOrdering::getHistoryScore(Move move, bool c, const GameState& state) const{
    if(!state.board.isTactical(move)){
        return getQuietScore<id>(move, c, state);
    }else{
        return getCaptScore(move, c, state);
    }
}

template int HelpOrdering::getQuietScore<TunableHist::ORDER>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<TunableHist::LMR>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<TunableHist::MHP>(Move, bool, const GameState&) const;
template int HelpOrdering::getQuietScore<TunableHist::FP>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<TunableHist::ORDER>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<TunableHist::LMR>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<TunableHist::MHP>(Move, bool, const GameState&) const;
template int HelpOrdering::getHistoryScore<TunableHist::FP>(Move, bool, const GameState&) const;

int HelpOrdering::getMoveScore(Move move, bool c, int relDepth, const GameState& state) const{
    int score = 0;
    if(state.board.getCapture(move) == SPACE && isKiller(move, relDepth))
        score = KILLER_ADVANTAGE;
    return score+getHistoryScore<TunableHist::ORDER>(move, c, state);
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
            int moveHistory = history.getHistoryScore<TunableHist::ORDER>(moves[i], state.friendlyColor(), state);
            if(moveHistory != maxHistory){
                if(state.board.isTactical(moves[i])){
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
    #elif defined(__ARM_NEON__)
    vst1q_s32(&scores[nbMoves], vdupq_n_s32(INT_MIN));
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

#if defined(__AVX2__)
static inline __m256i hmax_epi32(__m256i v){
    v = _mm256_max_epi32(v, _mm256_permute2x128_si256(v, v, 1));
    v = _mm256_max_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm256_max_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return v;
}
static inline __m256i hmin_epi32(__m256i v){
    v = _mm256_min_epi32(v, _mm256_permute2x128_si256(v, v, 1));
    v = _mm256_min_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm256_min_epi32(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return v;
}
#endif

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
        
        // The horizontal reduction  should also minimise the index for the max
        maxScore = _mm256_extract_epi32(hmax_epi32(vMaxScore), 0);
        __m256i eqMax = _mm256_cmpeq_epi32(vMaxScore, _mm256_set1_epi32(maxScore));
        __m256i cand  = _mm256_blendv_epi8(_mm256_set1_epi32(INT_MAX), vMaxIdx, eqMax);
        bPointer = _mm256_extract_epi32(hmin_epi32(cand), 0);        
#elif defined(__ARM_NEON__)
        // NEON accelerated max finding
        int bPointer = pointer;
        int32x4_t vMaxScore = vdupq_n_s32(scores[pointer]);
        int32x4_t vMaxIdx   = vdupq_n_s32(bPointer);
        const int32x4_t iota = {0, 1, 2, 3};

        for(int i = pointer + 1; i < nbMoves; i += 4){
            int32x4_t vScores  = vld1q_s32(&scores[i]);
            int32x4_t vIndices = vaddq_s32(vdupq_n_s32(i), iota);
            uint32x4_t mask = vcgtq_s32(vScores, vMaxScore);
            vMaxScore = vbslq_s32(mask, vScores, vMaxScore);
            vMaxIdx   = vbslq_s32(mask, vIndices, vMaxIdx);
        }

        // Horizontal reduction (i think this also get the lowest index for the max)
        const int maxv = vmaxvq_s32(vMaxScore);
        uint32x4_t eq = vceqq_s32(vMaxScore, vdupq_n_s32(maxv));
        int32x4_t cand = vbslq_s32(eq, vMaxIdx, vdupq_n_s32(INT_MAX));
        bPointer = vminvq_s32(cand);
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