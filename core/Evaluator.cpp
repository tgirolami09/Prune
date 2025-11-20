#include "Evaluator.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include <cstring>

const int* mg_pesto_table[6] =
{
    mg_pawn_table,
    mg_knight_table,
    mg_bishop_table,
    mg_rook_table,
    mg_queen_table,
    mg_king_table
};

const int* eg_pesto_table[6] =
{
    eg_pawn_table,
    eg_knight_table,
    eg_bishop_table,
    eg_rook_table,
    eg_queen_table,
    eg_king_table
};
int mg_table[2][6][64];
int eg_table[2][6][64];
void init_tables(){
    int p, sq;
    for (p = PAWN; p <= KING; p++) {
        for (sq = 0; sq < 64; sq++) {
            mg_table[WHITE][p][sq] = mg_value[p] + mg_pesto_table[p][sq^63];
            eg_table[WHITE][p][sq] = eg_value[p] + eg_pesto_table[p][sq^63];
            mg_table[BLACK][p][sq] = mg_value[p] + mg_pesto_table[p][sq^7];
            eg_table[BLACK][p][sq] = eg_value[p] + eg_pesto_table[p][sq^7];
        }
    }
}
big mask_forward[64];
big mask_forward_inv[64];
void init_forwards(){
    for(int square=0; square<56; square++){
        big triCol = (colH << col(square)) | (colH << max(0, col(square)-1)) | (colH << min(7, col(square)+1));
        if(row(square) != 7)mask_forward[square] = (MAX_BIG << (row(square)+1)*8) & triCol;
        else mask_forward[square] = 0;
        if(row(square) != 0)mask_forward_inv[square] = (MAX_BIG >> (8-row(square))*8) & triCol;
        else mask_forward_inv[square] = 0;
    }
}

int SEE(int square, GameState& state, LegalMoveGenerator& generator){
    Move goodMove = generator.getLVA(square, state);
    int value = 0;
    if(goodMove.moveInfo != nullMove.moveInfo){
        state.playMove(goodMove);
        int SEErec = value_pieces[goodMove.capture < 0?0:goodMove.capture]-SEE(square, state, generator);
        if(goodMove.promotion() != -1)
            SEErec += value_pieces[goodMove.promotion()];
        value = max(0, SEErec);
        state.undoLastMove();
    }
    return value;
}

int score_move(const Move& move, big& dangerPositions, int historyScore, bool useSEE, GameState& state, ubyte& flag, LegalMoveGenerator& generator){
    int score = 0;
    int SEEscore = 0;
    flag = 0;
    if(useSEE){
        state.playMove(move);
        SEEscore = -SEE(move.to(), state, generator);
        if(move.capture != -2)
            SEEscore += value_pieces[move.capture == -1?0:move.capture];
        state.undoLastMove();
        if(SEEscore > 0)
            flag += 2;
    }else if(move.isTactical()){
        int cap = move.capture;
        if(cap == -1)cap = 0;
        if(cap != -2)
            SEEscore = value_pieces[cap]*10;
        if((1ULL << move.to())&dangerPositions)
            SEEscore -= value_pieces[move.piece];
    }
    if(!move.isTactical()){
        score += historyScore;
        score += SEEscore*maxHistory;
    }else{
        flag++;
        score += SEEscore;
        if(move.promotion() != -1)score += value_pieces[move.promotion()];
    }
    return score;
}

void IncrementalEvaluator::print(){
    printf("phase = %d\n", mgPhase);
    for(int i=0; i<2; i++){
        for(int j=0; j<6; j++){
            printf("piece = %d, color = %d, nbPieces = %d\n", j, i, presentPieces[i][j]);
        }
    }
}

IncrementalEvaluator::IncrementalEvaluator():nnue(){
    init_tables();
    init_forwards();
    memset(presentPieces, 0, sizeof(presentPieces));
}

void IncrementalEvaluator::init(const GameState& state){//should be only call at the start of the search
    mgPhase = 0;
    nnue.clear();
    memset(presentPieces, 0, sizeof(presentPieces));
    for(int square=0; square<64; square++){
        int piece=state.getfullPiece(square);
        if(type(piece) != SPACE){
            changePiece<1>(square, type(piece), color(piece));
            //printf("intermediate eval : %d\n", getScore(state.friendlyColor()));
        }
    }
}

bool IncrementalEvaluator::isInsufficientMaterial() const{
    if(mgPhase <= 1 && !presentPieces[WHITE][PAWN] && !presentPieces[BLACK][PAWN]){
        return true;
    }
    return false;
}

bool IncrementalEvaluator::isOnlyPawns() const{
    return !mgPhase;
}

int IncrementalEvaluator::getRaw(bool c) const{
    return nnue.eval(c);
}

int IncrementalEvaluator::getScore(bool c, const corrhists& ch, const GameState& state) const{
    int raw_eval = nnue.eval(c);
    return raw_eval+ch.probe(state);
}
void IncrementalEvaluator::undoMove(Move move, bool c){
    playMove<-1>(move, c);
}