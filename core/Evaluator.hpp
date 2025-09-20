#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "NNUE.cpp"
#include <climits>
#include <cstring>
#include <cmath>
//#define NNUE_CORRECT
//https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
const int mg_value[6] = { 82, 337, 365, 477, 1025,  0};
const int eg_value[6] = { 94, 281, 297, 512,  936,  0};

/* piece/sq tables */
/* values from Rofchade: http://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19 */

const int mg_pawn_table[64] = {
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
};

const int eg_pawn_table[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};

const int mg_knight_table[64] = {
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
};

const int eg_knight_table[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};

const int mg_bishop_table[64] = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};

const int eg_bishop_table[64] = {
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
};

const int mg_rook_table[64] = {
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
};

const int eg_rook_table[64] = {
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
};

const int mg_queen_table[64] = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};

const int eg_queen_table[64] = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};

const int mg_king_table[64] = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};

const int eg_king_table[64] = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
};

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

const int gamephaseInc[6] = {0,1,1,2,4,0};
static int mg_table[2][6][64];
static int eg_table[2][6][64];

void init_tables()
{
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

//Class to evaluate a position
static big mask_forward[64];
static big mask_forward_inv[64];
void init_forwards(){
    for(int square=0; square<64; square++){
        big triCol = (colH << col(square)) | (colH << max(0, col(square)-1)) | (colH << min(7, col(square)+1));
        mask_forward[square] = (MAX_BIG << (row(square)+1)*8) & triCol;
        mask_forward_inv[square] = (MAX_BIG >> (8-row(square))*8) & triCol;
    }
}

int SEE(int square, GameState& state, LegalMoveGenerator& generator){
    Move goodMove = generator.getLVA(square, state);
    int value = 0;
    if(goodMove.moveInfo != nullMove.moveInfo){
        state.playMove<false, false>(goodMove);
        int SEErec = value_pieces[goodMove.capture < 0?0:goodMove.capture]-SEE(square, state, generator);
        if(goodMove.promotion() != -1)
            SEErec += value_pieces[goodMove.promotion()];
        value = max(0, SEErec);
        state.undoLastMove<false>();
    }
    return value;
}

inline int score_move(const Move& move, bool c, big& dangerPositions, int historyScore, bool useSEE, GameState& state, ubyte& flag, LegalMoveGenerator& generator){
    int score = 0;
    int SEEscore = 0;
    flag = 0;
    if(useSEE){
        state.playMove<false, false>(move);
        SEEscore = -SEE(move.to(), state, generator);
        if(move.capture != -2)
            SEEscore += value_pieces[move.capture == -1?0:move.capture];
        state.undoLastMove<false>();
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
    score += mg_table[c][move.piece][move.to()]-mg_table[c][move.piece][move.from()];
    return score;
}

static const int tableSize=1<<10;//must be a power of two, for now it's pretty small because we should hit the table very often, and so we didn't use too much memory

class IncrementalEvaluator{
    int mgPhase;
#ifdef NNUE_CORRECT
    int mgScore, egScore;
#endif
    int presentPieces[2][6]; //keep trace of number of pieces by side
    template<int f>
    void changePiece(int pos, int piece, bool c){
        int sign = (c == WHITE) ? 1 : -1;
#ifdef NNUE_CORRECT
        mgScore += f*sign*mg_table[c][piece][pos];
        egScore += f*sign*eg_table[c][piece][pos];
#endif
        nnue.change2<f>(piece*2+c, pos);
        mgPhase += f*gamephaseInc[piece];
        presentPieces[c][piece] += f;
    }
public:
    NNUE nnue;
    void print(){
        printf("phase = %d\n", mgPhase);
        for(int i=0; i<2; i++){
            for(int j=0; j<6; j++){
                printf("piece = %d, color = %d, nbPieces = %d\n", j, i, presentPieces[i][j]);
            }
        }
    }

    IncrementalEvaluator():nnue("../nnue/model.bin"){
        init_tables();
        init_forwards();
#ifdef NNUE_CORRECT
        mgScore = egScore = 0;
#endif
        memset(presentPieces, 0, sizeof(presentPieces));
    }

    void init(const GameState& state){//should be only call at the start of the search
        mgPhase = 0;
#ifdef NNUE_CORRECT
        mgScore = egScore = 0;
#endif
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

    bool isInsufficientMaterial(){
        if(mgPhase < 4 && !presentPieces[WHITE][PAWN] && !presentPieces[BLACK][PAWN] && !presentPieces[WHITE][QUEEN] && !presentPieces[BLACK][QUEEN] && !presentPieces[WHITE][ROOK] && !presentPieces[BLACK][ROOK]){
            //theoric draw must have only knight or bishop, and must have at most 4 pieces (2 knight per side is a draw for a computer)
            if(presentPieces[WHITE][BISHOP])
                return presentPieces[WHITE][BISHOP] < 2 && presentPieces[WHITE][KNIGHT] == 0;
            if(presentPieces[BLACK][BISHOP])
                return presentPieces[BLACK][BISHOP] < 2 && presentPieces[BLACK][KNIGHT] == 0;
            return presentPieces[BLACK][KNIGHT] <= 2 && presentPieces[WHITE][KNIGHT] <= 2;
        }
        return false;
    }

    inline bool isOnlyPawns() const{
        return !mgPhase;
    }

    int getScore(bool c, pawnStruct s){
#ifdef NNUE_CORRECT
        int clampPhase = min(mgPhase, 24);
        int score = (clampPhase*mgScore+(24-clampPhase)*egScore)/24;
        if(c == BLACK)score = -score;
        int nnueCorrection = nnue.eval(c);
        //printf("%d\n", nnueCorrection);
        return score-nnueCorrection;
#else
        return nnue.eval(c);
#endif
    }
    template<int f=1>
    void playMove(Move move, bool c){
        if(move.capture != -2){
            int posCapture = move.to();
            int pieceCapture = move.capture;
            if(move.capture == -1){ // for en passant
                if(c == WHITE)posCapture -= 8;
                else posCapture += 8;
                pieceCapture = PAWN;
            }
            changePiece<-f>(posCapture, pieceCapture, !c);
        }
        int toPiece = (move.promotion() == -1) ? move.piece : move.promotion(); //for promotion
        changePiece<-f>(move.from(), move.piece, c);
        changePiece<f>(move.to(), toPiece, c);
        if(move.piece == KING && abs(move.from()-move.to()) == 2){ //castling
            int rookStart = move.from();
            int rookEnd = move.to();
            if(move.from() > move.to()){//queen side
                rookStart &= ~7;
                rookEnd++;
            }else{//king side
                rookStart |= 7;
                rookEnd--;
            }
            changePiece<-f>(rookStart, ROOK, c);
            changePiece<f>(rookEnd, ROOK, c);
        }
    }
    void undoMove(Move move, bool c){
        playMove<-1>(move, c);
    }
};
#endif