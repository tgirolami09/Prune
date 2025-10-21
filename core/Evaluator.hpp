#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#ifdef __SSE2__
    #include "NNUE.cpp"
#else
    #include "NNUEnoSIMD.cpp"
#endif
#include <climits>
#include <cstring>
#include <cmath>
const int gamephaseInc[6] = {0,1,1,2,4,0};

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

inline int score_move(const Move& move, big& dangerPositions, int historyScore, bool useSEE, GameState& state, ubyte& flag, LegalMoveGenerator& generator){
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
    return score;
}

static const int tableSize=1<<10;//must be a power of two, for now it's pretty small because we should hit the table very often, and so we didn't use too much memory

class IncrementalEvaluator{
    int mgPhase;
    int nbPieces;
    int presentPieces[2][6]; //keep trace of number of pieces by side
public:
    template<int f>
    void changePiece(int pos, int piece, bool c){
        nnue.change2<f>(piece*2+c, pos);
        mgPhase += f*gamephaseInc[piece];
        presentPieces[c][piece] += f;
        nbPieces += f;
    }
    NNUE nnue;
    void print(){
        printf("phase = %d\n", mgPhase);
        for(int i=0; i<2; i++){
            for(int j=0; j<6; j++){
                printf("piece = %d, color = %d, nbPieces = %d\n", j, i, presentPieces[i][j]);
            }
        }
    }

    IncrementalEvaluator():nnue(){
        memset(presentPieces, 0, sizeof(presentPieces));
    }

    void init(const GameState& state){//should be only call at the start of the search
        mgPhase = 0;
        nbPieces = 0;
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
        if(mgPhase <= 1 && !presentPieces[WHITE][PAWN] && !presentPieces[BLACK][PAWN]){
            return true;
        }
        return false;
    }

    inline bool isOnlyPawns() const{
        return !mgPhase;
    }

    int getScore(bool c){
        return nnue.eval(c, (nbPieces-2)/DIVISOR);
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