#include "Evaluator.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "NNUE.hpp"
#include <assert.h>
#include <cstring>
#include <iso646.h>

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

big get_rook_lines(big occupancy, int square){
    return moves_table(square+64, occupancy);
}
big get_bishop_lines(big occupancy, int square){
    return moves_table(square, occupancy);
}

inline int getLVA(int square, const GameState& state, bool stm, big occupancy, int& pieceType){ // return the square where the lva come from, set pieceType
    //Pawns
    big mask = occupancy&state.boardRepresentation[stm][PAWN]&attackPawns[(!stm)*64+square];
    if(mask){
        pieceType = PAWN;
        return __builtin_ctzll(mask);
    }
    //Knight
    mask = occupancy&state.boardRepresentation[stm][KNIGHT]&KnightMoves[square];
    if(mask){
        pieceType = KNIGHT;
        return __builtin_ctzll(mask);
    }
    //Bishop
    big maskB = occupancy&get_bishop_lines(occupancy&mask_empty_bishop(square), square);
    mask = state.boardRepresentation[stm][BISHOP]&maskB;
    if(mask){
        pieceType = BISHOP;
        return __builtin_ctzll(mask);
    }
    //Rook
    big maskR = occupancy&get_rook_lines(occupancy&mask_empty_rook(square), square);
    mask = state.boardRepresentation[stm][ROOK]&maskR;
    if(mask){
        pieceType = ROOK;
        return __builtin_ctzll(mask);
    }
    //Queen
    mask = state.boardRepresentation[stm][QUEEN]&(maskR|maskB);
    if(mask){
        pieceType = QUEEN;
        return __builtin_ctzll(mask);
    }
    //KING
    mask = occupancy&state.boardRepresentation[stm][KING]&normalKingMoves[square];
    if(mask){
        pieceType = KING;
        return __builtin_ctzll(mask);
    }
    return -1;
}

int fastSEE(const Move& move, const GameState& state){
    big occupancy = 0;
    for(int c=0; c<2; c++)
        for(int p=0; p<6; p++)
            occupancy |= state.boardRepresentation[c][p];
    occupancy ^= 1ULL << move.from();
    int square = move.to();
    bool stm = !state.friendlyColor();
    int atk;
    int pieceType;
    ubyte stack[16];
    int idStack = 0;
    int lastPiece = move.piece;
    while((atk = getLVA(square, state, stm, occupancy, pieceType)) != -1){
        stack[idStack++] = lastPiece;
        occupancy ^= 1ULL << atk;
        lastPiece = pieceType;
        stm = !stm;
    }
    int res = 0;
    idStack--;
    for(;idStack >= 0; idStack--){
        res = max(0, value_pieces[stack[idStack]]-res);
    }
    return res;
}

big get_mask(const GameState& state, int p){
    return state.boardRepresentation[0][p]|state.boardRepresentation[1][p];
}

SEE_BB::SEE_BB(const GameState& state){
    Qs = get_mask(state, QUEEN);
    Rs = get_mask(state, ROOK)|Qs;
    Bs = get_mask(state, BISHOP)|Qs;
    Ns = get_mask(state, KNIGHT);
    Ks = get_mask(state, KING);
    occupancy = 0;
    for(int _c=0; _c<2; _c++)
        for(int p=0; p<6; p++)
            occupancy |= state.boardRepresentation[_c][p];
}

bool see_ge(const SEE_BB& bb, int born, const Move& move, const GameState& state){
    int square = move.to();
    //occupancy ^= 1ULL << move.from();
    bool stm = state.friendlyColor();
    big atk = 1ULL << move.from();
    int lastPiece = move.capture != -2 ? max<int8_t>(0, move.capture) : 6;
    int pieceType = move.piece;
    bool sstm = stm;
    big occupancy = bb.occupancy ^ atk;
    born = value_pieces[lastPiece]-born;
    stm = !stm;
    lastPiece = pieceType;
    if(born < 0)
        return false;
    big bishopAtk = mask_empty_bishop(square);
    big rooksAtk = mask_empty_rook(square);
    big attacks =((get_bishop_lines(occupancy&bishopAtk, square)&bb.Bs) | (get_rook_lines(occupancy&rooksAtk, square)&bb.Rs) |
                  (KnightMoves[square]&bb.Ns) |
                  (attackPawns[square]&state.boardRepresentation[1][PAWN]) | (attackPawns[square+64]&state.boardRepresentation[0][PAWN]) |
                  (normalKingMoves[square]&bb.Ks))&occupancy;
    while(1){
        pieceType = -1;
        for(int p=0; p<nbPieces; p++){
            big mask = state.boardRepresentation[stm][p]&attacks;
            if(mask){
                atk = mask&-mask;
                pieceType = p;
                break;
            }
        }
        if((pieceType == KING && countbit(attacks) > 1) || pieceType == -1)
            break;
        occupancy ^= atk;
        //printf("atk=%d p=%d b=%d ", atk, pieceType, born);
        born = value_pieces[lastPiece]-born;
        //printf("nb=%d\n", born);
        stm = !stm;
        lastPiece = pieceType;
        if(stm == sstm){
            if(born <= 0)return true;
        }else if(born < 0)
            return false;
        if(lastPiece >= QUEEN)
            attacks |= (get_bishop_lines(occupancy&bishopAtk, square)&bb.Bs) | (get_rook_lines(occupancy&rooksAtk, square)&bb.Rs);
        else if(!(lastPiece & 1))
            attacks |= get_bishop_lines(occupancy&bishopAtk, square)&bb.Bs;
        else if(lastPiece == ROOK)
            attacks |= get_rook_lines(occupancy&rooksAtk, square)&bb.Rs;
        attacks &= occupancy;
    }
    //printf("%d %d %d\n", stm, sstm, born);
    return stm != sstm || born <= 0;
}

int score_move(const Move& move, int historyScore, const SEE_BB& bb, const GameState& state, ubyte& flag){
    int score = 0;
    flag = 0;
    if(see_ge(bb, 0, move, state)){
        flag += 1;
    }if(move.isTactical()){
        int cap = move.capture;
        if(cap == -1)cap = 0;
        if(cap != -2)
            score = value_pieces[cap]*10;
        score -= value_pieces[move.piece];
        flag += 2;
        if(move.promotion() != -1)score += value_pieces[move.promotion()];
    }else{
        score += historyScore;
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

IncrementalEvaluator::IncrementalEvaluator(){
    init_tables();
    init_forwards();
    memset(presentPieces, 0, sizeof(presentPieces));
}

void IncrementalEvaluator::init(const GameState& state){//should be only call at the start of the search
    mgPhase = 0;
    stackIndex = 0;
    globnnue.initAcc(stackAcc[stackIndex]);
    memset(presentPieces, 0, sizeof(presentPieces));
    for(int square=0; square<64; square++){
        int piece=state.getfullPiece(square);
        if(type(piece) != SPACE){
            changePiece<1, true>(square, type(piece), color(piece));
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
    return globnnue.eval(stackAcc[stackIndex], c);
}

int IncrementalEvaluator::getScore(bool c, const corrhists& ch, const GameState& state) const{
    int raw_eval = globnnue.eval(stackAcc[stackIndex], c);
    return raw_eval+ch.probe(state);
}
void IncrementalEvaluator::undoMove(Move move, bool c){
    playMove<-1>(move, c);
}

template<int f, bool updateNNUE>
void IncrementalEvaluator::changePiece(int pos, int piece, bool c){
    if(updateNNUE)
        globnnue.change2<f>(stackAcc[stackIndex], piece*2+c, pos);
    mgPhase += f*gamephaseInc[piece];
    presentPieces[c][piece] += f;
}


template<int f, bool updateNNUE>
void IncrementalEvaluator::changePiece2(int pos, int piece, bool c){
    if(updateNNUE){
        globnnue.change2<f>(stackAcc[stackIndex], stackAcc[stackIndex+1], piece*2+c, pos);
        stackIndex++;
    }else{
        stackIndex--;
    }
    mgPhase += f*gamephaseInc[piece];
    presentPieces[c][piece] += f;
}


template<int f>
void IncrementalEvaluator::playMove(Move move, bool c){
    int toPiece = move.piece;
    if(move.promotion() != -1){
        toPiece = move.promotion();
        changePiece<-f, false>(move.from(), move.piece, c);
        changePiece<f, false>(move.to(), toPiece, c);
    }
    if(move.capture != -2){
        int posCapture = move.to();
        int pieceCapture = move.capture;
        if(move.capture == -1){ // for en passant
            if(c == WHITE)posCapture -= 8;
            else posCapture += 8;
            pieceCapture = PAWN;
        }
        changePiece<-f, false>(posCapture, pieceCapture, !c);
        if(f == 1)
            globnnue.move3(stackAcc[stackIndex], stackAcc[stackIndex+1],
                globnnue.get_index(move.piece*2+c, move.from()),
                globnnue.get_index(toPiece*2+c, move.to()),
                globnnue.get_index(pieceCapture*2+!c, posCapture)
            );
    }else if(move.piece == KING && abs(move.from()-move.to()) == 2){ //castling
        int rookStart = move.from();
        int rookEnd = move.to();
        if(move.from() > move.to()){//queen side
            rookStart &= ~7;
            rookEnd++;
        }else{//king side
            rookStart |= 7;
            rookEnd--;
        }
        if(f == 1)
            globnnue.move4(stackAcc[stackIndex], stackAcc[stackIndex+1],
                globnnue.get_index(move.piece*2+c, move.from()),
                globnnue.get_index(toPiece*2+c, move.to()),
                globnnue.get_index(ROOK*2+c, rookStart), 
                globnnue.get_index(ROOK*2+c, rookEnd)
            );
    }else if(f == 1){
        globnnue.move2(stackAcc[stackIndex], stackAcc[stackIndex+1],
            globnnue.get_index(move.piece*2+c, move.from()),
            globnnue.get_index(toPiece*2+c, move.to())
        );
    }
    if(f == 1)
        stackIndex++;
    else
        stackIndex--;
}

void IncrementalEvaluator::backStack(){
    stackIndex--;
}

void IncrementalEvaluator::playNoBack(Move move, bool c){
    int toPiece = (move.promotion() == -1) ? move.piece : move.promotion(); //for promotion
    changePiece<-1, true>(move.from(), move.piece, c);
    changePiece<1, true>(move.to(), toPiece, c);
    if(move.capture != -2){
        int posCapture = move.to();
        int pieceCapture = move.capture;
        if(move.capture == -1){ // for en passant
            if(c == WHITE)posCapture -= 8;
            else posCapture += 8;
            pieceCapture = PAWN;
        }
        changePiece<-1, true>(posCapture, pieceCapture, !c);
    }
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
        changePiece<-1, true>(rookStart, ROOK, c);
        changePiece<1, true>(rookEnd, ROOK, c);
    }

}

template void IncrementalEvaluator::playMove<-1>(Move, bool);
template void IncrementalEvaluator::playMove<1>(Move, bool);
template void IncrementalEvaluator::changePiece2<-1, true>(int, int, bool);
template void IncrementalEvaluator::changePiece2<1, true>(int, int, bool);
template void IncrementalEvaluator::changePiece2<-1, false>(int, int, bool);
template void IncrementalEvaluator::changePiece2<1, false>(int, int, bool);