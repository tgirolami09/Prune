#include "NNUE.hpp"
#include "Const.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include <utility>
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "simd_definitions.hpp"

using namespace std;

#ifdef DEBUG_MACRO
StatVar<sbig, 64, 0> TIupdateRemStat;
StatVar<sbig, 64, 0> TIupdateAddStat;
StatVar<sbig, 64, 0> TIupdateTotStat;
StatVar<sbig, 128, -128> TIupdateDiffStat;
#endif
int threatIndex[(nbPieces-1)*2][64][64];
int threatoffset[(nbPieces-1)*2];
const int valid_targets[5] = {3, 5, 4, 4, 5};
const int piecesThreat[nbPieces][nbPieces] = {
    { 0,  1, -1,  2, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    {-1, -1, -1, -1, -1, -1},
};
static_assert(sizeof(threatIndex)+sizeof(threatoffset) < 1024*1024, "way too big for nothing");

__attribute__((constructor(106))) void initThreatIndices(){
    memset(threatIndex, 0xff, sizeof(threatIndex));
    int index=0;
    for(int _atk=0; _atk < (nbPieces-1)*2; _atk++){
        int atk = _atk%5*2+_atk/5;
        int lastindex=index;
        for(int from=0; from < 64; from++){
            if(type(atk) == PAWN && (row(from) == 0 || row(from) == 7))continue;
            big mask=0;
            switch(type(atk)){
                case PAWN:
                    mask = attackPawns[from+color(atk)*64];
                    break;
                case KNIGHT:
                    mask = KnightMoves[from];
                    break;
                case BISHOP:
                    mask = mask_full_bishop(from);
                    break;
                case ROOK:
                    mask = mask_full_rook(from);
                    break;
                case QUEEN:
                    mask = mask_full_rook(from) | mask_full_bishop(from);
                    break;
                default:assert(false);
            }
            ubyte topos[64];
            int nbto = places(mask, topos);
            for(int idto=0; idto < nbto; idto++){
                threatIndex[atk][from^7][topos[idto]^7] = index++;
            }
        }
        int curpiece = index-lastindex;
        threatoffset[atk] = curpiece;
        index += (valid_targets[type(atk)]*2-1)*curpiece;
    }
    //printf("info string counted %d/%d threats\n", index, THREAT_SIZE);
}

int getThreatIndex(Index atk, Index def){
    int index = threatIndex[atk.fullpiece()][atk.square][def.square]+threatoffset[atk.fullpiece()]*(piecesThreat[atk.piece][def.piece]+def.color*valid_targets[atk.piece]);
    return index;
}

int getInputBucket(int Kpos, bool side, bool mirror){
    if(side)Kpos ^= 56;
    if(!mirror)Kpos ^= 7;
    return inputBuckets[(col(Kpos)) | (row(Kpos) << 2)];
}

int turn(int index){
    return ((index^56)+384)%768;
}

Index::Index():square(0), piece(6), color(false){}
Index::Index(int _square, int _piece, bool _color):square(_square), piece(_piece), color(_color){}
void Index::smirror(bool needs){
    square ^= 7*needs;
}
Index Index::mirror(bool needs) const{
    return Index(square^(7*needs), piece, color);
}
void Index::schangepov(){
    square ^= 56;
    color ^= 1;
}
void Index::schangepov(bool needs){
    square ^= 56*needs;
    color ^= needs;
}
Index Index::changepov() const{
    return Index(square^56, piece, !color);
}
Index Index::changepov(bool needs) const{
    return Index(square^(56*needs), piece, color^needs);
}
Index::operator int(){
    int index = ((6*color+piece)<<6)|(square^7);
    return index;
}
bool Index::operator==(const Index a) const{
    return square == a.square && piece == a.piece && color == a.color;
}
bool Index::isnull(){
    return piece == 6;
}
int Index::fullpiece() const{
    return piece*2+color;
}

void Index::print() const{
    printf("%d %d %d", piece, color, square^7);
}
ThreatIndex::ThreatIndex(){}
ThreatIndex::ThreatIndex(Index _from, Index _to):from(_from), to(_to){}
ThreatIndex::ThreatIndex(int fromsquare, int frompiece, int fromcolor, int tosquare, int topiece, int tocolor):
    from(Index(fromsquare, frompiece, fromcolor)),
    to(Index(tosquare, topiece, tocolor)){}

bool ThreatIndex::isexcluded() const{
    return piecesThreat[from.piece][to.piece] == -1;
}
bool ThreatIndex::issemiexcluded() const{
    return from.piece == to.piece && (from.piece != PAWN || from.color != to.color) && (from.square^7) < (to.square^7);
}
ThreatIndex::operator int() const{
    int index = ((int)threatIndex[from.fullpiece()][from.square][to.square])+threatoffset[from.fullpiece()]*(piecesThreat[from.piece][to.piece]+to.color*valid_targets[from.piece]);
    return index;
}
ThreatIndex ThreatIndex::changepov(bool needs) const{
    return ThreatIndex(from.changepov(needs), to.changepov(needs));
}
ThreatIndex ThreatIndex::mirror(bool needs) const{
    return ThreatIndex(from.mirror(needs), to.mirror(needs));
}
void ThreatIndex::swap(){
    std::swap(from, to);
}
ThreatIndex ThreatIndex::rswap() const{
    return ThreatIndex(to, from);
}
void ThreatIndex::print() const{
    from.print();
    printf(" ");
    to.print();
    printf(" => %d\n", (int)*this);
}

updateBuffer::updateBuffer():nbThreats{0, 0}, dirty(true){}
void updateBuffer::reset(Index _add1, Index _add2, Index _sub1, Index _sub2){
    nbThreats[0] = 0;
    nbThreats[1] = 0;
    dirty = true;
    add1[0] = _add1;
    add2[0] = _add2;
    sub1[0] = _sub1;
    sub2[0] = _sub2;
    add1[1] = _add1.changepov();
    add2[1] = _add2.changepov();
    sub1[1] = _sub1.changepov();
    sub2[1] = _sub2.changepov();
    type = !_sub2.isnull()+!_add2.isnull();
}

void updateBuffer::print(){
    printf("%d %d %d; %d %d %d", add1[0].square, add1[0].piece, add1[0].color, sub1[0].square, sub1[0].piece, sub1[0].color);
    if(type >= 1)
        printf("; %d %d %d", sub2[0].square, sub2[0].piece, sub2[0].color);
    if(type == 2)
        printf("; %d %d %d", add2[0].square, add2[0].piece, add2[0].color);
    printf("\n");
}

inline big firstInDirection(int square, int square2, big occupancy){
    big mask = fullDir[square][square2]&occupancy;
    if(!mask)return 0;
    if(square2 > square)
        return mask&-mask;
    else
        return 1ULL << (__builtin_clzll(mask)^63);
}
inline big firstafter(int square, int square2, big occupancy, big atkmask){
    if(!((1ULL << square)&atkmask))return 0;
    return fullDir[square][square2]&occupancy&atkmask;
}

template<bool enPassant, bool tworemove>
void Accumulator::updateXrays(const int8_t mailbox[64], int pos, bool remove, int removepos, int removepos2){
    big masks[3] = {
        moves_table(pos   , occupied&mask_empty_bishop(pos)),
        moves_table(pos+64, occupied&mask_empty_rook  (pos))
    };
    if constexpr(enPassant)
        masks[1] &= ~mask_row[row(pos)];
    masks[2] = masks[0] | masks[1];
    big maskremove = (1ULL << removepos)|firstafter(removepos, pos, occupied, masks[2]);
    if constexpr(tworemove){
        maskremove |= (1ULL << removepos2)|firstafter(removepos2, pos, occupied, masks[2]);
    }
    const big maskFkings = board.pieces[WHITE][KING ] | board.pieces[BLACK][KING ] |
        ((board.pieces[WHITE][KING]&masks[2]) ? fullDir[__builtin_ctzll(board.pieces[WHITE][KING])][pos]&occupied&masks[2] : 0)|
        ((board.pieces[BLACK][KING]&masks[2]) ? fullDir[__builtin_ctzll(board.pieces[BLACK][KING])][pos]&occupied&masks[2] : 0);
    const big filterout = ~(maskremove|maskFkings);
    big mask = (
        (masks[0]&(board.pieces[WHITE][BISHOP] | board.pieces[BLACK][BISHOP])) |
        (masks[1]&(board.pieces[WHITE][ROOK  ] | board.pieces[BLACK][ROOK  ])) |
        (masks[2]&(board.pieces[WHITE][QUEEN ] | board.pieces[BLACK][QUEEN ]))
    ) & filterout;
    while(mask){
        const int posatk = __builtin_ctzll(mask);
        const big maskdef = fullDir[posatk][pos]&masks[2]&occupied;
        if(maskdef){
            const bool coloratk = color(mailbox[posatk]);
            const int pieceatk = type(mailbox[posatk]);
            const int posdef = __builtin_ctzll(maskdef);
            const bool colordef = color(mailbox[posdef]);
            const int piecedef = type(mailbox[posdef]);
            update.addThreat(ThreatIndex(
                Index(posatk, pieceatk, coloratk),
                Index(posdef, piecedef, colordef)
            ).swapExcluded(), remove);
            mask &= ~maskdef;
        }
        mask &= mask-1;
    }
}

void Accumulator::updatePieceOutComing(const int8_t mailbox[64], const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos, const big sliders[3]){
    const Index posatk(pos, piece, colorpiece);
    big atkmask;
    if(piece == PAWN)
        atkmask = attackPawns[pos+colorpiece*64];
    else if(piece == KNIGHT)
        atkmask = KnightMoves[pos];
    else
        atkmask = sliders[piece-BISHOP];
    big authMask = 0;
    for(int x=0; x<nbPieces-1; x++)
        if(piecesThreat[piece][x] != -1)
            authMask |= board.pieces[WHITE][x]|board.pieces[BLACK][x];
    atkmask &= authMask;
    if(removepos != -1)atkmask &= ~(1ULL << removepos);
    while(atkmask){
        const int _posdef = __builtin_ctzll(atkmask);
        const int piecedef = type(mailbox[_posdef]);
        const int colorPiece = color(mailbox[_posdef]);
        update.addThreat(ThreatIndex(
            posatk,
            Index(_posdef, piecedef, colorPiece)
        ), remove);
        atkmask &= atkmask-1;
    }
}

void Accumulator::updatePieceIncoming(const int8_t mailbox[64], const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos, const big sliders[3]){
    const Index posdef(pos, piece, colorpiece);
    const big maskremove = ~((removepos == -1 ? 0 : (1ULL << removepos)) | board.pieces[WHITE][KING] | board.pieces[BLACK][KING]);
    big possMask = 
        (piecesThreat[PAWN][piece] != -1)*(
                            (attackPawns[pos+64*!colorpiece]&board.pieces[ colorpiece][PAWN])|
            (piece != PAWN)*(attackPawns[pos+64* colorpiece]&board.pieces[!colorpiece][PAWN])
        ) | //pawns
        ((piece != KNIGHT)*KnightMoves[pos] & (board.pieces[WHITE][KNIGHT] | board.pieces[BLACK][KNIGHT])) | // knights
        ((piece != QUEEN && piece != BISHOP)*sliders[0] & (board.pieces[WHITE][BISHOP] | board.pieces[BLACK][BISHOP])) | // bishops
        ((piece != QUEEN && piece != ROOK)*sliders[1] & (board.pieces[WHITE][ROOK] | board.pieces[BLACK][ROOK])) | // queens
        ((piece != QUEEN)*sliders[2] & (board.pieces[WHITE][QUEEN] | board.pieces[BLACK][QUEEN])); // queens
    possMask &= maskremove;
    for(; possMask; possMask &= possMask-1){
        const int atkpos = __builtin_ctzll(possMask);
        const bool atkcolor = color(mailbox[atkpos]);
        const int atkpiece = type(mailbox[atkpos]);
        update.addThreat(ThreatIndex(
            Index(atkpos, atkpiece, atkcolor),
            posdef
        ), remove);
    }
}

void Accumulator::updatePiece(const int8_t mailbox[64], const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos){
    big sliders[3] = {
        moves_table(pos   , occupied&mask_empty_bishop(pos)),
        moves_table(pos+64, occupied&mask_empty_rook  (pos)),
    };
    sliders[2] = sliders[0]|sliders[1];
    updatePieceIncoming(mailbox, piece, colorpiece, pos, remove, removepos, sliders);
    updatePieceOutComing(mailbox, piece, colorpiece, pos, remove, removepos, sliders);
}

void Accumulator::getThreatUpdates(const PositionState& state1, const PositionState& state2, const Move& move){
    const int toPiece = move.promotion() == -1 ? move.piece : move.promotion();
    const bool isCapture = move.capture != -2;
    const int capture = move.capture;
    if(move.capture == -1){//en passant
        defstaterelated(state1);
        const int enpassantpos = move.to()+((side == WHITE)?-8:8);
        updatePiece(state1.mailbox, PAWN, side, move.from(), true, -1);
        updateXrays<false, true>(state1.mailbox, move.to(), true, move.from(), enpassantpos);
        updatePiece(state1.mailbox, PAWN, !side, enpassantpos, true, move.from());
        defstaterelated(state2);
        updatePiece(state2.mailbox, PAWN, side, move.to(), false, -1);
        updateXrays(state2.mailbox, move.from(), false, move.to());
        updateXrays<true>(state2.mailbox, enpassantpos, false, move.to()); //remove the common rook side ray
    }else if(move.isCastling()){
        defstaterelated(state1);
        updatePiece(state1.mailbox, ROOK, side, update.sub2[0].square, true, -1);
        defstaterelated(state2);
        updatePiece(state2.mailbox, ROOK, side, update.add2[0].square, false, -1);
    }else{
        defstaterelated(state1);
        //first remove the threat that will disappear because of the move
        if(move.piece != KING)
            updatePiece(state1.mailbox, move.piece, side, move.from(), true, -1);
        if(isCapture){
            updatePiece(state1.mailbox, capture, !side, move.to(), true, move.from()); // threat including move.from has already been removed
        }else
            updateXrays(state1.mailbox, move.to(), true, move.from()); // threat including move.from has already been removed
        //then add the new threats
        defstaterelated(state2);
        if(move.piece != KING)
            updatePiece(state2.mailbox, toPiece, side, move.to(), false, -1);
        updateXrays(state2.mailbox, move.from(), false, move.to()); // threat including move.to() has already been added by addPiece
    }
}

void Accumulator::defstaterelated(const PositionState& _state){
    memcpy(&board, &_state, sizeof(board));
    occupied = 0;
    for(int p=0; p<nbPieces; p++)
        occupied |= board.pieces[WHITE][p];
    for(int p=0; p<nbPieces; p++)
        occupied |= board.pieces[BLACK][p];
}

void Accumulator::reinit(const Move& move, const PositionState& state1, const PositionState& state2, Accumulator& prevAcc, bool _side, bool mirror, Index sub1, Index add1, Index sub2, Index add2){
    side = _side;
    update.reset(add1, add2, sub1, sub2);
    getThreatUpdates(state1, state2, move);
    Kside[0] = prevAcc.Kside[0];
    Kside[1] = prevAcc.Kside[1];
    idInputBucket[0] = prevAcc.idInputBucket[0];
    idInputBucket[1] = prevAcc.idInputBucket[1];
    pstrefresh = false;
    threatrefresh = false;
    if(mirror){
        threatrefresh = true;
        pstrefresh = true;
        Kside[side] ^= 1;
    }
    if(add1.piece == KING && getInputBucket(add1.square, side, Kside[side]) != idInputBucket[side]){ //king moves are always represented in sub1/add1
        idInputBucket[side] = getInputBucket(add1.square, side, Kside[side]);
        pstrefresh = true;
    }
}

void Accumulator::applythreatsUpdates(const Accumulator& accIn, const bool pov){
    if(update.nbThreats[0]+update.nbThreats[1] == 0){
        memcpy(accs[pov+2], accIn.accs[pov+2], sizeof(accs[pov+2]));
        return;
    }
    uint16_t updates[2][32];
    for(int j=0; j<2; j++)
        for(int i=0; i<update.nbThreats[j]; i++){
            updates[j][i] = update.threatUpdates[j][i].changepov(pov).mirror(Kside[pov]).swapSemiExcluded();
            __builtin_prefetch(&globnnue.threatWeights[updates[j][i]]);
        }
    int maxi = update.nbThreats[0] < update.nbThreats[1];
    if(!update.nbThreats[maxi^1]){
        if(maxi)
            globnnue.addThreat<-1>(accIn, *this, pov, updates[maxi][0]);
        else
            globnnue.addThreat< 1>(accIn, *this, pov, updates[maxi][0]);
    }else{
        if(update.nbThreats[maxi^1] >= 2){
            globnnue.add2Threataddsub(accIn, *this, pov, updates[0][0], updates[1][0], updates[0][1], updates[1][1]);
            int i;
            for(i=2; i<update.nbThreats[maxi^1]-1; i += 2)
                globnnue.add2Threataddsub(*this, pov, updates[0][i], updates[1][i], updates[0][i+1], updates[1][i+1]);
            if(i < update.nbThreats[maxi^1])
                globnnue.addThreataddsub(*this, pov, updates[0][i], updates[1][i]);
        }else
            globnnue.addThreataddsub(accIn, *this, pov, updates[0][0], updates[1][0]);
    }
    if(maxi)
        for(int i=max(update.nbThreats[0], 1); i<update.nbThreats[1]; i++)
            globnnue.addThreat<-1>(*this, pov, updates[maxi][i]);
    else
        for(int i=max(update.nbThreats[1], 1); i<update.nbThreats[0]; i++)
            globnnue.addThreat< 1>(*this, pov, updates[maxi][i]);
}

void Accumulator::updateSelf(const Accumulator& accIn, FinnyTables& finny){
#ifdef DEBUG_MACRO
    TIupdateAddStat.update(update.nbThreats[0]);
    TIupdateRemStat.update(update.nbThreats[1]);
    TIupdateTotStat.update(update.nbThreats[0]+update.nbThreats[1]);
    TIupdateDiffStat.update(update.nbThreats[0]-update.nbThreats[1]);
#endif
    if(threatrefresh){
        globnnue.calcThreats(*this, side, board);
        applythreatsUpdates(accIn, !side);
    }else{
        applythreatsUpdates(accIn, WHITE);
        applythreatsUpdates(accIn, BLACK);
    }
    if(pstrefresh){
        int index = (idInputBucket[side]*2+Kside[side])*2+side;
        oneAccumulator& curAcc=finny.normals[index].accs;
        for(int c=0; c<2; c++)
            for(int piece=0; piece<nbPieces; piece++){
                const big common = board.pieces[c][piece]&finny.normals[index].bitboards[c][piece];
                big maskadd = board.pieces[c][piece]&~common;
                big maskrem = finny.normals[index].bitboards[c][piece]&~common;
                while(maskadd && maskrem){
                    const int posrem = __builtin_ctzll(maskrem);
                    const int posadd = __builtin_ctzll(maskadd);
                    globnnue.move2In(curAcc, Index(posrem, piece, c).mirror(Kside[side]).changepov(side), Index(posadd, piece, c).mirror(Kside[side]).changepov(side), idInputBucket[side]);
                    maskrem &= maskrem-1;
                    maskadd &= maskadd-1;
                }
                while(maskrem){
                    const int posrem = __builtin_ctzll(maskrem);
                    globnnue.change1acc<-1>(curAcc, Index(posrem, piece, c).mirror(Kside[side]).changepov(side), idInputBucket[side]);
                    maskrem &= maskrem-1;
                }
                while(maskadd){
                    const int posadd = __builtin_ctzll(maskadd);
                    globnnue.change1acc<1>(curAcc, Index(posadd, piece, c).mirror(Kside[side]).changepov(side), idInputBucket[side]);
                    maskadd &= maskadd-1;
                }
            }
        memcpy(accs[side], curAcc, sizeof(accs[side]));
        memcpy(finny.normals[index].bitboards, board.pieces, sizeof(board.pieces));
        if(update.type == 0)
            globnnue.move2(!side, accIn, *this, update.sub1[!side].mirror(Kside[!side]), update.add1[!side].mirror(Kside[!side]), idInputBucket[!side]);
        else if(update.type == 1)
            globnnue.move3(!side, accIn, *this, update.sub1[!side].mirror(Kside[!side]), update.add1[!side].mirror(Kside[!side]), update.sub2[!side].mirror(Kside[!side]), idInputBucket[!side]);
        else if(update.type == 2)
            globnnue.move4(!side, accIn, *this, update.sub1[!side].mirror(Kside[!side]), update.add1[!side].mirror(Kside[!side]), update.sub2[!side].mirror(Kside[!side]), update.add2[!side].mirror(Kside[!side]), idInputBucket[!side]);
        update.dirty = false;
        return;
    }
    if(update.type == 0){
        globnnue.move2(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]), idInputBucket[WHITE]);
        globnnue.move2(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]), idInputBucket[BLACK]);
    }else if(update.type == 1){
        globnnue.move3(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]), update.sub2[0].mirror(Kside[WHITE]), idInputBucket[WHITE]);
        globnnue.move3(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]), update.sub2[1].mirror(Kside[BLACK]), idInputBucket[BLACK]);
    }else{
        globnnue.move4(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]), update.sub2[0].mirror(Kside[WHITE]), update.add2[0].mirror(Kside[WHITE]), idInputBucket[WHITE]);
        globnnue.move4(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]), update.sub2[1].mirror(Kside[BLACK]), update.add2[1].mirror(Kside[BLACK]), idInputBucket[BLACK]);
    }
    update.dirty = false;
}

void FinnyTables::init(){
    for(int i=0; i<nbInputBuckets*4; i++){
        memset(normals[i].bitboards, 0, sizeof(normals[i].bitboards));
        globnnue.init1Acc(normals[i].accs);
    }
}

template<typename T>
dbyte NNUE::read_bytes(ifstream& file){
    T ret;
    file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    return ret;
}

dbyte read_i16(ifstream& file){
    dbyte ret;
    file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    return ret;
}

dbyte read_i16(const unsigned char* file){
    dbyte ret;
    memcpy(&ret, file, sizeof(ret));
    return ret;
}

// Helper to set individual elements in SIMD vectors
void NNUE::set_simd16_element(simd16& vec, int index, dbyte value) {
    dbyte* ptr = reinterpret_cast<dbyte*>(&vec);
    ptr[index] = value;
}

big genRandom(big& state){
    big z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

NNUE::NNUE(string name){
    if(name == "random"){
        big state = 42;
        for(int idInputBucket=0; idInputBucket<nbInputBuckets; idInputBucket++){
            for(int i=0; i<INPUT_SIZE; i++) {
                for(int j=0; j<HL_SIZE/nb16; j++) {
                    hlWeights[idInputBucket][i][j] = simd16_zero();
                    for(int k=0; k<nb16; k++) {
                        set_simd16_element(hlWeights[idInputBucket][i][j], k, genRandom(state)%256-128);
                    }
                }
            }
        }
        
        for(int i=0; i<HL_SIZE/nb16; i++){
            hlBiases[i] = simd16_zero();
            for(int id16=0; id16<nb16; id16++) {
                set_simd16_element(hlBiases[i], id16, genRandom(state)%256-128);
            }
        }
        
        for(int idB = 0; idB < BUCKET; idB++){
            for(int side=0; side < 2; side++){
                for(int i=0; i<HL_SIZE/nb16; i++) {
                    outWeights[idB][side][i] = simd16_zero();
                    for(int id16=0; id16<nb16; id16++) {
                        set_simd16_element(outWeights[idB][side][i], id16, genRandom(state)%256-128);
                    }
                }
            }
        }
        for(int idB = 0; idB < BUCKET; idB++)
            outbias[idB] = genRandom(state)%256-128;
    }else{
        ifstream file(name);
        file.read(reinterpret_cast<char*>(hlWeights), sizeof(hlWeights));
        file.read(reinterpret_cast<char*>(hlBiases), sizeof(hlBiases));
        file.read(reinterpret_cast<char*>(outWeights), sizeof(outWeights));
        file.read(reinterpret_cast<char*>(outbias), sizeof(outbias));
    }
}
template<typename T>
T get_int(const unsigned char* source, int length){
    T res;
    memcpy(&res, source, length);
    return res;
}

NNUE::NNUE(){
    int pointer = 0;
    memcpy(hlWeights, baseModel, sizeof(hlWeights));
    pointer += sizeof(hlWeights);
    memcpy(hlBiases, &baseModel[pointer], sizeof(hlBiases));
    pointer += sizeof(hlBiases);
    memcpy(outWeights, &baseModel[pointer], sizeof(outWeights));
    pointer += sizeof(outWeights);
    memcpy(outbias, &baseModel[pointer], sizeof(outbias));
}

void NNUE::initAcc(Accumulator& accs) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
    }
}

void NNUE::initAcc(Accumulator& accs, bool color) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[color][i] = hlBiases[i];
    }
}

void NNUE::init1Acc(oneAccumulator& accs) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[i] = hlBiases[i];
    }
}

int NNUE::get_index(int piece, int c, int square) const{
    return ((6*c+piece)<<6)|(square^7);
}

static const simd16 mini = simd16_set1(0);
static const simd16 maxi = simd16_set1(QA);
simdint doOut(simd16 a, simd16 w){
    simd16 clamped = simd16_clamp(a, mini, maxi);
    simd16 mul = simd16_mullo(clamped, w);
    simdint overall = mull_add(mul, clamped);
    return overall;
}

template<int f>
void NNUE::addThreat(Accumulator& accs, bool pov, int index) const{
    static_assert(f == 1 || f == -1, "f should be either 1 or -1");
    const simdhalf* weights = (simdhalf*)&threatWeights[index];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 low=simdh8_16(weights[i]);
        if constexpr (f == 1) {
            accs[pov+2][i] = simd16_add(accs[pov+2][i], low);
        } else {
            accs[pov+2][i] = simd16_sub(accs[pov+2][i], low);
        }
    }
}

void NNUE::addThreataddsub(Accumulator& accs, bool pov, int indexadd, int indexrem) const{
    const simdhalf* weightsadd = (simdhalf*)&threatWeights[indexadd];
    const simdhalf* weightsrem = (simdhalf*)&threatWeights[indexrem];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 add=simdh8_16(weightsadd[i]);
        simd16 rem=simdh8_16(weightsrem[i]);
        simd16 low = simd16_sub(add, rem);
        accs[pov+2][i] = simd16_add(accs[pov+2][i], low);
    }
}


void NNUE::addThreataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, int indexadd, int indexrem) const{
    const simdhalf* weightsadd = (simdhalf*)&threatWeights[indexadd];
    const simdhalf* weightsrem = (simdhalf*)&threatWeights[indexrem];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 add=simdh8_16(weightsadd[i]);
        simd16 rem=simdh8_16(weightsrem[i]);
        simd16 low = simd16_sub(add, rem);
        accs[pov+2][i] = simd16_add(accIn[pov+2][i], low);
    }
}

void NNUE::add2Threataddsub(Accumulator& accs, bool pov, int indexadd1, int indexrem1, int indexadd2, int indexrem2) const{
    const simdhalf* weightsadd1 = (simdhalf*)&threatWeights[indexadd1];
    const simdhalf* weightsadd2 = (simdhalf*)&threatWeights[indexadd2];
    const simdhalf* weightsrem1 = (simdhalf*)&threatWeights[indexrem1];
    const simdhalf* weightsrem2 = (simdhalf*)&threatWeights[indexrem2];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 add1=simdh8_16(weightsadd1[i]);
        simd16 add2=simdh8_16(weightsadd2[i]);
        simd16 rem1=simdh8_16(weightsrem1[i]);
        simd16 rem2=simdh8_16(weightsrem2[i]);
        simd16 low1 = simd16_sub(add1, rem1);
        simd16 low2 = simd16_sub(add2, rem2);
        simd16 low = simd16_add(low1, low2);
        accs[pov+2][i] = simd16_add(accs[pov+2][i], low);
    }
}


void NNUE::add2Threataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, int indexadd1, int indexrem1, int indexadd2, int indexrem2) const{
    const simdhalf* weightsadd1 = (simdhalf*)&threatWeights[indexadd1];
    const simdhalf* weightsadd2 = (simdhalf*)&threatWeights[indexadd2];
    const simdhalf* weightsrem1 = (simdhalf*)&threatWeights[indexrem1];
    const simdhalf* weightsrem2 = (simdhalf*)&threatWeights[indexrem2];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 add1=simdh8_16(weightsadd1[i]);
        simd16 add2=simdh8_16(weightsadd2[i]);
        simd16 rem1=simdh8_16(weightsrem1[i]);
        simd16 rem2=simdh8_16(weightsrem2[i]);
        simd16 low1 = simd16_sub(add1, rem1);
        simd16 low2 = simd16_sub(add2, rem2);
        simd16 low = simd16_add(low1, low2);
        accs[pov+2][i] = simd16_add(accIn[pov+2][i], low);
    }
}


template<int f>
void NNUE::addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, int index) const{
    static_assert(f == 1 || f == -1, "f should be either 1 or -1");
    const simdhalf* weights = (simdhalf*)&threatWeights[index];
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 low=simdh8_16(weights[i]);
        if constexpr (f == 1) {
            accOut[pov+2][i] = simd16_add(accIn[pov+2][i], low);
        } else {
            accOut[pov+2][i] = simd16_sub(accIn[pov+2][i], low);
        }
    }
}

void NNUE::calcThreats(Accumulator& accs, bool pov, const PositionState& state) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[pov+2][i] = simd16_zero();
    }
    big blackbb = 0;
    big whitebb = 0;
    for(int p=0; p<nbPieces; p++)
        whitebb |= state.pieces[WHITE][p];
    for(int p=0; p<nbPieces; p++)
        blackbb |= state.pieces[BLACK][p];
    const big occupied = whitebb | blackbb;
    big authMasks[nbPieces-1];
    for(int i=0; i<nbPieces-1; i++){
        authMasks[i] = 0;
        for(int p=0; p<nbPieces; p++)
            if(p != i && piecesThreat[i][p] != -1)
                authMasks[i] |= state.pieces[WHITE][p] | state.pieces[BLACK][p];
    }
    bool mirror = col(__builtin_ctzll(state.pieces[pov][KING])) <= 3;
    big mask = occupied;
    while(mask){
        const int pos = __builtin_ctzll(mask);
        const int idPiece = state.mailbox[pos];
        big authMask = authMasks[type(idPiece)];
        big semiexcluded = 0;
        if(type(idPiece) == PAWN)
            authMask |= state.pieces[color(idPiece)][type(idPiece)];
        else
            semiexcluded = state.pieces[color(idPiece)][type(idPiece)];
        semiexcluded |= state.pieces[color(idPiece) ^ 1][type(idPiece)];

        big atkmask=0;
        Index posatk(pos, type(idPiece), color(idPiece));
        posatk.smirror(mirror);
        posatk.schangepov(pov);
        switch (type(idPiece)) {
            case PAWN:
                atkmask = attackPawns[pos+color(idPiece)*64];
                break;
            case KNIGHT:
                atkmask = KnightMoves[pos];
                break;
            case BISHOP:
                atkmask = moves_table(pos, occupied&mask_empty_bishop(pos));
                break;
            case ROOK:
                atkmask = moves_table(pos+64, occupied&mask_empty_rook(pos));
                break;
            case QUEEN:
                atkmask = moves_table(pos, occupied&mask_empty_bishop(pos)) | moves_table(pos+64, occupied&mask_empty_rook(pos));
                break;
        }
        big semiEmask = (MAX_BIG>>(63-pos))^(mask_row[row(pos)]*(!mirror^pov));
        if(pov == BLACK)semiEmask = ~semiEmask;
        atkmask &= authMask|(semiexcluded&semiEmask);
        while(atkmask){
            const int _posdef = __builtin_ctzll(atkmask);
            const int piece = type(state.mailbox[_posdef]);
            const int colorPiece = color(state.mailbox[_posdef]);
            Index posdef(_posdef, piece, colorPiece);
            posdef.smirror(mirror);
            posdef.schangepov(pov);
            int threatindex = getThreatIndex(posatk, posdef);
            addThreat<1>(accs, pov, threatindex);
            atkmask &= atkmask-1;
        }
        mask &= mask-1;
    }
}

dbyte NNUE::eval(Accumulator& accs, bool side, int idB) const{
    simdint res = simdint_zero();
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 pov = simd16_add(accs[side][i], accs[side+2][i]);
        simd16 npov = simd16_add(accs[side^1][i], accs[(side^1)+2][i]);
        res = simdint_add(res, doOut(pov, outWeights[idB][0][i]));
        res = simdint_add(res, doOut(npov, outWeights[idB][1][i]));
    }
    int finRes = mysum(res);
    finRes /= QA;
    finRes += outbias[idB];
    finRes = finRes*SCALE/(QA*QB);
    return finRes;
}
template<int f>
void NNUE::change1(Accumulator& accs, bool pov, int index, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accs[pov][i] = simd16_add(accs[pov][i], hlWeights[idInputBucket][index][i]);
        } else {
            accs[pov][i] = simd16_sub(accs[pov][i], hlWeights[idInputBucket][index][i]);
        }
    }
}
template<int f>
void NNUE::change1acc(oneAccumulator& acc, int index, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            acc[i] = simd16_add(acc[i], hlWeights[idInputBucket][index][i]);
        } else {
            acc[i] = simd16_sub(acc[i], hlWeights[idInputBucket][index][i]);
        }
    }
}
template<int f>
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accOut[pov][i] = simd16_add(accIn[pov][i], hlWeights[idInputBucket][index][i]);
        } else {
            accOut[pov][i] = simd16_sub(accIn[pov][i], hlWeights[idInputBucket][index][i]);
        }
    }
}
void NNUE::move3(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[idInputBucket][indexto][i], simd16_add(hlWeights[idInputBucket][indexfrom][i], hlWeights[idInputBucket][indexcap][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}
void NNUE::move2(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[idInputBucket][indexto][i], hlWeights[idInputBucket][indexfrom][i]);
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}

void NNUE::move2In(oneAccumulator& acc, int indexfrom, int indexto, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[idInputBucket][indexto][i], hlWeights[idInputBucket][indexfrom][i]);
        acc[i] = simd16_add(acc[i], update);
    }
}

void NNUE::move4(int color, const Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(simd16_add(hlWeights[idInputBucket][indexto1][i], hlWeights[idInputBucket][indexto2][i]), simd16_add(hlWeights[idInputBucket][indexfrom1][i], hlWeights[idInputBucket][indexfrom2][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}

void NNUE::updateStack(Accumulator* stack, int stackIndex, FinnyTables& tables) const{
    int startUpdate;
    for(startUpdate=stackIndex; startUpdate >= 1 && stack[startUpdate].update.dirty; startUpdate--);
    startUpdate++;
    for(int i=startUpdate; i<=stackIndex; i++){
        stack[i].updateSelf(stack[i-1], tables);
    }
}

template void NNUE::change1<-1>(Accumulator&, bool, int, int) const;
template void NNUE::change1<1>(Accumulator&, bool, int, int) const;
template void NNUE::change2<-1>(Accumulator&, Accumulator&, bool, int, int) const;
template void NNUE::change2<1>(Accumulator&, Accumulator&, bool, int, int) const;
