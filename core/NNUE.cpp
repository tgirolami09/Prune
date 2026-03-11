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

uint16_t threatIndex[(nbPieces-1)*2][64][64];
uint16_t threatoffset[(nbPieces-1)*2];
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
    printf("info string counted %d/%d threats\n", index, THREAT_SIZE);
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
    return ((6*color+piece)<<6)|(square^7);
}
bool Index::isnull(){
    return piece == 6;
}
int Index::fullpiece() const{
    return piece*2+color;
}

void Index::print(){
    printf("%d %d %d", piece, color, square^7);
}
ThreatIndex::ThreatIndex(){}
ThreatIndex::ThreatIndex(Index _from, Index _to, bool _remove):from(_from), to(_to), remove(_remove){}
ThreatIndex::ThreatIndex(int fromsquare, int frompiece, int fromcolor, int tosquare, int topiece, int tocolor, bool _remove):
    from(Index(fromsquare, frompiece, fromcolor)),
    to(Index(tosquare, topiece, tocolor)),
    remove(_remove){}

bool ThreatIndex::isexcluded() const{
    return piecesThreat[from.piece][to.piece] == -1;
}
bool ThreatIndex::issemiexcluded() const{
    return from.piece == to.piece && from.square > to.square;
}
ThreatIndex::operator int() const{
    return threatIndex[from.fullpiece()][from.square][to.square]+threatoffset[from.fullpiece()]*(piecesThreat[from.piece][to.piece]+to.color*valid_targets[from.piece]);
}
void ThreatIndex::changepov(bool needs){
    from.schangepov(needs);
    to.schangepov(needs);
}
void ThreatIndex::mirror(bool needs){
    from.smirror(needs);
    to.smirror(needs);
}

updateBuffer::updateBuffer():nbThreats(0), dirty(true){}
updateBuffer::updateBuffer(Index _add1, Index _add2, Index _sub1, Index _sub2):nbThreats(0), dirty(true){
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

big firstInDirection(int square, int square2, big occupancy){
    big mask = fullDir[square][square2]&occupancy;
    if(!mask)return 0;
    if(square2 > square)
        return mask&-mask;
    else
        return 1ULL << (__builtin_clzll(mask)^63);
}

void Accumulator::addXrays(const GameState* state, int pos, bool remove){
    for(auto x:{make_pair(ROOK, moves_table(pos, occupied&mask_empty_rook(pos))), make_pair(BISHOP, moves_table(pos, occupied&mask_empty_bishop(pos)))}){
        big maskPiece = 0;
        for(int piece:{x.first, QUEEN})
            maskPiece |= state->boardRepresentation[WHITE][piece] | state->boardRepresentation[BLACK][piece];
        big mask = x.second&maskPiece; //only cares about the pieces
        while(mask){
            const int posatk = __builtin_ctzll(mask);
            const big maskdef = firstInDirection(posatk, pos, occupied);
            if(maskdef){
                const int posdef = __builtin_ctzll(maskdef);
                const bool colordef = (maskdef&whitebb)?WHITE:BLACK;
                const bool coloratk = ((1ULL << posatk)&whitebb)?WHITE:BLACK;
                if(colordef == BLACK)assert(maskdef&blackbb);
                const int piecedef = state->getPiece(posdef, colordef);
                assert(type(state->getfullPiece(posdef)) == piecedef);
                assert(piecedef != SPACE);
                const int pieceatk = state->getPiece(posatk, coloratk);
                if(piecedef != KING){
                    const ThreatIndex threat(Index(posatk, pieceatk, coloratk), Index(posdef, piecedef, colordef), remove);
                    if(!threat.isexcluded())
                        update.threatUpdates[update.nbThreats++] = threat;
                }
            }
            mask &= mask-1;
        }
    }
}

void Accumulator::addPiece(int piece, bool colorpiece, int pos, bool remove){
    Index posatk(pos, piece, colorpiece);
    big atkmask;
    switch (piece) {
        case PAWN:
            atkmask = attackPawns[pos+colorpiece*64];
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
        default:assert(false);
    }
    big authMask = 0;
    for(int _c=0; _c<2; _c++)
        for(int p=0; p<nbPieces; p++)
            if(piecesThreat[piece][p] != -1)
                authMask |= bitboards[_c][p];
    atkmask &= authMask;
    while(atkmask){
        const int _posdef = __builtin_ctzll(atkmask);
        const big maskPiece = 1ULL << _posdef;
        int piecedef = -1;
        const int colorPiece = (whitebb&maskPiece)?WHITE:BLACK;
        for(int x=0; x<nbPieces; x++)
            if(bitboards[colorPiece][x]&maskPiece){
                piecedef = x;
                break;
            }
        assert(piecedef != -1);
        Index posdef(_posdef, piecedef, colorPiece);
        ThreatIndex threat(posatk, posdef, remove);
        assert(!threat.isexcluded());
        update.threatUpdates[update.nbThreats++] = threat;
        atkmask &= atkmask-1;
    }
}

void Accumulator::getThreatUpdates(const GameState* state, const Move& move){
    const int toPiece = move.promotion() == -1 ? move.piece : move.promotion();
    const bool isCapture = move.capture != -2;
    const int capture = move.capture;
    if(move.capture == -1){//en passant
        threatfullupdate = true;
    }else if(move.isCastling()){
        threatfullupdate = true;
    }else{
        threatfullupdate = false;
        if(toPiece != KING){
            addPiece(toPiece, side, move.to(), false);
            addPiece(move.piece, side, move.from(), true);
        }
        if(isCapture){
            addPiece(capture, !side, move.to(), true);
            update.threatUpdates[update.nbThreats++] = ThreatIndex(Index(move.from(), move.piece, side), Index(move.to(), capture, !side), true);
        }else
           addXrays(state, move.to(), true);
        addXrays(state, move.from(), false);
        //printf("%s %d\n", move.to_str().c_str(), update.nbThreats);
    }
}

void Accumulator::reinit(const Move& move, const GameState* state, Accumulator& prevAcc, bool _side, bool mirror, Index sub1, Index add1, Index sub2, Index add2){
    memcpy(bitboards, state->boardRepresentation, sizeof(bitboards));
    update = updateBuffer(add1, add2, sub1, sub2);
    blackbb = 0;
    whitebb = 0;
    for(int p=0; p<nbPieces; p++)
        whitebb |= bitboards[WHITE][p];
    for(int p=0; p<nbPieces; p++)
        blackbb |= bitboards[BLACK][p];
    occupied = whitebb | blackbb;
    getThreatUpdates(state, move);
    Kside[0] = prevAcc.Kside[0];
    Kside[1] = prevAcc.Kside[1];
    idInputBucket[0] = prevAcc.idInputBucket[0];
    idInputBucket[1] = prevAcc.idInputBucket[1];
    pstrefresh = false;
    threatrefresh = false;
    if(mirror){
        threatrefresh = true;
        pstrefresh = true;
        Kside[_side] ^= 1;
    }
    if(add1.piece == KING && getInputBucket(add1.square, _side, Kside[_side]) != idInputBucket[_side]){ //king moves are always represented in sub1/add1
        idInputBucket[_side] = getInputBucket(add1.square, _side, Kside[_side]);
        pstrefresh = true;
    }
    side = _side;
}

void Accumulator::applythreatsUpdates(bool pov){
    if(update.nbThreats < 0 || update.nbThreats > maxThreatUpdates){
        printf("%d\n", update.nbThreats);
    }
    assert(update.nbThreats >= 0);
    assert(update.nbThreats < maxThreatUpdates);
    for(int i=0; i<update.nbThreats; i++){
        ThreatIndex curThreat = update.threatUpdates[i];
        curThreat.changepov(pov);
        curThreat.mirror(Kside[pov]);
        if(curThreat.issemiexcluded())continue;
        if(curThreat.remove)
            globnnue.addThreat<-1>(*this, pov, curThreat);
        else
            globnnue.addThreat<1>(*this, pov, curThreat);
    }
}

void Accumulator::updateSelf(Accumulator& accIn){
    if(threatrefresh || threatfullupdate){
        globnnue.calcThreats(*this, side, bitboards);
        if(threatfullupdate)
            globnnue.calcThreats(*this, !side, bitboards);
        else
            applythreatsUpdates(!side);
    }else{
        applythreatsUpdates(WHITE);
        applythreatsUpdates(BLACK);
    }
    if(pstrefresh){
        globnnue.initAcc(*this, side);
        ubyte pos[10];
        for(int c=0; c<2; c++)
            for(int piece=0; piece<nbPieces; piece++){
                int nbp = places(bitboards[c][piece], pos);
                for(int i=0; i<nbp; i++)
                    globnnue.change1<1>(*this, side, Index(pos[i], piece, c).mirror(Kside[side]).changepov(side), idInputBucket[side]);
            }
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
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        if constexpr (f == 1) {
            accs[pov+2][i  ] = simd16_add(accs[pov+2][i  ], simd8_16l(threatWeights[index][i/2]));
            accs[pov+2][i+1] = simd16_add(accs[pov+2][i+1], simd8_16h(threatWeights[index][i/2]));
        } else {
            accs[pov+2][i  ] = simd16_sub(accs[pov+2][i  ], simd8_16l(threatWeights[index][i/2]));
            accs[pov+2][i+1] = simd16_sub(accs[pov+2][i+1], simd8_16h(threatWeights[index][i/2]));
        }
    }
}

void NNUE::calcThreats(Accumulator& accs, bool pov, const big bitboards[2][6]) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[pov+2][i] = simd16_zero();
    }
    big blackbb = 0;
    big whitebb = 0;
    for(int p=0; p<nbPieces; p++)
        whitebb |= bitboards[WHITE][p];
    for(int p=0; p<nbPieces; p++)
        blackbb |= bitboards[BLACK][p];
    const big occupied = whitebb | blackbb;
    bool mirror = col(__builtin_ctzll(bitboards[pov][KING])) <= 3;
    for(int idPiece=0; type(idPiece)<nbPieces-1; idPiece++){
        big mask = bitboards[color(idPiece)][type(idPiece)];
        big authMask = 0;
        for(int _c=0; _c<2; _c++)
            for(int p=0; p<nbPieces; p++)
                if(p != type(idPiece) && piecesThreat[type(idPiece)][p] != -1)
                    authMask |= bitboards[_c][p];
        big semiexcluded = 0;
        if(type(idPiece) == PAWN)
            authMask |= bitboards[color(idPiece)][type(idPiece)];
        else
            semiexcluded = bitboards[color(idPiece)][type(idPiece)];
        semiexcluded |= bitboards[color(idPiece) ^ 1][type(idPiece)];

        while(mask){
            int pos = __builtin_ctzll(mask);
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
                const big maskPiece = 1ULL << _posdef;
                int piece = -1;
                const int colorPiece = (whitebb&maskPiece)?WHITE:BLACK;
                for(int x=0; x<nbPieces; x++)
                    if(bitboards[colorPiece][x]&maskPiece){
                        piece = x;
                        break;
                    }
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
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accOut[pov][i] = simd16_add(accIn[pov][i], hlWeights[idInputBucket][index][i]);
        } else {
            accOut[pov][i] = simd16_sub(accIn[pov][i], hlWeights[idInputBucket][index][i]);
        }
    }
}
void NNUE::move3(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[idInputBucket][indexto][i], simd16_add(hlWeights[idInputBucket][indexfrom][i], hlWeights[idInputBucket][indexcap][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}
void NNUE::move2(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[idInputBucket][indexto][i], hlWeights[idInputBucket][indexfrom][i]);
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}
void NNUE::move4(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2, int idInputBucket) const{
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(simd16_add(hlWeights[idInputBucket][indexto1][i], hlWeights[idInputBucket][indexto2][i]), simd16_add(hlWeights[idInputBucket][indexfrom1][i], hlWeights[idInputBucket][indexfrom2][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}

void NNUE::updateStack(Accumulator* stack, int stackIndex) const{
    int startUpdate;
    for(startUpdate=stackIndex; startUpdate >= 1 && stack[startUpdate].update.dirty; startUpdate--);
    startUpdate++;
    for(int i=startUpdate; i<=stackIndex; i++){
        stack[i].updateSelf(stack[i-1]);
    }
}

template void NNUE::change1<-1>(Accumulator&, bool, int, int) const;
template void NNUE::change1<1>(Accumulator&, bool, int, int) const;
template void NNUE::change2<-1>(Accumulator&, Accumulator&, bool, int, int) const;
template void NNUE::change2<1>(Accumulator&, Accumulator&, bool, int, int) const;
