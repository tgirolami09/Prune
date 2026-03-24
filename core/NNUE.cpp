#include "NNUE.hpp"
#include "Const.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include <utility>
#include "Functions.hpp"
#include "LegalMoveGenerator.hpp"
#include "simd_definitions.hpp"

using namespace std;

#ifdef DEBUG_MACRO
StatVar<sbig, 64, 0> TIupdateRemStat;
StatVar<sbig, 64, 0> TIupdateAddStat;
StatVar<sbig, 64, 0> TIupdateTotStat;
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
    big mask = fullDir[square][square2]&occupancy;
    if(!mask)return 0;
    if(square2 > square)
        return mask&-mask;
    else
        return 1ULL << (__builtin_clzll(mask)^63);
}

template<bool enPassant, bool tworemove>
void Accumulator::updateXrays(int pos, bool remove, int removepos, int removepos2){
    const big queenbb = bitboards[WHITE][QUEEN] | bitboards[BLACK][QUEEN];
    const big kingsbb = bitboards[WHITE][KING ] | bitboards[BLACK][KING ];
    big rookmask =   moves_table(pos+64, occupied&mask_empty_rook  (pos));
    if constexpr(enPassant)
        rookmask &= ~mask_row[row(pos)];
    const big bishopmask = moves_table(pos   , occupied&mask_empty_bishop(pos));
    big maskremove = (1ULL << removepos)|firstafter(removepos, pos, occupied, rookmask|bishopmask);
    if constexpr(tworemove){
        maskremove |= (1ULL << removepos2)|firstafter(removepos2, pos, occupied, rookmask|bishopmask);
    }
    const big maskFkings = kingsbb|
        firstafter(__builtin_ctzll(bitboards[WHITE][KING]), pos, occupied, rookmask|bishopmask)|
        firstafter(__builtin_ctzll(bitboards[BLACK][KING]), pos, occupied, rookmask|bishopmask);
    const big filterout = ~(maskremove|maskFkings);
    #pragma GCC unroll 2
    for(const auto& [simppiece, maskPos]:{
        make_pair(ROOK  , rookmask),
        make_pair(BISHOP, bishopmask)
    }){
        const big maskPiece = bitboards[WHITE][simppiece] | bitboards[BLACK][simppiece] | queenbb;
        big mask = maskPos & maskPiece & filterout; //only cares about the pieces
        while(mask){
            const int posatk = __builtin_ctzll(mask);
            const big maskdef = firstInDirection(posatk, pos, occupied);
            if(maskdef){
                const big maskatk = 1ULL << posatk;
                const int posdef = __builtin_ctzll(maskdef);
                const bool colordef = (maskdef&whitebb)?WHITE:BLACK;
                const bool coloratk = (maskatk&whitebb)?WHITE:BLACK;
                int piecedef = -1;
                for(int i=0; i<nbPieces-1; i++)
                    if(bitboards[colordef][i]&maskdef)piecedef = i;
                const int pieceatk = (maskatk&queenbb)?QUEEN:simppiece;
                const ThreatIndex threat(Index(posatk, pieceatk, coloratk), Index(posdef, piecedef, colordef));
                if(!threat.isexcluded() && !threat.issemiexcluded()){ // the non excluded threat will be in the reverse
                    update.threatUpdates[remove][update.nbThreats[remove]++] = threat;
                    mask &= ~maskdef;
                }
            }
            mask &= mask-1;
        }
    }
}

void Accumulator::updatePieceOutComing(const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos, const big sliders[3]){
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
            authMask |= bitboards[WHITE][x]|bitboards[BLACK][x];
    atkmask &= occupied&authMask;
    if(removepos != -1)atkmask &= ~(1ULL << removepos);
    while(atkmask){
        const int _posdef = __builtin_ctzll(atkmask);
        const big maskPiece = 1ULL << _posdef;
        int piecedef = -1;
        const int colorPiece = (whitebb&maskPiece)?WHITE:BLACK;
        for(int x=0; x<nbPieces-1; x++)
            if(bitboards[colorPiece][x]&maskPiece){
                piecedef = x;
                break;
            }
        Index posdef(_posdef, piecedef, colorPiece);
        ThreatIndex threat(posatk, posdef);
        update.threatUpdates[remove][update.nbThreats[remove]++] = threat;
        atkmask &= atkmask-1;
    }
}

void Accumulator::updatePieceIncoming(const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos, const big sliders[3]){
    big atkmask;
    const Index posdef(pos, piece, colorpiece);
    const big maskremove = ~(removepos == -1 ? 0 : (1ULL << removepos));
    // ------------------------ non slider pieces (PAWN & KNIGHT) ---------------------------------
    //pawns are doing separatly because of the color importance
    if(piecesThreat[PAWN][piece] != -1){
        #pragma GCC unroll 2
        for(const bool c:{WHITE, BLACK}){
            if(piece == PAWN && colorpiece != c)continue;
            atkmask = attackPawns[pos+(!c)*64];
            atkmask &= bitboards[c][PAWN]&maskremove;
            while(atkmask){
                int atkpos = __builtin_ctzll(atkmask);
                const ThreatIndex threat(Index(atkpos, PAWN, c), posdef);
                update.threatUpdates[remove][update.nbThreats[remove]++] = threat;
                atkmask &= atkmask-1;
            }
        }
    }
    if(piece != KNIGHT){
        atkmask = KnightMoves[pos]&maskremove;
        atkmask &= bitboards[WHITE][KNIGHT] | bitboards[BLACK][KNIGHT];
        while(atkmask){
            int atkpos = __builtin_ctzll(atkmask);
            const bool c=((1ULL << atkpos)&whitebb) ? WHITE : BLACK;
            const ThreatIndex threat(Index(atkpos, KNIGHT, c), posdef);
            update.threatUpdates[remove][update.nbThreats[remove]++] = threat;
            atkmask &= atkmask-1;
        }
    }
    // -------------------------- slider pieces (ROOK & BISHOP & QUEEN) ---------------------------
    if(piece != QUEEN){ //otherwise all are either excluded or already added by outcoming
        //they are doing separatly because of bishopatk/rookatk
        const big bishopatk = maskremove & sliders[0];
        const big rookatk   = maskremove & sliders[1];
        #pragma GCC unroll 3
        for(const int atkpiece:{BISHOP, ROOK, QUEEN}){
            if(atkpiece != piece){
                atkmask = (bishopatk*(atkpiece != ROOK))|(rookatk*(atkpiece != BISHOP));
                atkmask &= (bitboards[WHITE][atkpiece]|bitboards[BLACK][atkpiece]);
                while(atkmask){
                    int atkpos = __builtin_ctzll(atkmask);
                    const bool c=((1ULL << atkpos)&whitebb) ? WHITE : BLACK;
                    const ThreatIndex threat(Index(atkpos, atkpiece, c), posdef);
                    update.threatUpdates[remove][update.nbThreats[remove]++] = threat;
                    atkmask &= atkmask-1;
                }
            }
        }
    }
}

void Accumulator::updatePiece(const int piece, const bool colorpiece, const int pos, const bool remove, const int removepos){
    big sliders[3] = {
        moves_table(pos   , occupied&mask_empty_bishop(pos)),
        moves_table(pos+64, occupied&mask_empty_rook  (pos)),
    };
    sliders[2] = sliders[0]|sliders[1];
    updatePieceIncoming(piece, colorpiece, pos, remove, removepos, sliders);
    updatePieceOutComing(piece, colorpiece, pos, remove, removepos, sliders);
}

void Accumulator::getThreatUpdates(const big state1[2][6], const big state2[2][6], const Move& move){
    const int toPiece = move.promotion() == -1 ? move.piece : move.promotion();
    const bool isCapture = move.capture != -2;
    const int capture = move.capture;
    if(move.capture == -1){//en passant
        defstaterelated(state1);
        const int enpassantpos = move.to()+((side == WHITE)?-8:8);
        updatePiece(PAWN, side, move.from(), true, -1);
        updateXrays<false, true>(move.to(), true, move.from(), enpassantpos);
        updatePiece(PAWN, !side, enpassantpos, true, move.from());
        defstaterelated(state2);
        updatePiece(PAWN, side, move.to(), false, -1);
        updateXrays(move.from(), false, move.to());
        updateXrays<true>(enpassantpos, false, move.to()); //remove the common rook side ray
    }else if(move.isCastling()){
        defstaterelated(state1);
        updatePiece(ROOK, side, update.sub2[0].square, true, -1);
        defstaterelated(state2);
        updatePiece(ROOK, side, update.add2[0].square, false, -1);
    }else{
        defstaterelated(state1);
        //first remove the threat that will disappear because of the move
        if(move.piece != KING)
            updatePiece(move.piece, side, move.from(), true, -1);
        if(isCapture){
            updatePiece(capture, !side, move.to(), true, move.from()); // threat including move.from has already been removed
        }else
            updateXrays(move.to(), true, move.from()); // threat including move.from has already been removed
        //then add the new threats
        defstaterelated(state2);
        if(move.piece != KING)
            updatePiece(toPiece, side, move.to(), false, -1);
        updateXrays(move.from(), false, move.to()); // threat including move.to() has already been added by addPiece
    }
}

void Accumulator::defstaterelated(const big state[2][6]){
    memcpy(bitboards, state, sizeof(bitboards));
    blackbb = 0;
    whitebb = 0;
    for(int p=0; p<nbPieces; p++)
        whitebb |= bitboards[WHITE][p];
    for(int p=0; p<nbPieces; p++)
        blackbb |= bitboards[BLACK][p];
    occupied = whitebb | blackbb;
}

void Accumulator::reinit(const Move& move, const big state1[2][6], const big state2[2][6], Accumulator& prevAcc, bool _side, bool mirror, Index sub1, Index add1, Index sub2, Index add2){
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
    int maxi = update.nbThreats[0] < update.nbThreats[1];
    if(!update.nbThreats[maxi^1]){
        ThreatIndex _curThreat = update.threatUpdates[maxi][0].changepov(pov).mirror(Kside[pov]);
        if(_curThreat.issemiexcluded())_curThreat.swap();
        const int _threatind = (int)_curThreat;
        if(maxi)
            globnnue.addThreat<-1>(accIn, *this, pov, _threatind);
        else
            globnnue.addThreat< 1>(accIn, *this, pov, _threatind);
    }else{
        if(update.nbThreats[maxi^1] >= 2){
            {
            ThreatIndex threatadd1 = update.threatUpdates[0][0].changepov(pov).mirror(Kside[pov]);
            ThreatIndex threatrem1 = update.threatUpdates[1][0].changepov(pov).mirror(Kside[pov]);
            if(threatadd1.issemiexcluded())threatadd1.swap();
            if(threatrem1.issemiexcluded())threatrem1.swap();
            ThreatIndex threatadd2 = update.threatUpdates[0][1].changepov(pov).mirror(Kside[pov]);
            ThreatIndex threatrem2 = update.threatUpdates[1][1].changepov(pov).mirror(Kside[pov]);
            if(threatadd2.issemiexcluded())threatadd2.swap();
            if(threatrem2.issemiexcluded())threatrem2.swap();
            globnnue.add2Threataddsub(accIn, *this, pov, threatadd1, threatrem1, threatadd2, threatrem2);
            }
            int i;
            for(i=2; i<update.nbThreats[maxi^1]-1; i+=2){
                ThreatIndex threatadd1 = update.threatUpdates[0][i].changepov(pov).mirror(Kside[pov]);
                ThreatIndex threatrem1 = update.threatUpdates[1][i].changepov(pov).mirror(Kside[pov]);
                if(threatadd1.issemiexcluded())threatadd1.swap();
                if(threatrem1.issemiexcluded())threatrem1.swap();
                ThreatIndex threatadd2 = update.threatUpdates[0][i+1].changepov(pov).mirror(Kside[pov]);
                ThreatIndex threatrem2 = update.threatUpdates[1][i+1].changepov(pov).mirror(Kside[pov]);
                if(threatadd2.issemiexcluded())threatadd2.swap();
                if(threatrem2.issemiexcluded())threatrem2.swap();
                globnnue.add2Threataddsub(*this, pov, threatadd1, threatrem1, threatadd2, threatrem2);
            }
            if(i < update.nbThreats[maxi^1]){
                ThreatIndex threatadd = update.threatUpdates[0][i].changepov(pov).mirror(Kside[pov]);
                ThreatIndex threatrem = update.threatUpdates[1][i].changepov(pov).mirror(Kside[pov]);
                if(threatrem.issemiexcluded())threatrem.swap();
                if(threatadd.issemiexcluded())threatadd.swap();
                globnnue.addThreataddsub(*this, pov, threatadd, threatrem);
            }
        }else{
            ThreatIndex threatadd = update.threatUpdates[0][0].changepov(pov).mirror(Kside[pov]);
            ThreatIndex threatrem = update.threatUpdates[1][0].changepov(pov).mirror(Kside[pov]);
            if(threatrem.issemiexcluded())threatrem.swap();
            if(threatadd.issemiexcluded())threatadd.swap();
            globnnue.addThreataddsub(accIn, *this, pov, threatadd, threatrem);
        }
    }
    for(int i=max(update.nbThreats[maxi^1], 1); i<update.nbThreats[maxi]; i++){
        ThreatIndex curThreat = update.threatUpdates[maxi][i].changepov(pov).mirror(Kside[pov]);
        if(curThreat.issemiexcluded())curThreat.swap();
        if(maxi)
            globnnue.addThreat<-1>(*this, pov, curThreat);
        else
            globnnue.addThreat< 1>(*this, pov, curThreat);
    }
}

void Accumulator::updateSelf(const Accumulator& accIn, FinnyTables& finny){
#ifdef DEBUG_MACRO
    TIupdateAddStat.update(update.nbThreats[0]);
    TIupdateRemStat.update(update.nbThreats[1]);
    TIupdateTotStat.update(update.nbThreats[0]+update.nbThreats[1]);
#endif
    if(threatrefresh){
        globnnue.calcThreats(*this, side, bitboards);
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
                const big common = bitboards[c][piece]&finny.normals[index].bitboards[c][piece];
                big maskadd = bitboards[c][piece]&~common;
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
        memcpy(finny.normals[index].bitboards, bitboards, sizeof(bitboards));
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
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        simd16 low=simd8_16l(threatWeights[index][i/2]);
        simd16 high=simd8_16h(threatWeights[index][i/2]);
        if constexpr (f == 1) {
            accs[pov+2][i  ] = simd16_add(accs[pov+2][i  ], low);
            accs[pov+2][i+1] = simd16_add(accs[pov+2][i+1], high);
        } else {
            accs[pov+2][i  ] = simd16_sub(accs[pov+2][i  ], low);
            accs[pov+2][i+1] = simd16_sub(accs[pov+2][i+1], high);
        }
    }
}

void NNUE::addThreataddsub(Accumulator& accs, bool pov, int indexadd, int indexrem) const{
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        simd16 lowa=simd8_16l(threatWeights[indexadd][i/2]);
        simd16 higha=simd8_16h(threatWeights[indexadd][i/2]);
        simd16 lowr=simd8_16l(threatWeights[indexrem][i/2]);
        simd16 highr=simd8_16h(threatWeights[indexrem][i/2]);
        simd16 low = simd16_sub(lowa, lowr);
        simd16 high = simd16_sub(higha, highr);
        accs[pov+2][i  ] = simd16_add(accs[pov+2][i  ], low);
        accs[pov+2][i+1] = simd16_add(accs[pov+2][i+1], high);
    }
}


void NNUE::addThreataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, int indexadd, int indexrem) const{
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        simd16 lowa=simd8_16l(threatWeights[indexadd][i/2]);
        simd16 higha=simd8_16h(threatWeights[indexadd][i/2]);
        simd16 lowr=simd8_16l(threatWeights[indexrem][i/2]);
        simd16 highr=simd8_16h(threatWeights[indexrem][i/2]);
        simd16 low = simd16_sub(lowa, lowr);
        simd16 high = simd16_sub(higha, highr);
        accs[pov+2][i  ] = simd16_add(accIn[pov+2][i  ], low);
        accs[pov+2][i+1] = simd16_add(accIn[pov+2][i+1], high);
    }
}

void NNUE::add2Threataddsub(Accumulator& accs, bool pov, int indexadd1, int indexrem1, int indexadd2, int indexrem2) const{
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        simd16 lowa1=simd8_16l(threatWeights[indexadd1][i/2]);
        simd16 higha1=simd8_16h(threatWeights[indexadd1][i/2]);
        simd16 lowr1=simd8_16l(threatWeights[indexrem1][i/2]);
        simd16 highr1=simd8_16h(threatWeights[indexrem1][i/2]);
        simd16 lowa2=simd8_16l(threatWeights[indexadd2][i/2]);
        simd16 higha2=simd8_16h(threatWeights[indexadd2][i/2]);
        simd16 lowr2=simd8_16l(threatWeights[indexrem2][i/2]);
        simd16 highr2=simd8_16h(threatWeights[indexrem2][i/2]);
        simd16 low1 = simd16_sub(lowa1, lowr1);
        simd16 high1 = simd16_sub(higha1, highr1);
        simd16 low2 = simd16_sub(lowa2, lowr2);
        simd16 high2 = simd16_sub(higha2, highr2);
        simd16 low = simd16_add(low1, low2);
        simd16 high = simd16_add(high1, high2);
        accs[pov+2][i  ] = simd16_add(accs[pov+2][i  ], low);
        accs[pov+2][i+1] = simd16_add(accs[pov+2][i+1], high);
    }
}


void NNUE::add2Threataddsub(const Accumulator& accIn, Accumulator& accs, bool pov, int indexadd1, int indexrem1, int indexadd2, int indexrem2) const{
    for(int i=0; i<HL_SIZE*2/nb8; i += 2){
        simd16 lowa1=simd8_16l(threatWeights[indexadd1][i/2]);
        simd16 higha1=simd8_16h(threatWeights[indexadd1][i/2]);
        simd16 lowr1=simd8_16l(threatWeights[indexrem1][i/2]);
        simd16 highr1=simd8_16h(threatWeights[indexrem1][i/2]);
        simd16 lowa2=simd8_16l(threatWeights[indexadd2][i/2]);
        simd16 higha2=simd8_16h(threatWeights[indexadd2][i/2]);
        simd16 lowr2=simd8_16l(threatWeights[indexrem2][i/2]);
        simd16 highr2=simd8_16h(threatWeights[indexrem2][i/2]);
        simd16 low1 = simd16_sub(lowa1, lowr1);
        simd16 high1 = simd16_sub(higha1, highr1);
        simd16 low2 = simd16_sub(lowa2, lowr2);
        simd16 high2 = simd16_sub(higha2, highr2);
        simd16 low = simd16_add(low1, low2);
        simd16 high = simd16_add(high1, high2);
        accs[pov+2][i  ] = simd16_add(accIn[pov+2][i  ], low);
        accs[pov+2][i+1] = simd16_add(accIn[pov+2][i+1], high);
    }
}


template<int f>
void NNUE::addThreat(const Accumulator& accIn, Accumulator& accOut, bool pov, int index) const{
    static_assert(f == 1 || f == -1, "f should be either 1 or -1");
    for(int i=0; i<HL_SIZE/nb16; i += 2){
        simd16 low=simd8_16l(threatWeights[index][i/2]);
        simd16 high=simd8_16h(threatWeights[index][i/2]);
        if constexpr (f == 1) {
            accOut[pov+2][i  ] = simd16_add(accIn[pov+2][i  ], low);
            accOut[pov+2][i+1] = simd16_add(accIn[pov+2][i+1], high);
        } else {
            accOut[pov+2][i  ] = simd16_sub(accIn[pov+2][i  ], low);
            accOut[pov+2][i+1] = simd16_sub(accIn[pov+2][i+1], high);
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
        for(int p=0; p<nbPieces; p++)
            if(p != type(idPiece) && piecesThreat[type(idPiece)][p] != -1)
                for(int _c=0; _c<2; _c++)
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
