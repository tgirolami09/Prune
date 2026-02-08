#include "NNUE.hpp"
#include "Const.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include "GameState.hpp"
#include "embeder.hpp"
#include "simd_definitions.hpp"

using namespace std;

int turn(int index){
    return ((index^56)+384)%768;
}

Index::Index():square(0), piece(6), color(false){}
Index::Index(int _square, int _piece, bool _color):square(_square), piece(_piece), color(_color){}
void Index::smirror(bool needs){
    square ^= 7*needs;
}
Index Index::mirror(bool needs){
    return Index(square^(7*needs), piece, color);
}
void Index::schangepov(){
    square ^= 56;
    color ^= 1;
}
Index Index::changepov(){
    return Index(square^56, piece, !color);
}
Index Index::changepov(bool needs){
    if(needs)
        return Index(square^56, piece, !color);
    else
        return *this;
}
Index::operator int(){
    return ((6*color+piece)<<6)|(square^7);
}
bool Index::isnull(){
    return piece >= 6;
}

updateBuffer::updateBuffer():dirty(true){}
updateBuffer::updateBuffer(Index _add1, Index _add2, Index _sub1, Index _sub2):dirty(true){
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

void Accumulator::reinit(const GameState* state, Accumulator& prevAcc, bool side, bool mirror, Index sub1, Index add1, Index sub2, Index add2){
    Kside[0] = prevAcc.Kside[0];
    Kside[1] = prevAcc.Kside[1];
    if(mirror){
        memcpy(bitboards, state->boardRepresentation, sizeof(bitboards));
        update = updateBuffer();
        Kside[side] ^= 1;
    }else{
        update = updateBuffer(add1, add2, sub1, sub2);
    }
    mustmirror = mirror;
}

void Accumulator::updateSelf(Accumulator& accIn){
    if(mustmirror){
        globnnue.initAcc(*this);
        ubyte pos[10];
        for(int c=0; c<2; c++)
            for(int piece=0; piece<nbPieces; piece++){
                int nbp = places(bitboards[c][piece], pos);
                for(int i=0; i<nbp; i++)
                    for(int pov=0; pov<2; pov++)
                        globnnue.change1<1>(*this, pov, Index(pos[i], piece, c).mirror(Kside[pov]).changepov(pov));
            }
        update.dirty = false;
        return;
    }
    if(update.type == 0){
        globnnue.move2(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]));
        globnnue.move2(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]));
    }else if(update.type == 1){
        globnnue.move3(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]), update.sub2[0].mirror(Kside[WHITE]));
        globnnue.move3(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]), update.sub2[1].mirror(Kside[BLACK]));
    }else{
        globnnue.move4(WHITE, accIn, *this, update.sub1[0].mirror(Kside[WHITE]), update.add1[0].mirror(Kside[WHITE]), update.sub2[0].mirror(Kside[WHITE]), update.add2[0].mirror(Kside[WHITE]));
        globnnue.move4(BLACK, accIn, *this, update.sub1[1].mirror(Kside[BLACK]), update.add1[1].mirror(Kside[BLACK]), update.sub2[1].mirror(Kside[BLACK]), update.add2[1].mirror(Kside[BLACK]));
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
        for(int i=0; i<INPUT_SIZE; i++) {
            for(int j=0; j<HL_SIZE/nb16; j++) {
                hlWeights[i][j] = simd16_zero();
                for(int k=0; k<nb16; k++) {
                    set_simd16_element(hlWeights[i][j], k, genRandom(state)%256-128);
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

void NNUE::initAcc(Accumulator& accs){
    for(int i=0; i<HL_SIZE/nb16; i++){
        accs[WHITE][i] = hlBiases[i];
        accs[BLACK][i] = hlBiases[i];
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

dbyte NNUE::eval(const Accumulator& accs, bool side, int idB) const{
    simdint res = simdint_zero();
    for(int i=0; i<HL_SIZE/nb16; i++){
        res = simdint_add(res, doOut(accs[side^1][i], outWeights[idB][1][i]));
        res = simdint_add(res, doOut(accs[side][i], outWeights[idB][0][i]));
    }
    int finRes = mysum(res);
    finRes /= QA;
    finRes += outbias[idB];
    finRes = finRes*SCALE/(QA*QB);
    return finRes;
}
template<int f>
void NNUE::change1(Accumulator& accs, bool pov, int index){
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accs[pov][i] = simd16_add(accs[pov][i], hlWeights[index][i]);
        } else {
            accs[pov][i] = simd16_sub(accs[pov][i], hlWeights[index][i]);
        }
    }
}
template<int f>
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, bool pov, int index){
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accOut[pov][i] = simd16_add(accIn[pov][i], hlWeights[index][i]);
        } else {
            accOut[pov][i] = simd16_sub(accIn[pov][i], hlWeights[index][i]);
        }
    }
}
void NNUE::move3(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto, int indexcap){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[indexto][i], simd16_add(hlWeights[indexfrom][i], hlWeights[indexcap][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}
void NNUE::move2(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom, int indexto){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(hlWeights[indexto][i], hlWeights[indexfrom][i]);
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}
void NNUE::move4(int color, Accumulator& accIn, Accumulator& accOut, int indexfrom1, int indexto1, int indexfrom2, int indexto2){
    for(int i=0; i<HL_SIZE/nb16; i++){
        simd16 update = simd16_sub(simd16_add(hlWeights[indexto1][i], hlWeights[indexto2][i]), simd16_add(hlWeights[indexfrom1][i], hlWeights[indexfrom2][i]));
        accOut[color][i] = simd16_add(accIn[color][i], update);
    }
}

void NNUE::updateStack(Accumulator* stack, int stackIndex){
    int startUpdate;
    for(startUpdate=stackIndex; startUpdate >= 1 && stack[startUpdate].update.dirty; startUpdate--);
    startUpdate++;
    for(int i=startUpdate; i<=stackIndex; i++){
        stack[i].updateSelf(stack[i-1]);
    }
}

template void NNUE::change1<-1>(Accumulator&, bool, int);
template void NNUE::change1<1>(Accumulator&, bool, int);
template void NNUE::change2<-1>(Accumulator&, Accumulator&, bool, int);
template void NNUE::change2<1>(Accumulator&, Accumulator&, bool, int);

NNUE globnnue = NNUE();
