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
updateBuffer::updateBuffer(){}
updateBuffer::updateBuffer(int _add1, int _add2, int _sub1, int _sub2):dirty(true), type(2){
    add1[0] = _add1;
    add1[1] = turn(_add1);
    add2[0] = _add2;
    add2[1] = turn(_add2);
    sub1[0] = _sub1;
    sub1[1] = turn(_sub1);
    sub2[0] = _sub2;
    sub2[1] = turn(_sub2);
}

updateBuffer::updateBuffer(int _add1, int _sub1):dirty(true), type(0){
    add1[0] = _add1;
    add1[1] = turn(_add1);
    add2[0] = -1;
    add2[1] = -1;
    sub1[0] = _sub1;
    sub1[1] = turn(_sub1);
    sub2[0] = -1;
    sub2[1] = -1;
}
updateBuffer::updateBuffer(int _add1, int _sub1, int _sub2):dirty(true), type(1){
    add1[0] = _add1;
    add1[1] = turn(_add1);
    add2[0] = -1;
    add2[1] = -1;
    sub1[0] = _sub1;
    sub1[1] = turn(_sub1);
    sub2[0] = _sub2;
    sub2[1] = turn(_sub2);
}

void Accumulator::reinit(const GameState* state, Accumulator& prevAcc, bool side, bool mirror, int sub1, int add1, int sub2, int add2){
    if(mirror){
        memcpy(bitboards, state->boardRepresentation, sizeof(bitboards));
        update = updateBuffer();
    }else if(sub2 == -1)
        update = updateBuffer(add1, sub1);
    else if(add2 == -1)
        update = updateBuffer(add1, sub1, sub2);
    else
        update = updateBuffer(add1, add2, sub1, sub2);
    mustmirror = mirror;
    Kside[0] = prevAcc.Kside[0];
    Kside[1] = prevAcc.Kside[1];
    Kside[side] ^= mirror;
}

int mirrorSquare(int square, bool mirror){
    return mirror?square^7:square;
}

void Accumulator::updateSelf(Accumulator& accIn){
    if(mustmirror){
        globnnue.initAcc(*this);
        ubyte pos[8];
        for(int C=0; C<2; C++){
            const int povChange = C?56:0;
            for(int c=0; c<2; c++)
                for(int piece=0; piece<nbPieces; piece++){
                    int nbp = places(bitboards[c][piece], pos);
                    for(int i=0; i<nbp; i++)
                        globnnue.change1<1>(*this, C, piece, c^C, mirrorSquare(pos[i]^povChange, Kside[C]));
                }
        }
        return;
    }
    assert(update.sub1[0] != update.sub1[1]);
    if(update.type == 0){
        globnnue.move2(WHITE, accIn, *this, update.sub1[0], update.add1[0]);
        globnnue.move2(BLACK, accIn, *this, update.sub1[1], update.add1[1]);
    }else if(update.type == 1){
        globnnue.move3(WHITE, accIn, *this, update.sub1[0], update.add1[0], update.sub2[0]);
        globnnue.move3(BLACK, accIn, *this, update.sub1[1], update.add1[1], update.sub2[1]);
    }else{
        globnnue.move4(WHITE, accIn, *this, update.sub1[0], update.add1[0], update.sub2[0], update.add2[0]);
        globnnue.move4(BLACK, accIn, *this, update.sub1[1], update.add1[1], update.sub2[1], update.add2[1]);
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
void NNUE::change2(Accumulator& accs, int piece, int c, int square){
    int index = get_index(piece, c, square);
    int index2 = turn(index); // point of view change
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accs[WHITE][i] = simd16_add(accs[WHITE][i], hlWeights[index][i]);
            accs[BLACK][i] = simd16_add(accs[BLACK][i], hlWeights[index2][i]);
        } else {
            accs[WHITE][i] = simd16_sub(accs[WHITE][i], hlWeights[index][i]);
            accs[BLACK][i] = simd16_sub(accs[BLACK][i], hlWeights[index2][i]);
        }
    }
}
template<int f>
void NNUE::change1(Accumulator& accs, bool C, int piece, int c, int square){
    int index = get_index(piece, c, square);
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accs[C][i] = simd16_add(accs[C][i], hlWeights[index][i]);
        } else {
            accs[C][i] = simd16_sub(accs[C][i], hlWeights[index][i]);
        }
    }
}
template<int f>
void NNUE::change2(Accumulator& accIn, Accumulator& accOut, bool C, int piece, int c, int square){
    int index = get_index(piece, c, square);
    for(int i=0; i<HL_SIZE/nb16; i++){
        if constexpr (f == 1) {
            accOut[C][i] = simd16_add(accIn[C][i], hlWeights[index][i]);
        } else {
            accOut[C][i] = simd16_sub(accIn[C][i], hlWeights[index][i]);
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

template void NNUE::change2<-1>(Accumulator&, int, int, int);
template void NNUE::change2<1>(Accumulator&, int, int, int);
template void NNUE::change1<-1>(Accumulator&, bool, int, int, int);
template void NNUE::change1<1>(Accumulator&, bool, int, int, int);
template void NNUE::change2<-1>(Accumulator&, Accumulator&, bool, int, int, int);
template void NNUE::change2<1>(Accumulator&, Accumulator&, bool, int, int, int);

NNUE globnnue = NNUE();
