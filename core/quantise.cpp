#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <immintrin.h>
#include <array>
#include "simd_definitions.hpp"
#include "NNUE.hpp"
using namespace std;

const float ftClip = 1.98F;

const int OB=BUCKET;
const int IB=nbInputBuckets;

const bool isPW=true;
const bool isFactorised=true;
const char zero = 0;// for padding
const float scaler = (float)QA/(1 << L1shift);

template<typename T, int Q>
T _quantise(float w){
    w = clamp(w, -ftClip, ftClip);
    w = round(w * static_cast<float>(Q));
    if(is_same_v<T, int32_t>){
        return static_cast<T>(w);
    }
    return static_cast<T>(clamp<float>(w, -numeric_limits<T>::max(), numeric_limits<T>::max()));
}

constexpr int simdSize = sizeof(_simd)/sizeof(int16_t);
constexpr int mask = simdSize*2-1;
alignas(64) constexpr array<int16_t, simdSize> first = []{
    array<int16_t, simdSize> res{};
    for(int i=0; i<simdSize; i++){
        res[i] = i;
    }
    return res;
}();
alignas(64) constexpr array<int16_t, simdSize> second = []{
    array<int16_t, simdSize> res{};
    for(int i=0; i<simdSize; i++){
        res[i] = i+simdSize;
    }
    return res;
}();

const _simd _packed = ADDMM(packus_epi16)(*(_simd*)&first, *(_simd*)&second);
const int8_t* packed = (int8_t*)&_packed;

int transpose(int n){
    return (n&~mask) | packed[n&mask];
}

template<int input, int output>
struct layer{
    float weights[input][OB*output];
    float bias[OB*output];
    template<typename T1, bool isL1, bool isLast>
    void quantise(int id, FILE* file){
        if constexpr(isLast){ // L3
            for(int o=0; o<output; o++){
                for(int i=0; i<input; i++){
                    T1 quantised = _quantise<T1, QC>(weights[i][output*id+o]);
                    fwrite(&quantised, sizeof(T1), 1, file);
                }
            }
        }else if constexpr(isL1){ // L1
            for(int i=0; i<input/I8inI32; i++){
                for(int o=0; o<output; o++){
                    for(int k=0; k<I8inI32; k++){
                        T1 quantised = _quantise<T1, QB>(weights[i*I8inI32+k][output*id+o]/(scaler*scaler));
                        fwrite(&quantised, sizeof(T1), 1, file);
                    }
                }
            }
        }else{ // L2
            for(int i=0; i<input; i++){
                for(int o=0; o<output; o++){
                    T1 quantised = _quantise<T1, QC>(weights[i][output*id+o]);
                    fwrite(&quantised, sizeof(T1), 1, file);
                }
            }
        }
    }

    template<typename T2, int Qbias>
    void quantise_biases(int id, FILE* file){
        for(int i=0; i<output; i++){
            T2 quantised = _quantise<T2, Qbias>(bias[output*id+i]);
            fwrite(&quantised, sizeof(T2), 1, file);
        }
    }
};

void padd(FILE* file){
    while(ftell(file)%64 != 0)fwrite(&zero, 1, 1, file);
}

struct inputlayer{
    float threatweights[THREAT_SIZE][L1];
    float psqweights[IB+isFactorised][INPUT_SIZE][L1];
    float biases[L1];
    void quantise(FILE* file){
        for(int i=0; i<IB; i++)for(int j=0; j<INPUT_SIZE; j++)for(int k=0; k<L1; k++){
            float param = psqweights[i+isFactorised][j][transpose(k)];
            if constexpr(isFactorised){
                param += psqweights[0][j][transpose(k)];
            }
            int16_t quantised = _quantise<int16_t, QA>(param);
            fwrite(&quantised, sizeof(int16_t), 1, file);
        }
        padd(file);
        for(int i=0; i<THREAT_SIZE; i++)for(int k=0; k<L1; k++){
            int8_t quantised = _quantise<int8_t, QA>(threatweights[i][transpose(k)]);
            fwrite(&quantised, sizeof(int8_t), 1, file);
        }
        padd(file);
        for(int k=0; k<L1; k++){
            int16_t quantised = _quantise<int16_t, QA>(biases[transpose(k)]);
            fwrite(&quantised, sizeof(int16_t), 1, file);
        }
        padd(file);
    }
};

struct nn{
    inputlayer FT;
    layer<L1*(2-isPW), L2> l1;
    layer<L2, L3> l2;
    layer<L3, 1> l3;
};

int main(int argc, char** argv){
    FILE* fin=fopen(argv[1], "r");
    FILE* fout=fopen(argv[2], "w");
    unique_ptr<nn> nnue = make_unique<nn>();
    fread(&nnue->FT, sizeof(nnue->FT), 1, fin);
    fread(&nnue->l1, sizeof(nnue->l1), 1, fin);
    fread(&nnue->l2, sizeof(nnue->l2), 1, fin);
    fread(&nnue->l3, sizeof(nnue->l3), 1, fin);
    fclose(fin);
    nnue->FT.quantise(fout);
    for(int id=0; id<OB; id++){
        nnue->l1.quantise<int8_t, true, false>(id, fout);
        padd(fout);
        nnue->l1.quantise_biases<int32_t, QC << L1shift>(id, fout);
        padd(fout);
        nnue->l2.quantise<int32_t, false, false>(id, fout);
        padd(fout);
        nnue->l2.quantise_biases<int32_t, QC*QC*QC>(id, fout);
        padd(fout);
        nnue->l3.quantise<int32_t, false, true>(id, fout);
        padd(fout);
        nnue->l3.quantise_biases<int32_t, QC*QC*QC*QC>(id, fout);
        padd(fout);
    }
    fclose(fout);
}