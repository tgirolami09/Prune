#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
using namespace std;

const int psqSize = 768;
const int threatSize=60144;

const int QA = 255;
const int QB = 64;
const float ftClip = 1.98F;

const int OB=8;
const int IB=4;

const int L1=384;
const int L2=16;
const int L3=32;

const bool isPW=true;

template<typename T, int Q>
T _quantise(float w){
    w = clamp(w, -ftClip, ftClip);
    return clamp<int>(round(w*Q), -numeric_limits<T>::max(), numeric_limits<T>::max());
}

template<int input, int output, int ob>
struct layer{
    float weights[input][ob*output];
    float bias[ob*output];
    template<typename T1, typename T2, int Qbias, bool transpose=true>
    void quantise(int id, FILE* file){
        if(transpose){
            for(int i=0; i<output; i++){
                for(int j=0; j<input; j++){
                    T1 quantised = _quantise<T1, QB>(weights[j][i+output*id]);
                    fwrite(&quantised, sizeof(T1), 1, file);
                }
            }
        }else{
            for(int j=0; j<input; j++){
                for(int i=0; i<output; i++){
                    T1 quantised = _quantise<T1, QB>(weights[j][i+output*id]);
                    fwrite(&quantised, sizeof(T1), 1, file);
                }
            }
        }
        for(int i=0; i<output; i++){
            T2 quantised = _quantise<T2, Qbias>(bias[i+output*id]);
            fwrite(&quantised, sizeof(T2), 1, file);
        }
    }
};

template<int ib, int hl>
struct inputlayer{
    float threatweights[threatSize][hl];
    float psqweights[ib][psqSize][hl];
    float biases[hl];
    void quantise(FILE* file){
        for(int i=0; i<ib; i++)for(int j=0; j<psqSize; j++)for(float w:psqweights[i][j]){
            int16_t quantised = _quantise<int16_t, QA>(w);
            fwrite(&quantised, sizeof(int16_t), 1, file);
        }
        for(int i=0; i<threatSize; i++)for(float w:threatweights[i]){
            int8_t quantised = _quantise<int8_t, QA>(w);
            fwrite(&quantised, sizeof(int8_t), 1, file);
        }
        for(float b:biases){
            int16_t quantised = _quantise<int16_t, QA>(b);
            fwrite(&quantised, sizeof(int16_t), 1, file);
        }
    }
};

struct nn{
    inputlayer<IB, L1> FT;
    layer<L1*(2-isPW), L2, OB> l1;
    layer<L2, L3, OB> l2;
    layer<L3, 1, OB> l3;
};

int main(int argc, char** argv){
    FILE* fin=fopen(argv[1], "r");
    FILE* fout=fopen(argv[2], "w");
    unique_ptr<nn> nnue = make_unique<nn>();
    fread(&nnue->FT, 1, sizeof(nnue->FT), fin);
    fread(&nnue->l1, 1, sizeof(nnue->l1), fin);
    fread(&nnue->l2, 1, sizeof(nnue->l2), fin);
    fread(&nnue->l3, 1, sizeof(nnue->l3), fin);
    fclose(fin);
    nnue->FT.quantise(fout);
    for(int id=0; id<OB; id++){
        nnue->l1.quantise<int8_t, int32_t, QB, false>(id, fout);
        nnue->l2.quantise<int32_t, int32_t, QB*QB*QB>(id, fout);
        nnue->l3.quantise<int32_t, int32_t, QB*QB*QB*QB>(id, fout);
    }
    fclose(fout);
}