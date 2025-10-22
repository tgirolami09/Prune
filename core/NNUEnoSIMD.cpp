#ifndef NNUE_CPP
#define NNUE_CPP
#include "Const.hpp"
#include "Functions.hpp"
#include <cstdint>
#include <fstream>
#include <string>

using namespace std;

const int INPUT_SIZE = 12*64;
const int HL_SIZE = 64;
const int BUCKET=8;
const int DIVISOR=(31+BUCKET)/BUCKET;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;

#define dbyte int16_t

template<int min, int max>
int SCReLU(dbyte value){
    if(value >= max)return max*max;
    else if(value <= min)return min*min;
    return value*value;
}

int activation(dbyte value){
    return SCReLU<0, QA>(value);
}

BINARY_ASM_INCLUDE("model.bin", baseModel);

class NNUE{
public:
    dbyte hlWeights[INPUT_SIZE][HL_SIZE];
    dbyte hlBiases[HL_SIZE];
    int outWeights[BUCKET][2*HL_SIZE];
    int outbias[BUCKET];
    dbyte accs[2][HL_SIZE];
    
    dbyte read_bytes(ifstream& file){
        char ret;
        file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
        return ret;
    }
    
    NNUE(string name){
        ifstream file(name);
        if(!file.is_open()){
            printf("info string file %s not found\n", name.c_str());
            return;
        }
        for(int i=0; i<INPUT_SIZE; i++)
            for(int j=0; j<HL_SIZE; j++)
                hlWeights[i][j] = read_bytes(file);
        
        for(int i=0; i<HL_SIZE; i++){
            hlBiases[i] = read_bytes(file);
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
        for(int ob=0; ob<BUCKET; ob++)
            for(int i=0; i<2*HL_SIZE; i++)
                outWeights[ob][i] = read_bytes(file);
        for(int ob=0; ob<BUCKET; ob++)
            outbias[ob] = read_bytes<dbyte>(file);
    }
    
    NNUE(){
        int pointer = 0;
        for(int i=0; i<INPUT_SIZE; i++)
            for(int j=0; j<HL_SIZE; j++)
                hlWeights[i][j] = transform(baseModel[pointer++]);
        
        for(int i=0; i<HL_SIZE; i++){
            hlBiases[i] = transform(baseModel[pointer++]);
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
        for(int ob=0; ob<BUCKET; ob++)
            for(int i=0; i<2*HL_SIZE; i++)
                outWeights[ob][i] = transform(baseModel[pointer++]);
        for(int ob=0; ob<BUCKET; ob++){
            outbias[ob] = transform(baseModel[pointer], baseModel[pointer+1]);
            pointer += 2;
        }
    }
    
    void clear(){
        for(int i=0; i<HL_SIZE; i++){
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
    }
    
    int get_index(int piece, int square) const{
        return (piece<<6)|(square^56);
    }
    
    template<int f>
    void change2(const int piece, const int square){
        int index = get_index(piece, square);
        int index2 = index ^ 56 ^ 64; // point of view change
        for(int i=0; i<HL_SIZE; i++){
            accs[WHITE][i] += f*hlWeights[index][i];
            accs[BLACK][i] += f*hlWeights[index2][i];
        }
    }
    
    dbyte eval(const bool side, const int idBucket) const{
        int res = 0;
        for(int i=0; i<HL_SIZE; i++){
            res += activation(accs[side][i])*outWeights[idBucket][i];
            res += activation(accs[side^1][i])*outWeights[idBucket][i+HL_SIZE];
        }
        res /= QA;
        res += outbias[idBucket];
        if(res >= 0)
            return res*SCALE/(QA*QB);
        else 
            return (res*SCALE-(QA*QB-1))/(QA*QB);
    }
};
#endif