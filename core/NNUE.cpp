#include <cstdint>
#include <experimental/simd>
using namespace std::experimental;
const int INPUT_SIZE = 12*64;
const int HL_SIZE = 64;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
#define dbyte int16_t
#define simd16 native_simd<dbyte>
const int nb16 = std::size(simd16());
#define simdint fixed_size_simd<int, nb16>
static_assert(HL_SIZE%nb16 == 0);

simdint convert16(simd16 var){
    return simdint([var](int i){return var[i];});
}

template<int min, int max>
simdint SCReLU(simdint value){
    where(value <= min, value) = min*min;
    where(value >= max, value) = max*max;
    where(value > min && value < max, value) = value*value;
    return value;
}

simdint activation(simdint value){
    return SCReLU<0, QA>(value);
}
int mysum(simdint x){
    int res=0;
    for(int i=0; i<size(x); i++)
        res += x[i];
    return res;
}
class NNUE{
public:
    simd16 hlWeights[INPUT_SIZE][HL_SIZE/nb16];
    simd16 hlBiases[HL_SIZE/nb16];
    simdint outWeights[2*HL_SIZE/nb16];
    dbyte outbias;
    simd16 accs[2][HL_SIZE/nb16];
    dbyte read_bytes(ifstream& file){
        dbyte ret;
        file.read(reinterpret_cast<char*>(&ret), sizeof(ret));
        return ret;
    }
    NNUE(string name){
        ifstream file(name);
        for(int i=0; i<INPUT_SIZE; i++)
            for(int j=0; j<HL_SIZE/nb16; j++)
                for(int k=0; k<nb16; k++)
                    hlWeights[i][j][k] = read_bytes(file);
        for(int i=0; i<HL_SIZE/nb16; i++){
            for(int id16=0; id16<nb16; id16++)hlBiases[i][id16] = read_bytes(file);
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
        for(int i=0; i<2*HL_SIZE/nb16; i++){
            for(int id16=0; id16<nb16; id16++)outWeights[i][id16] = read_bytes(file);
        }
        outbias = read_bytes(file);
    }

    void clear(){
        for(int i=0; i<HL_SIZE/nb16; i++){
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
    }

    void boardFlip(int& piece, int& square){
        piece ^= 1;
        square ^= 56;
    }
    int get_index(int piece, int square){
        return piece*64+(square^56);
    }
    template<int f> // -1 for remove, +1 for add (a piece)
    void acc(int index, bool side){
        for(int i=0; i<HL_SIZE/nb16; i++){
            accs[side][i] += f*hlWeights[index][i];
        }
    }

    template<int f>
    void change2(int piece, int square){
        //printf("change with p=%d & sq=%d (f=%d)\n", piece, square, f);
        acc<f>(get_index(piece, square), WHITE);
        boardFlip(piece, square);
        acc<f>(get_index(piece, square), BLACK);
    }

    dbyte eval(bool side){
        simdint res = 0;
        for(int i=0; i<HL_SIZE/nb16; i++){
            res += activation(convert16(accs[side  ][i]))*outWeights[i];
            res += activation(convert16(accs[side^1][i]))*outWeights[i+HL_SIZE/nb16];
        }
        int finRes = mysum(res);
        finRes /= QA;
        finRes += outbias;

        if(finRes >= 0)
            finRes = finRes*SCALE/(QA*QB);
        else finRes = (finRes*SCALE-(QA*QB-1))/(QA*QB);
        return finRes;
    }
};