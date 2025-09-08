#include <cstdint>
#include <experimental/simd>
namespace stdx = std::experimental;
const int INPUT_SIZE = 12*64;
const int HL_SIZE = 64;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
#define dbyte int16_t
#define simd16 stdx::native_simd<dbyte>
const int nb16 = std::size(simd16());
#define simdint stdx::fixed_size_simd<int, nb16>
static_assert(HL_SIZE%nb16 == 0);

simdint convert16(simd16 var){
    return stdx::static_simd_cast<int>(var);
}

template<int min, int max>
simdint SCReLU(simd16 value){
    static const simd16 mini = min;
    static const simd16 maxi = max;
    value = stdx::clamp(value, mini, maxi);
    simdint res = convert16(value);
    return res*res;
}

simdint activation(simd16 value){
    return SCReLU<0, QA>(value);
}
int mysum(simdint x){
    return stdx::reduce(x, std::plus{});
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
            for(int id16=0; id16<nb16; id16++)
                hlBiases[i][id16] = read_bytes(file);
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
        for(int i=0; i<2*HL_SIZE/nb16; i++)
            for(int id16=0; id16<nb16; id16++)
                outWeights[i][id16] = read_bytes(file);
        outbias = read_bytes(file);
    }

    void clear(){
        for(int i=0; i<HL_SIZE/nb16; i++){
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
    }

    int get_index(int piece, int square){
        return piece*64+(square^56);
    }

    template<int f>
    void change2(int piece, int square){
        int index = get_index(piece, square);
        int index2 = index ^ 56 ^ 64; //point of vue change (^64 for piece color and ^56 for horizontal mirroring)
        for(int i=0; i<HL_SIZE/nb16; i++){
            accs[WHITE][i] += f*hlWeights[index ][i];
            accs[BLACK][i] += f*hlWeights[index2][i];
        }
    }

    dbyte eval(bool side){
        simdint res = 0;
        for(int i=0; i<HL_SIZE/nb16; i++){
            res += activation(accs[side  ][i])*outWeights[i];
            res += activation(accs[side^1][i])*outWeights[i+HL_SIZE/nb16];
        }
        int finRes = mysum(res);
        finRes /= QA;
        finRes += outbias;

        if(finRes >= 0)
            finRes = finRes*SCALE/(QA*QB);
        else finRes = (finRes*SCALE-(QA*QB-1))/(QA*QB); //integer division like python's one, because my trainer is on python
        return finRes;
    }
};