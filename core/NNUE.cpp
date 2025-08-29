#include <cstdint>
const int INPUT_SIZE = 12*64;
const int HL_SIZE = 64;
const int SCALE = 400;
const int QA = 255;
const int QB = 64;
#define dbyte int16_t

template<dbyte min, dbyte max>
dbyte CReLU(dbyte value){
    if (value <= min)
        return min;

    if (value >= max)
        return max;

    return value;
}
template<dbyte min, dbyte max>
dbyte SCReLU(dbyte value){
    if(value <= min)
        return min*min;
    else if(value >= max)
        return max*max;
    return value*value;
}

int activation(dbyte value){
    return SCReLU<0, QA>(value);
}

class NNUE{
public:
    dbyte hlWeights[INPUT_SIZE][HL_SIZE];
    dbyte hlBiases[HL_SIZE];
    dbyte outWeights[2*HL_SIZE];
    dbyte outbias;
    dbyte accs[2][HL_SIZE];

    NNUE(string name){
        ifstream file(name);
        for(int i=0; i<INPUT_SIZE; i++)
            for(int j=0; j<HL_SIZE; j++)
                file.read(reinterpret_cast<char*>(&hlWeights[i][j]), sizeof(hlWeights[i][j]));
        for(int i=0; i<HL_SIZE; i++){
            file.read(reinterpret_cast<char*>(&hlBiases[i]), sizeof(hlBiases[i]));
            accs[WHITE][i] = hlBiases[i];
            accs[BLACK][i] = hlBiases[i];
        }
        for(int i=0; i<2*HL_SIZE; i++)
            file.read(reinterpret_cast<char*>(&outWeights[i]), sizeof(outWeights[i]));
        file >> outbias;
    }

    void clear(){
        for(int i=0; i<HL_SIZE; i++){
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
        for(int i=0; i<HL_SIZE; i++){
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
        int res=0;
        for(int i=0; i<HL_SIZE; i++){
            res += static_cast<int>(activation(accs[side  ][i]))*outWeights[i];
            res += static_cast<int>(activation(accs[side^1][i]))*outWeights[i+HL_SIZE];
        }
        res /= QA;
        res += outbias;
        return -res*SCALE/(QA*QB);
    }
};