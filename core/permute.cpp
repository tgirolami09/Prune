#include "NNUE.hpp"
#include <memory>
#include <array>
#include "simd_definitions.hpp"

static constexpr bool isPW = true;
constexpr int simdSize = nb<16>;
constexpr int mask = simdSize*2-1;
alignas(64) const auto first = []{
    array<int16_t, simdSize> res{};
    for(int i=0; i<simdSize; i++){
        res[i] = i;
    }
    return res;
}();
alignas(64) const auto second = []{
    array<int16_t, simdSize> res{};
    for(int i=0; i<simdSize; i++){
        res[i] = i+simdSize;
    }
    return res;
}();

const _simd _packed = ADDMM(packus_epi16)(*(_simd*)&first, *(_simd*)&second);
const int8_t* packed = (int8_t*)&_packed;

const auto unpacked = []{
    array<int8_t, simdSize*2> res{};
    for(int i=0; i<simdSize*2; i++){
        res[packed[i]] = i;
    }
    return res;
}();

int permute(int n){
    return (n&~mask) | unpacked[n&mask];
}

template<int input, int output, typename wtype>
struct layer{
    alignas(64) wtype weights[input][output];
    alignas(64) int32_t bias[output];
};

struct inputlayer{
    alignas(64) int16_t psqweights[nbInputBuckets][INPUT_SIZE][L1];
    alignas(64) int8_t threatweights[THREAT_SIZE][L1];
    alignas(64) int16_t biases[L1];
};

struct lastLayers{
    layer<L1*(2-isPW), L2, int8_t> l1;
    layer<L2, L3, int32_t> l2;
    layer<L3, 1, int32_t> l3;
};

struct nn{
    inputlayer FT;
    lastLayers laterLayers[BUCKET];
};

int main(int argc, char** argv){
    assert(argc > 2);
    unique_ptr<nn> nn_in  = make_unique<nn>();
    unique_ptr<nn> nn_out = make_unique<nn>();
    FILE* fin = fopen(argv[1], "rb");
    FILE* fout = fopen(argv[2], "wb");
    fread(&nn_in->FT, sizeof(nn_in->FT), 1, fin);
    memcpy(&nn_out->FT, &nn_in->FT, sizeof(nn_in->FT));
    for(int ob=0; ob<BUCKET; ob++){
        fread(&nn_in->laterLayers[ob], sizeof(nn_in->laterLayers[ob]), 1, fin);
        memcpy(&nn_out->laterLayers[ob], &nn_in->laterLayers[ob], sizeof(nn_in->laterLayers[ob]));
    }

    for(int ib=0; ib<nbInputBuckets; ib++)for(int i=0; i<INPUT_SIZE; i++)for(int k=0; k<L1; k++){
        nn_out->FT.psqweights[ib][i][k] = nn_in->FT.psqweights[ib][i][permute(k)];
    }
    for(int i=0; i<THREAT_SIZE; i++)for(int k=0; k<L1; k++){
        nn_out->FT.threatweights[i][k] = nn_in->FT.threatweights[i][permute(k)];
    }
    for(int k=0; k<L1; k++){
        nn_out->FT.biases[k] = nn_in->FT.biases[permute(k)];
    }
    fwrite(&nn_out->FT, sizeof(nn_out->FT), 1, fout);
    for(int i=0; i<BUCKET; i++)
        fwrite(&nn_out->laterLayers[i], sizeof(nn_out->laterLayers[i]), 1, fout);
    fclose(fin);
    fclose(fout);
}