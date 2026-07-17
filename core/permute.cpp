#include "NNUE.hpp"
#include <memory>
#include <array>
#include "simd_definitions.hpp"

static constexpr bool isPW = true;
static constexpr bool isMergedKingPlanes=true;
constexpr int RawInputSize = INPUT_SIZE+isMergedKingPlanes*64;
#ifdef ARCHSIZE
constexpr int simdSize = ARCHSIZE/16;
#else
constexpr int simdSize = nb<16>;
#endif
constexpr int mask = simdSize*2-1;
constexpr int LaneSize = 128/16;
constexpr int nbLane = simdSize*2/LaneSize;
const auto unpacked = []{
    array<int8_t, simdSize*2> res{};
    for(int i=0; i<simdSize*2; i++){
        res[i] = i%simdSize/LaneSize*2*LaneSize+i%LaneSize+(i/simdSize)*LaneSize;
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

template<bool isPermuted>
struct inputlayer{
    alignas(64) int16_t psqweights[nbInputBuckets][isPermuted?INPUT_SIZE:RawInputSize][L1];
    alignas(64) int8_t threatweights[THREAT_SIZE][L1];
    alignas(64) int16_t biases[L1];
};

struct lastLayers{
    layer<L1*(2-isPW), L2, int8_t> l1;
    layer<L2, L3, int32_t> l2;
    layer<L3, 1, int32_t> l3;
};

template<bool isPermuted>
struct nn{
    inputlayer<isPermuted> FT;
    lastLayers laterLayers[BUCKET];
};

int col(const int& square){
    return square&7;
}

int row(const int& square){
    return square >> 3;
}

int main(int argc, char** argv){
    assert(argc > 2);
    unique_ptr<nn<false>> nn_in  = make_unique<nn<false>>();
    unique_ptr<nn<true>> nn_out = make_unique<nn<true>>();
    FILE* fin = fopen(argv[1], "rb");
    FILE* fout = fopen(argv[2], "wb");
    fread(&nn_in->FT, sizeof(nn_in->FT), 1, fin);
    for(int ob=0; ob<BUCKET; ob++){
        fread(&nn_in->laterLayers[ob], sizeof(nn_in->laterLayers[ob]), 1, fin);
        memcpy(&nn_out->laterLayers[ob], &nn_in->laterLayers[ob], sizeof(nn_in->laterLayers[ob]));
    }

    for(int ib=0; ib<nbInputBuckets; ib++)for(int i=0; i<RawInputSize; i++)for(int k=0; k<L1; k++){
        const int pieceType = i/64%6;
        const int color = i/64/6;
        const int pos = i%64;
        if(pieceType != KING || (col(pos) > 3 && color) || (col(pos) <= 3 && (inputBuckets[col(pos) | (row(pos) << 2)] != ib) == color)){
            nn_out->FT.psqweights[ib][i-6*64*(pieceType == KING && color)][k] = nn_in->FT.psqweights[ib][i][permute(k)];
        }
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