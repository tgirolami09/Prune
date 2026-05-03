#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "viriformatUtil.hpp"
#include "LegalMoveGenerator.hpp"
#include <cstdint>
#include <random>
#include <cassert>
#include <string>
using namespace std;

string niceNumber(big N){
    int count=0;
    while(N > 1000){
        N /= 1000;
        count++;
    }
    string suffix[5] = {"", "k", "M", "G", "T"};
    return to_string(N)+suffix[count];
}

class filtering{
public:
    int min_ply;
    int min_pieces;
    int max_eval;
    bool filter_tactical,
        filter_check,
        filter_castling;
    int max_eval_incorrectness;
    bool random_fen_skipping;
    double random_fen_skip_probability;
    int material_min,
        material_max;
    std::mt19937 randomGen;
    std::uniform_real_distribution<double> dist;

    filtering():
        min_ply(8),
        min_pieces(4),
        max_eval(31339),
        filter_tactical(true),
        filter_check(true),
        filter_castling(false),
        max_eval_incorrectness(INT32_MAX),
        random_fen_skipping(false),
        random_fen_skip_probability(0.0),
        material_min(17),
        material_max(78),
        randomGen(0),
        dist(0, 1)
    {}

    bool filter(const GameState& state, MoveInfo move, bool inCheck, int result){
        if(state.turnNumber < min_ply) return true;
        if(filter_check && inCheck)return true;
        if(abs(move.score) > max_eval)return true;
        if(filter_tactical && move.move.isTactical())return true;
        if(filter_castling && move.move.isCastling())return true;
        int nbMan = countbit(state.board.colors[WHITE]|state.board.colors[BLACK]);
        if(nbMan < min_pieces)return true;
        int value_pieces[5] = {1, 3, 3, 5, 9};
        int material=0;
        for(int j=0; j<5; j++)
            material += value_pieces[j]*countbit(state.board.pieces[j]);
        if(material < material_min || material > material_max)return true;
        if(random_fen_skipping && dist(randomGen) > random_fen_skip_probability)return true;
        if(result == 1 && abs(move.score) > max_eval_incorrectness)return true; // draw
        if(result == 2 && -move.score > max_eval_incorrectness)return true; // white win
        if(result == 0 && move.score > max_eval_incorrectness)return true;  // black win
        return false;
    }
};

class HyperLogLog{
public:
    int b;
    vector<int8_t> M;
    big mask;
    big size;
    HyperLogLog(int _b):b(_b), size(1ULL << b){
        M = vector<int8_t>(size, 0);
        mask = size-1;
    }
    void add(big data){
        big j = data >> (64-b);
        big w = data << b;
        M[j] = max<int8_t>(M[j], w?__builtin_clzll(w)+1:(64-b+1));
    }
    big count(){
        const double alpha = [&]{
            switch(size){
                case 16:
                    return 0.673;
                case 32:
                    return 0.697;
                case 64:
                    return 0.709;
                default:
                    return 0.7213/(1+1.079/size);
            }
        }();
        double sum=0;
        big V=0;
        for(big i=0; i<size; i++){
            if(M[i] == 0)V++;
            sum += pow(2.0, -M[i]);
        }
        double E = alpha*size*size/sum;
        if(E <= 2.5*size && V > 0){
            E = size*log((double)size/V);
        }
        double two64 = pow(2, 64);
        if(E > two64/30){
            E = -two64*log(1-E/two64);
        }
        return (big)E;
    }
};

int main(int argc, char** argv){
    if(argc == 1){
        printf("usage : \n%s filesToAnalyse\n", argv[0]);
        return 0;
    }
    filtering filter;
    big countFiltered=0;
    big countUnfiltered=0;
    HyperLogLog uniqueness(20);
    LegalMoveGenerator movegen;
    for(int idFile=1; idFile<argc; idFile++){
        FILE* file=fopen(argv[idFile], "r");
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        int idGame = 0;
        while(ftell(file) < file_size){
            GamePlayed game = readGame(file);
            //movegen.initDangers(game.startPos);
            idGame++;
            for(MoveInfo move:game.game){
                bool inCheck = movegen.initDangers(game.startPos);
                game.startPos.initMove(move.move);
                if(filter.filter(game.startPos, move, inCheck, game.result))countFiltered++;
                else countUnfiltered++;
                uniqueness.add(game.startPos.zobristHash);
                game.startPos.playMove(move.move);
            }
            if((idGame&16383) == 0){
                big nbPos = countFiltered+countUnfiltered;
                big countUnique = uniqueness.count();
                printf("\r%s %s/%s(%s %.2f) : %.2f %.2f                   ", niceNumber(countFiltered).c_str(), niceNumber(countUnfiltered).c_str(), niceNumber(nbPos).c_str(), niceNumber(countUnique).c_str(), countUnique*100.0/nbPos, (double)countFiltered*100/nbPos, (double)(countUnfiltered)*100/nbPos);
                fflush(stdout);
            }
        }
    }
    big nbPos = countFiltered+countUnfiltered;
    printf("\n%ld %ld/%ld (%ld): %.2f %.2f\n", countFiltered, countUnfiltered, nbPos, uniqueness.count(), (double)countFiltered*100/nbPos, (double)(countUnfiltered)*100/nbPos);
}