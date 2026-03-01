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

    bool filter(const GameState& state, MoveInfo move, bool inCheck){
        if(state.turnNumber < min_ply) return true;
        if(filter_check && inCheck)return true;
        if(abs(move.score) > max_eval)return true;
        if(filter_tactical && move.move.isTactical())return true;
        if(filter_castling && move.move.isCastling())return true;
        int nbMan = 0;
        for(int i=0; i<2; i++)
            for(int j=0; j<6; j++)
                nbMan += countbit(state.boardRepresentation[i][j]);
        if(nbMan < min_pieces)return true;
        int value_pieces[5] = {1, 3, 3, 5, 9};
        int material=0;
        for(int i=0; i<2; i++)
            for(int j=0; j<5; j++)
                material += value_pieces[j]*countbit(state.boardRepresentation[i][j])*(i == WHITE?1:-1);
        if(material < material_min || material > material_max)return true;
        if(random_fen_skipping && dist(randomGen) > random_fen_skip_probability)return true;
        return false;
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
    LegalMoveGenerator movegen;
    for(int idFile=1; idFile<argc; idFile++){
        FILE* file=fopen(argv[idFile], "r");
        while(!feof(file)){
            GamePlayed game = readGame(file);
            //movegen.initDangers(game.startPos);
            for(MoveInfo move:game.game){
                game.startPos.initMove(move.move);
                game.startPos.playMove(move.move);
                bool inCheck = movegen.initDangers(game.startPos);
                if(filter.filter(game.startPos, move, inCheck))countFiltered++;
                else countUnfiltered++;
            }
            big nbPos = countFiltered+countUnfiltered;
            printf("\r%s %s/%s : %.2f %.2f                   ", niceNumber(countFiltered).c_str(), niceNumber(countUnfiltered).c_str(), niceNumber(nbPos).c_str(), (double)countFiltered*100/nbPos, (double)(countUnfiltered)*100/nbPos);
            fflush(stdout);
        }
    }
    big nbPos = countFiltered+countUnfiltered;
    printf("\n%ld %ld/%ld : %.2f %.2f\n", countFiltered, countUnfiltered, nbPos, (double)countFiltered*100/nbPos, (double)(countUnfiltered)*100/nbPos);
}