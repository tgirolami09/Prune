#include "Const.hpp"
#include "NNUE.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Evaluator.hpp"
#include <cstdio>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
using namespace std;

string verbose(int add, int rem){
    string res = "";
    if(add+rem > 0)
        res += " 1 file changed";
    else res += " 0 files changed";
    if(add > 0)
        res += ", "+to_string(add)+" insertion"+(add > 1?"s":"")+"(+)";
    if(rem > 0)
        res += ", "+to_string(rem)+" deletion"+(rem > 1?"s":"")+"(-)";
    return res;
}

int main(int argc, char** argv){
    unique_ptr<GameState> state=make_unique<GameState>();
    unique_ptr<LegalMoveGenerator> gen = make_unique<LegalMoveGenerator>();
    Move listMove[maxMoves];
    unique_ptr<IncrementalEvaluator> eval = make_unique<IncrementalEvaluator>();
    vector<int> isUpdated(THREAT_SIZE, -1);
    int upd = 0;
    for(int i=1; i<argc; i++){
        //printf("fen %s\n", argv[i]);
        string fen(argv[i]);
        state->fromFen(fen);
        gen->initDangers(*state);
        bool inCheck;
        big dngpos;
        int nbMoves = gen->generateLegalMoves(*state, inCheck, listMove, dngpos);
        eval->init(*state);
        const bool pov=state->friendlyColor();
        const bool mirror = (*eval)[0].Kside[pov];
        for(int idMove=0; idMove<nbMoves; idMove++){
            Move& curMove = listMove[idMove];
            eval->playMove(curMove, state->friendlyColor(), &*state);
            const Accumulator& acc=(*eval)[eval->stackIndex];
            int countremoved = 0;
            int countadded = 0;
            for(int idThreat=0; idThreat<acc.update.nbThreats; idThreat++){
                ThreatIndex curThreat = acc.update.threatUpdates[idThreat].changepov(pov).mirror(mirror);
                if(curThreat.issemiexcluded())curThreat.swap();
                if(curThreat.isexcluded())curThreat.swap();
                assert(!curThreat.isexcluded() && !curThreat.issemiexcluded());
                if(isUpdated[(int)curThreat] == upd)assert(false);
                curThreat.print();
                isUpdated[(int)curThreat] = upd;
                countadded += !curThreat.remove;
                countremoved += curThreat.remove;
            }
            int ev=eval->getRaw(state->friendlyColor());
            printf("%s%s : %d\n", curMove.to_str().c_str(), verbose(countadded, countremoved).c_str(), ev);
            upd++;
            state->undoLastMove();
            eval->undoMove(curMove, state->friendlyColor());
        }
    }
}