#include <map>
#include <vector>
#include <string>

// #include "util_magic.cpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "Move.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "Evaluator.hpp"
#include "BestMoveFinder.hpp"

// #include <strings.h>
#include <iostream>

// big* table[128]; // [bishop, rook]
// info constants[128];

using namespace std;
//Main class for the game
class Chess{
    public : GameState currentGame;

    //Get it from i/o
    float w_time;
    float b_time;
};

Move getBotMove(GameState gameState,float alloted_time){
    BestMoveFinder bestMoveFinder;
    Move moveToPlay = bestMoveFinder.bestMove(gameState,alloted_time);

    return moveToPlay;
}

Move getOpponentMove(){
    //TODO : Some logic to get it from i/o
    return {};
}

void doUCI(string UCI_instruction){
    //Implement actual logic for UCI management
}

int main(){
    string UCI_instruction = "programStart";

    while (UCI_instruction != "quit"){
        doUCI(UCI_instruction);

        getline(cin,UCI_instruction);
    }
}