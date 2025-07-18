#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include <climits>
#include <cmath>
const int value_pieces[5] = {100, 300, 300, 500, 900};
//Class to evaluate a position
const big mask_forward[64] = {
    0x0300000000000000, 0x0700000000000000, 0x0e00000000000000, 0x1c00000000000000, 0x3800000000000000, 0x7000000000000000, 0xe000000000000000, 0xc000000000000000, 0x0303000000000000, 0x0707000000000000, 0x0e0e000000000000, 0x1c1c000000000000, 0x3838000000000000, 0x7070000000000000, 0xe0e0000000000000, 0xc0c0000000000000, 0x0303030000000000, 0x0707070000000000, 0x0e0e0e0000000000, 0x1c1c1c0000000000, 0x3838380000000000, 0x7070700000000000, 0xe0e0e00000000000, 0xc0c0c00000000000, 0x0303030300000000, 0x0707070700000000, 0x0e0e0e0e00000000, 0x1c1c1c1c00000000, 0x3838383800000000, 0x7070707000000000, 0xe0e0e0e000000000, 0xc0c0c0c000000000, 0x0303030303000000, 0x0707070707000000, 0x0e0e0e0e0e000000, 0x1c1c1c1c1c000000, 0x3838383838000000, 0x7070707070000000, 0xe0e0e0e0e0000000, 0xc0c0c0c0c0000000, 0x0303030303030000, 0x0707070707070000, 0x0e0e0e0e0e0e0000, 0x1c1c1c1c1c1c0000, 0x3838383838380000, 0x7070707070700000, 0xe0e0e0e0e0e00000, 0xc0c0c0c0c0c00000, 0x0303030303030300, 0x0707070707070700, 0x0e0e0e0e0e0e0e00, 0x1c1c1c1c1c1c1c00, 0x3838383838383800, 0x7070707070707000, 0xe0e0e0e0e0e0e000, 0xc0c0c0c0c0c0c000, 0x0303030303030303, 0x0707070707070707, 0x0e0e0e0e0e0e0e0e, 0x1c1c1c1c1c1c1c1c, 0x3838383838383838, 0x7070707070707070, 0xe0e0e0e0e0e0e0e0, 0xc0c0c0c0c0c0c0c0
};
class Evaluator{
    //All the logic for evaluating a position
public:
    int MINIMUM=-INT_MAX;
    int MAXIMUM=INT_MAX;
    int MIDDLE=0;
private:
    big* reverse_all(big* pieces){
        big* res=(big*)calloc(6, sizeof(big));
        for(int i=0; i<6; i++){
            res[i] = reverse(pieces[i]);
        }
        return res;
    }
    int score(big* pieces, big* other){
        int score=0;
        for(int i=0; i<6; i++)
            if(i != KING)
                score += countbit(pieces[i])*value_pieces[i];
        ubyte* pawns;
        int nbPawns=places(pieces[0], pawns);
        // detect passed pawns
        for(int i=0; i<nbPawns; i++){
            if((other[0]&mask_forward[pawns[i]]) == 0)
                score += 8-row(pawns[i])+1;
        }
        return score;
    }

public:
    int positionEvaluator(GameState state){
        int scoreFriends, scoreEnemies;
        if(state.friendlyColor() == BLACK)
            scoreFriends=score(reverse_all(state.friendlyPieces()), reverse_all(state.enemyPieces()));
        else scoreFriends=score(state.friendlyPieces(), state.enemyPieces());
        if(state.enemyColor() == BLACK)
            scoreFriends=score(reverse_all(state.enemyPieces()), reverse_all(state.friendlyPieces()));
        else
            scoreEnemies=score(state.enemyPieces(), state.friendlyPieces());
        return scoreFriends-scoreEnemies;
        return scoreFriends/scoreEnemies;
    }
    int score_move(Move move, GameState state){
        int score=value_pieces[state.getPiece(move.end_pos)];

        if(move.promoteTo != -1)score += value_pieces[move.promoteTo];
        return score;
    }
};
#endif