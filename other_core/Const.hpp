#ifndef CONST_HPP
#define CONST_HPP
#include <cstdint>
#include <map>
#define big uint64_t
#define ubyte uint8_t
using namespace std;
const big MAX_BIG=~0ULL;
const int WHITE=0;
const int BLACK=1; // odd = black
const int PAWN=0;
const int KNIGHT=1;
const int BISHOP=2;
const int ROOK=3;
const int QUEEN=4;
const int KING=5;
const int SPACE=6;
const int nbPieces=6;
map<char, int> piece_to_id = {{'r', ROOK}, {'n', KNIGHT}, {'b', BISHOP}, {'q', QUEEN}, {'k', KING}, {'p', PAWN}};
const char id_to_piece[7] = {'p', 'n', 'b', 'r', 'q', 'k', ' '};
#endif