#ifndef CONST_HPP
#define CONST_HPP
#include <cstdint>
#include <map>
#define big uint64_t

using namespace std;

const int WHITE=0;
const int BLACK=1; // odd = black
const int PAWN=0;
const int KNIGHT=2;
const int BISHOP=4;
const int ROOK=6;
const int QUEEN=8;
const int KING=10;
const int SPACE=12;

map<char, int> piece_to_id = {{'r', ROOK}, {'n', KNIGHT}, {'b', BISHOP}, {'q', QUEEN}, {'k', KING}, {'p', PAWN}};
map<int, char> id_to_piece = {{ROOK, 'r'}, {KNIGHT, 'n'}, {BISHOP, 'b'}, {QUEEN, 'q'}, {KING, 'k'}, {PAWN, 'p'}, {SPACE, ' '}};
#endif