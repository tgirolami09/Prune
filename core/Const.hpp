#ifndef CONST_HPP
#define CONST_HPP
#include <cstdint>
#include <map>
#include <cassert>
//#define ASSERT
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
const big colA=0x8080808080808080;
const big colH=0x0101010101010101;
const map<char, int> piece_to_id = {{'r', ROOK}, {'n', KNIGHT}, {'b', BISHOP}, {'q', QUEEN}, {'k', KING}, {'p', PAWN}};
const char id_to_piece[7] = {'p', 'n', 'b', 'r', 'q', 'k', ' '};

big clipped_row[8];
big clipped_col[8];
big clipped_diag[15];
big clipped_idiag[15];
const big clipped_brow = (MAX_BIG >> 16 << 8);
const big clipped_bcol = (~0x8181818181818181);
const big clipped_mask = clipped_brow & clipped_bcol;
void init_lines(){
    big row = MAX_BIG >> (8*7+2) << 1;
    big col = 0x0001010101010100ULL;
    for(int i=0; i<8; i++){
        clipped_row[i] = row;
        //print_mask(row);
        //print_mask(col);
        clipped_col[i] = col;
        row <<= 8;
        col <<= 1;
    }
    big diag = 0;
    big idiag = 0;
    for(int i=0; i<15; i++){
        diag <<= 8;
        if(i < 8)diag |= 1 << i;
        idiag <<= 8;
        if(i < 8)idiag |= 1 << (7-i);
        clipped_diag[i] = diag&clipped_mask;
        clipped_idiag[i] = idiag&clipped_mask;
    }
}

const int maxDepth=200;
const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
const int maxExtension = 16;

const int MINIMUM=-10000;
const int MAXIMUM=10000;
const int INF=MAXIMUM+200;
const int MIDDLE=0;

const ubyte EXACT = 0;
const ubyte LOWERBOUND = 1;
const ubyte UPPERBOUND = 2;
#endif