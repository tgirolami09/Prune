#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include "Const.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
using namespace std;
int col(int square){
    return square&7;
}

int row(int square){
    return square >> 3;
}

int color(int piece){
    return piece%2;
}

int type(int piece){
    return piece/2;
}

const int countbit(const big board){
    return __builtin_popcountll(board);
}

int places(big mask, ubyte*& positions){
    int nbBits = countbit(mask);
    positions = (ubyte*)malloc(nbBits);
    for(ubyte i=0; mask; i++){
        ubyte bit=ffsll(mask)-1;
        mask ^= 1ULL << bit;
        positions[i] = bit;
    }
    return nbBits;
}

big reverse(big board){
    board = (board&0xFFFFFFFF00000000) >> 32 | (board&0x00000000FFFFFFFF) << 32;
    board = (board&0xFFFF0000FFFF0000) >> 16 | (board&0x0000FFFF0000FFFF) << 16;
    board = (board&0xFF00FF00FF00FF00) >>  8 | (board&0x00FF00FF00FF00FF) <<  8;
    return board;
}

void print_mask(big mask){
    int col=0;
    for(int row=8; row<=64; row+=8){
        for(; col<row; col++){
            printf("%d", (int)mask&1);
            mask >>= 1;
        }
        printf("\n");
    }
}

big addBitToMask(big mask, int pos){
    return mask | 1ul << pos;
}

big removeBitFromMask(big mask, int pos){
    return mask & ~(1ul << pos);
}

int from_str(string a){
    int col = a[0]-'a';
    int row = a[1]-'0';
    return row << 3|col;
}

#endif