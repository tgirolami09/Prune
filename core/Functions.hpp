#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include "Const.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
using namespace std;
inline  int col(const int& square){
    return square&7;
}

inline int row(const int& square){
    return square >> 3;
}

inline int color(const int& piece){
    return piece%2;
}

inline int type(const int& piece){
    return piece/2;
}

inline int countbit(const big& board){
    return __builtin_popcountll(board);
}
inline int flip(const int& square){
    return square^56;
}
inline int places(big mask, ubyte* positions){
    int nbBits = countbit(mask);
    for(ubyte i=0; mask; i++){
        ubyte bit=__builtin_ctzll(mask);
        mask &= mask-1; //even if it has a sub, whet it's compiled it will be as the blsr instruction
        positions[i] = bit;
    }
    return nbBits;
}

inline big reverse(big board){
    board = (board&0xFFFFFFFF00000000) >> 32 | (board&0x00000000FFFFFFFF) << 32;
    board = (board&0xFFFF0000FFFF0000) >> 16 | (board&0x0000FFFF0000FFFF) << 16;
    board = (board&0xFF00FF00FF00FF00) >>  8 | (board&0x00FF00FF00FF00FF) <<  8;
    return board;
}

inline void print_mask(big mask){
    for(int row=0; row<8; row++){
        for(int col=0; col<8; col++){
            big _mask = 1ULL << (63-(row*8+col));
            if(mask&_mask)printf("1");
            else printf("0");
        }
        printf("\n");
    }
}

inline big addBitToMask(const big& mask, const int& pos){
    return mask | 1ul << pos;
}

inline big removeBitFromMask(big mask, int pos){
    return mask & ~(1ul << pos);
}

inline int from_str(string a){
    int col = 7-(a[0]-'a');
    int row = (a[1]-'0')-1;
    return row << 3|col;
}

inline string to_uci(int pos){
    string uci;
    uci += (7-col(pos))+'a';
    uci += row(pos)+'1';
    return uci;
}

inline int clipped_right(int pos){
    if((pos&7) == 0)return pos;
    return pos-1;
}

inline int clipped_left(int pos){
    if((pos&7) == 7)return pos;
    return pos+1;
}

inline big mask_empty_rook(int square){
    return (clipped_col[square&7]|clipped_row[square >> 3])&(~(1ULL << square));
}
inline big mask_empty_bishop(int square){
    int col=square&7, row=square >> 3;
    return clipped_diag[col+row] ^ clipped_idiag[row-col+7];
}

inline big maskCol(int square){
    return colH << col(square);
}

inline char transform(ubyte n){
    if(n >= 128)
        return ((char)n) - 256;
    return n;
}
inline int sign(int n){
    return n < 0 ? -1 : 1;
}

class depthInfo{
public:
    int node, time, nps, depth, seldepth, score;
};

#endif