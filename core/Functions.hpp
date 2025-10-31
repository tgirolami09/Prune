#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include "Const.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
using namespace std;
int col(const int& square);
int row(const int& square);
int color(const int& piece);
int type(const int& piece);
int countbit(const big& board);
int flip(const int& square);
int places(big mask, ubyte* positions);
big reverse(big board);
void print_mask(big mask);
big addBitToMask(const big& mask, const int& pos);
big removeBitFromMask(big mask, int pos);
int from_str(string a);
string to_uci(int pos);
int clipped_right(int pos);
int clipped_left(int pos);
big mask_empty_rook(int square);
big mask_empty_bishop(int square);
big maskCol(int square);
char transform(ubyte n);
int sign(int n);
class depthInfo{
public:
    int node, time, nps, depth, seldepth, score;
};

#endif