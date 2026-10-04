#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <string>
#include "Const.hpp"
using namespace std;
int col(const int& square);
int row(const int& square);
int color(const int& piece);
int type(const int& piece);
int countbit(const u64& board);
int flip(const int& square);
int places(u64 mask, u8* positions);
u64 reverse(u64 board);
u64 reverse_col(u64 board);
void print_mask(u64 mask);
u64 addBitToMask(const u64& mask, const int& pos);
u64 removeBitFromMask(u64 mask, int pos);
int from_str(string a);
string to_uci(int pos);
int clipped_right(int pos);
int clipped_left(int pos);
u64 mask_empty_rook(int square);
u64 mask_empty_bishop(int square);
u64 mask_full_rook(int square);
u64 mask_full_bishop(int square);
u64 maskCol(int square);
char transform(u8 n);
int sign(int n);
class depthInfo {
   public:
    i64 node;
    int time, nps, depth, seldepth, score;
};

#endif