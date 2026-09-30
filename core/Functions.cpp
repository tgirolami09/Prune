#include "Functions.hpp"
#include "Const.hpp"
using namespace std;
int col(const int& square) {
    return square & 7;
}

int row(const int& square) {
    return square >> 3;
}

int color(const int& piece) {
    return piece % 2;
}

int type(const int& piece) {
    return piece / 2;
}

int countbit(const u64& board) {
    return __builtin_popcountll(board);
}
int flip(const int& square) {
    return square ^ 56;
}
int places(u64 mask, u8* positions) {
    int nbBits = countbit(mask);
    for (u8 i = 0; mask; i++) {
        u8 bit = __builtin_ctzll(mask);
        mask &= mask - 1;  // even if it has a sub, whet it's compiled it will be
                           // as the blsr instruction
        positions[i] = bit;
    }
    return nbBits;
}

u64 reverse(u64 board) {
    board = (board & 0xFFFFFFFF00000000) >> 32 | (board & 0x00000000FFFFFFFF) << 32;
    board = (board & 0xFFFF0000FFFF0000) >> 16 | (board & 0x0000FFFF0000FFFF) << 16;
    board = (board & 0xFF00FF00FF00FF00) >> 8 | (board & 0x00FF00FF00FF00FF) << 8;
    return board;
}

u64 reverse_col(u64 board) {
    board = (board & 0xF0F0F0F0F0F0F0F0) >> 4 | (board & 0x0F0F0F0F0F0F0F0F) << 4;
    board = (board & 0xCCCCCCCCCCCCCCCC) >> 2 | (board & 0x3333333333333333) << 2;
    board = (board & 0xAAAAAAAAAAAAAAAA) >> 1 | (board & 0x5555555555555555) << 1;
    return board;
}

void print_mask(u64 mask) {
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            u64 _mask = 1ULL << (63 - (row * 8 + col));
            if (mask & _mask)
                printf("1");
            else
                printf("0");
        }
        printf("\n");
    }
}

u64 addBitToMask(const u64& mask, const int& pos) {
    return mask | 1ull << pos;
}

u64 removeBitFromMask(u64 mask, int pos) {
    return mask & ~(1ull << pos);
}

int from_str(string a) {
    int col = 7 - (a[0] - 'a');
    int row = (a[1] - '0') - 1;
    return row << 3 | col;
}

string to_uci(int pos) {
    string uci;
    uci += (7 - col(pos)) + 'a';
    uci += row(pos) + '1';
    return uci;
}

int clipped_right(int pos) {
    return pos - ((pos & 7) != 0);
}

int clipped_left(int pos) {
    return pos + ((pos & 7) != 7);
}

u64 mask_empty_rook(int square) {
    return rook_empty[square];
}
u64 mask_empty_bishop(int square) {
    return bishop_empty[square];
}
u64 mask_full_rook(int square) {
    return rook_full[square];
}
u64 mask_full_bishop(int square) {
    return bishop_full[square];
}

u64 maskCol(int square) {
    return colH << col(square);
}

char transform(u8 n) {
    if (n >= 128)
        return ((char)n) - 256;
    return n;
}
int sign(int n) {
    return n < 0 ? -1 : 1;
}