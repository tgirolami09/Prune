#ifndef CONST_HPP
#define CONST_HPP
#include <cstdint>
#include <cinttypes>
#include <map>
#include <cassert>
//#define ASSERT
#define big uint64_t
#define ubyte uint8_t
#define dbyte int16_t
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
const big row1 = 0xff;
const big row8 = 0xffULL << 56;
const map<char, int> piece_to_id = {{'r', ROOK}, {'n', KNIGHT}, {'b', BISHOP}, {'q', QUEEN}, {'k', KING}, {'p', PAWN}};
const char id_to_piece[7] = {'p', 'n', 'b', 'r', 'q', 'k', ' '};

extern big clipped_row[8];
extern big clipped_col[8];
extern big clipped_diag[15];
extern big clipped_idiag[15];
const big clipped_brow = (MAX_BIG >> 16 << 8);
const big clipped_bcol = (~0x8181818181818181);
const big clipped_mask = clipped_brow & clipped_bcol;
void init_lines();

const int maxDepth=200;
const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
const int maxExtension = 16;
const int hashMul = 1024*1024;

const int MINIMUM=-100000;
const int MAXIMUM=100000;
const int INF=MAXIMUM+200;
const int MIDDLE=0;

const ubyte EXACT = 0;
const ubyte LOWERBOUND = 1;
const ubyte UPPERBOUND = 2;
const int KILLER_ADVANTAGE = 1<<20;
const int value_pieces[6] = {100, 300, 300, 500, 900, 0};
const int maxHistory=KILLER_ADVANTAGE/value_pieces[QUEEN];

class pawnStruct{
public:
    big blackPawn;
    big whitePawn;
    int score;
    bool operator==(pawnStruct s);
};

#ifdef __unix__
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section .rodata\n" \
    ".global " #buffername "\n" \
    ".type " #buffername ", @object\n" \
    ".align 4\n" \
    #buffername":\n" \
    ".incbin " #filename "\n" \
    #buffername"_end:\n" \
    ".global "#buffername"_size\n" \
    ".type "#buffername"_size, @object\n" \
    ".align 4\n" \
    #buffername"_size:\n" \
    ".int "#buffername"_end - "#buffername"\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#elif defined(__APPLE__)
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section __TEXT,__const\n" \
    ".globl _" #buffername "\n" \
    ".align 4\n" \
    "_" #buffername":\n" \
    ".incbin " #filename "\n" \
    "_" #buffername"_end:\n" \
    ".globl _" #buffername"_size\n" \
    ".align 4\n" \
    "_" #buffername"_size:\n" \
    ".long _" #buffername"_end - _" #buffername "\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#else
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section .rdata\n" \
    ".global " #buffername "\n" \
    ".global " #buffername "_end\n" \
    "" #buffername":\n" \
    ".incbin " #filename "\n" \
    "" #buffername"_end:\n" \
    ".globl " #buffername"_size\n" \
    "" #buffername"_size:\n" \
    ".long " #buffername"_end - " #buffername "\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#endif
#endif