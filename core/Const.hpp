#ifndef CONST_HPP
#define CONST_HPP
#include <cstdint>
#include <map>
#include <cinttypes>
//#define ASSERT
#define big uint64_t
#define ubyte uint8_t
#define dbyte int16_t
#define sbig int64_t
using namespace std;
extern int nbThreads;
extern bool DEBUG;
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

const int maxDepth=200;
const int maxMoves=218;
const int maxCaptures = 12*8+4*4;
const int maxExtension = 16;
const big hashMul = 1024*1024;

const int MINIMUM=-32767;
const int MAXIMUM=-MINIMUM;
const int INF=MAXIMUM;
const int MIDDLE=0;

const ubyte EXACT = 0;
const ubyte LOWERBOUND = 1;
const ubyte UPPERBOUND = 2;
const int KILLER_ADVANTAGE = 1<<20;
//const int value_pieces[7] = {100, 300, 300, 500, 900, 100000, 0};
const int maxHistory=1165;
const int granularDepth=128;
const int granularDepth2 = granularDepth*granularDepth;
static_assert(granularDepth < INT32_MAX/maxDepth, "granularDepth or maxDepth to big");

class pawnStruct{
public:
    big blackPawn;
    big whitePawn;
    int score;
    bool operator==(pawnStruct s);
};

#endif