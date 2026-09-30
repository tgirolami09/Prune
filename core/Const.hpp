#ifndef CONST_HPP
#define CONST_HPP
#include <cinttypes>
#include <cstdint>
#include <map>
// #define ASSERT
using u64 = uint64_t;
using i64 = int64_t;

using u16 = uint16_t;
using i16 = int16_t;

using u8 = uint8_t;
using namespace std;
#define forceinline __attribute__((always_inline))
#define _unused __attribute__((unused))
extern int nbThreads;
extern bool DEBUG;
extern bool isdfrc;
const u64 MAX_BIG = ~0ULL;
enum Color : uint8_t {
    White,
    Black,
};
enum Piece : uint8_t { Pawn, Knight, Bishop, Rook, Queen, King, Void };
const int nbPieces = 6;
const u64 colA = 0x8080808080808080;
const u64 colH = 0x0101010101010101;
const u64 row1 = 0xff;
const u64 row8 = 0xffULL << 56;
const map<char, int> piece_to_id = {{'r', Piece::Rook},  {'n', Piece::Knight}, {'b', Piece::Bishop},
                                    {'q', Piece::Queen}, {'k', Piece::King},   {'p', Piece::Pawn}};
const char id_to_piece[7] = {'p', 'n', 'b', 'r', 'q', 'k', ' '};

extern u64 clipped_row[8];
extern u64 clipped_col[8];
extern u64 clipped_diag[15];
extern u64 clipped_idiag[15];
extern u64 mask_row[8];
extern u64 mask_col[8];
extern u64 mask_diag[15];   // diag : index = column+row
extern u64 mask_idiag[15];  // idiag : index = row-column+7
extern u64 bishop_empty[64];
extern u64 rook_empty[64];
extern u64 bishop_full[64];
extern u64 rook_full[64];
const u64 clipped_brow = (MAX_BIG >> 16 << 8);
const u64 clipped_bcol = (~0x8181818181818181);
const u64 clipped_mask = clipped_brow & clipped_bcol;

const int maxDepth = 200;
const int maxMoves = 218;
const int maxCaptures = 12 * 8 + 4 * 4;
const int maxExtension = 16;
const u64 hashMul = 1024 * 1024;

const int MINIMUM = -32767;
const int MAXIMUM = -MINIMUM;
const int INF = MAXIMUM;
const int MIDDLE = 0;

enum Bound : uint8_t {
    Exact,
    Lower,
    Upper,
};
const int KILLER_ADVANTAGE = 1 << 20;
// const int value_pieces[7] = {100, 300, 300, 500, 900, 100000, 0};
const int maxHistory = 16384;

constexpr int kingposCastle[2] = {1, 5};
constexpr int rookposCastle[2] = {2, 4};
constexpr int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

constexpr int fracDepth = 128;
template <int d>
constexpr int fdepth = d * fracDepth;

extern u64 directions[64][64];
extern u64 fullDir[64][64];

extern u64 wide3[8];

#endif