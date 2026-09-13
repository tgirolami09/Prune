#include <immintrin.h>
#include <vector>
#include "GameState.hpp"
#include "Move.hpp"

template <typename T>
void fastWrite(T data, FILE* file);
class MoveInfo {
   public:
    Move move;
    int score;
    MoveInfo();
    void dump(FILE* datafile);
    static const int size = 4;
};
class GamePlayed {
   public:
    vector<MoveInfo> game;
    GameState startPos;
    ubyte result;
    static const int headerSize = 8 + 16 + 8;
    void dump(FILE* datafile);
    void clear();
};
void dumpPosition(__m256i position, const ubyte flags, FILE* datafile, int16_t score, ubyte bound,
                  Move move, int depth, int rule50);
GamePlayed readGame(FILE* file);