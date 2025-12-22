#include "Move.hpp"
#include "GameState.hpp"
#include <vector>

template<typename T> 
void fastWrite(T data, FILE* file);
class MoveInfo{
public:
    Move move;
    int score;
    MoveInfo();
    void dump(FILE* datafile);
    static const int size = 4;
};
class GamePlayed{
public:
    vector<MoveInfo> game;
    GameState startPos;
    ubyte result;
    static const int headerSize = 8+16+8;
    void dump(FILE* datafile);
    void clear();
};