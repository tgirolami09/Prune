#pragma once
#include <vector>
#include "GameState.hpp"
#include "Move.hpp"

template <typename T>
void fastWrite(T data, FILE* file);
void dumpposition(vector<uint8_t>& buffer, const GameState& startPos);
void dumpmove(Move move, vector<uint8_t>& buffer);
class MoveInfo {
   public:
    Move move;
    int score;
    MoveInfo();
    MoveInfo(Move move, int score);
    void dump(FILE* datafile) const;
    void dump(vector<uint8_t>& datafile) const;
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

GamePlayed readGame(FILE* file);