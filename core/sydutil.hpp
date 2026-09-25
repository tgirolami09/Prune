#pragma once
#include <cstdint>
#include <vector>
#include "GameState.hpp"
#include "Move.hpp"
#include "viriformatUtil.hpp"
using namespace std;

struct node {
   public:
    MoveInfo mvscore;
    vector<node> childs;
    void write(vector<uint8_t>& buffer) const;
    void writeroot(vector<uint8_t>& buffer, const GameState& root) const;
    bool operator<(const node& a) { return mvscore.move.moveInfo < a.mvscore.move.moveInfo; }
};