#pragma once
#include <cstdint>
#include <vector>
#include "GameState.hpp"
#include "Move.hpp"
#include "TranspositionTable.hpp"
#include "viriformatUtil.hpp"
using namespace std;

struct Link {
    uint32_t hashidx;
    uint32_t bucketidx;
    Move mv;
};

struct node {
   public:
    MoveInfo mvscore;
    uint8_t bound;
    uint16_t depth;
    residualHash rem;
    vector<Link> childs;
    node() {
        depth = -1;
        bound = UPPERBOUND;
        mvscore.score = 0;
        mvscore.move = nullMove;
        age = -1;
    }
    node(const infoScore& entry) {
        mvscore.score = entry.score;
        mvscore.move = entry.bestMove;
        depth = entry.depth;
        bound = entry.typeNode();
        rem = entry.hash;
    }
    node(MoveInfo mvsc, int _bound, int _depth, residualHash _rem) {
        mvscore = mvsc;
        depth = _depth;
        bound = _bound;
        rem = _rem;
    }
    int age;
    void write(vector<uint8_t>& buffer, const vector<vector<node>>& nodetable) const;
    void writeroot(vector<uint8_t>& buffer, const GameState& root,
                   const vector<vector<node>>& nodetable) const;
    void update(const infoScore& ttentry);
    void dumpinfo(vector<uint8_t>& buffer) const;
};
void totree(GameState& state, const transpositionTable& tt, Link curnode,
            vector<vector<node>>& nodetable, int curage, int& nbNew);