#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include "Const.hpp"
#include "GameState.hpp"
#include "NNUE.hpp"
#include "corrhist.hpp"
#include "tunables.hpp"

#ifdef DEBUG_MACRO
#include "stats_helpers.hpp"
extern StatVar<big, 48 * 1024, 0> matScalingStats;
#endif
const int gamephaseInc[6] = {0, 1, 1, 2, 4, 0};

int fastSEE(const Move& move, const GameState& state, const int* value_pieces);
bool see_ge(int born, const Move& move, const GameState& state, const int* value_pieces);
int score_move(const Move& move, int historyScore, const GameState& state, const int* value_pieces);

const int tableSize = 1 << 10;  // must be a power of two, for now it's pretty
                                // small because we should hit the table very
                                // often, and so we didn't use too much memory

class IncrementalEvaluator {
    int mgPhase;
    Accumulator stackAcc[maxDepth];
    FinnyTables finny;
    int nbMan;

   public:
    int stackIndex;
    template <int f, bool updateNNUE>
    void changePiece(const NNUE& nnue, int pos, int piece, bool c, bool updateNNUE2 = true);
    template <int f, bool updateNNUE>
    void changePiece2(const NNUE& nnue, int pos, int piece, bool c);
    void backStack();
    void print();
    IncrementalEvaluator();
    void init(const GameState& state, const NNUE& nnue);
    bool isInsufficientMaterial(const GameState& state) const;
    bool isOnlyPawns() const;
    int getScore(bool c, const corrhists& ch, const GameState& state, const tunables& parameters,
                 const NNUE& nnue);
    int getRaw(bool c, const NNUE& nnue);
    int correctEval(int eval, const corrhists& ch, const GameState& state,
                    const tunables& parameters) const;
    int getNbMan() const { return nbMan; }
    template <int f = 1>
    void playMove(const NNUE& nnue, Move move, bool c, const PositionState& state1,
                  const PositionState& state2);
    void playNoBack(const GameState& state, Move move, bool c, const NNUE& nnue);
    void undoMove(const NNUE& nnue, Move move, bool c, const PositionState& state1,
                  const PositionState& state2);
    const Accumulator& operator[](int idx) const;
};

#endif
