#ifndef TABLEBASEPROBE_HPP
#define TABLEBASEPROBE_HPP

#include "GameState.hpp"
#include "Move.hpp"
#include "Const.hpp"
#include "Functions.hpp"

// Tablebase result values (matching Fathom constants)
constexpr int TB_RESULT_LOSS = 0;
constexpr int TB_RESULT_BLESSED_LOSS = 1;
constexpr int TB_RESULT_DRAW = 2;
constexpr int TB_RESULT_CURSED_WIN = 3;
constexpr int TB_RESULT_WIN = 4;
constexpr int TB_RESULT_INVALID = -1;

// Score values for tablebase results
constexpr int TB_WIN_SCORE = 20000;  // Below mate but clearly winning
constexpr int TB_CURSED_WIN_SCORE = 1;  // Winning but drawable by 50-move
constexpr int TB_BLESSED_LOSS_SCORE = -1;  // Losing but drawable by 50-move

class TablebaseProbe {
private:
    bool initialized;
    int probeDepth;      // Minimum depth to probe
    int probeLimit;      // Maximum pieces to probe
    unsigned tbLargest;  // From Fathom: max pieces in loaded tables

public:
    TablebaseProbe();
    ~TablebaseProbe();

    // Initialize with path to tablebase files
    bool init(const std::string& path);

    // Configuration
    void setProbeDepth(int depth);
    void setProbeLimit(int limit);
    int getProbeDepth() const;
    int getProbeLimit() const;

    // Check if probing is available
    bool isAvailable() const;
    int maxPieces() const;

    // Probe functions
    // Returns TB_RESULT_* or TB_RESULT_INVALID
    int probeWDL(const GameState& state) const;

    // Root probe - returns best move and WDL
    // Used at root for perfect play
    int probeRoot(const GameState& state, Move& bestMove) const;

    // WDL-only root probe fallback (for when DTZ files are missing).
    // Filters moves[] in-place to only those with the optimal WDL rank.
    // Returns best WDL (TB_RESULT_*) or TB_RESULT_INVALID on failure.
    int probeRootWDLFallback(const GameState& state, Move* moves, int& nbMoves) const;

    // Convert WDL result to centipawn score adjusted for ply
    static int wdlToScore(int wdl, int ply);

    // Get piece count from state
    static int countPieces(const GameState& state);

    // Check if position can be probed (piece count, no castling)
    // Overload accepting precomputed piece count (from IncrementalEvaluator::getNbMan)
    // depth variant also checks depth >= probeDepth (for in-search gating)
    bool canProbe(const GameState& state, int nbMan, int depth) const;
    bool canProbe(const GameState& state, int nbMan) const;
    bool canProbe(const GameState& state) const;
};

// Global tablebase prober
extern TablebaseProbe tbProbe;

#endif
