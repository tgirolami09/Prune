#include "TablebaseProbe.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "../Fathom/src/tbprobe.h"

// Global tablebase prober instance
TablebaseProbe tbProbe;

TablebaseProbe::TablebaseProbe() : initialized(false), probeDepth(1), probeLimit(7), tbLargest(0) {}

TablebaseProbe::~TablebaseProbe() {
    if (initialized) {
        tb_free();
    }
}

bool TablebaseProbe::init(const std::string& path) {
    if (initialized) {
        tb_free();
        initialized = false;
    }

    if (path.empty() || path == "<empty>") {
        tbLargest = 0;
        return true;
    }

    bool result = tb_init(path.c_str());
    if (result) {
        initialized = true;
        tbLargest = TB_LARGEST;
    }
    return result;
}

void TablebaseProbe::setProbeDepth(int depth) {
    probeDepth = depth*fracDepth;
}

void TablebaseProbe::setProbeLimit(int limit) {
    probeLimit = limit;
}

int TablebaseProbe::getProbeDepth() const {
    return probeDepth;
}

int TablebaseProbe::getProbeLimit() const {
    return probeLimit;
}

bool TablebaseProbe::isAvailable() const {
    return initialized && tbLargest > 0;
}

int TablebaseProbe::maxPieces() const {
    return tbLargest;
}

int TablebaseProbe::countPieces(const GameState& state) {
    return countbit(state.board.colors[WHITE]|state.board.colors[BLACK]);
}

bool TablebaseProbe::canProbe(const GameState& state, int nbMan, int depth) const {
    return depth >= probeDepth && canProbe(state, nbMan);
}

bool TablebaseProbe::canProbe(const GameState& state, int nbMan) const {
    if (!initialized || tbLargest == 0) return false;

    if (nbMan > (int)tbLargest || nbMan > probeLimit) return false;

    // Cannot probe with castling rights
    if (state.castlingMask) {
        return false;
    }

    return true;
}

bool TablebaseProbe::canProbe(const GameState& state) const {
    return canProbe(state, countPieces(state));
}

// Helper function to convert GameState to Fathom format
static void stateToFathom(const GameState& state,
                          uint64_t& white, uint64_t& black,
                          uint64_t& kings, uint64_t& queens, uint64_t& rooks,
                          uint64_t& bishops, uint64_t& knights, uint64_t& pawns,
                          unsigned& ep, bool& turn) {
    // Combine color bitboard.piecess and convert to Fathom format using reverse_col
    white = reverse_col(state.board.colors[WHITE]);

    black = reverse_col(state.board.colors[BLACK]);

    // Combine piece type bitboard.piecess and convert to Fathom format
    kings   = reverse_col(state.board.pieces[KING]);
    queens  = reverse_col(state.board.pieces[QUEEN]);
    rooks   = reverse_col(state.board.pieces[ROOK]);
    bishops = reverse_col(state.board.pieces[BISHOP]);
    knights = reverse_col(state.board.pieces[KNIGHT]);
    pawns   = reverse_col(state.board.pieces[PAWN]);

    // En passant: convert engine square to Fathom square
    if (state.lastDoublePawnPush != -1) {
        // lastDoublePawnPush is the pawn's destination (rank 4 or 5 in standard terms)
        // Fathom wants the EP capture square
        int engineEpTarget;
        if (state.friendlyColor() == WHITE) {
            // Black just pushed, EP target is rank 6 (index 5)
            engineEpTarget = (state.lastDoublePawnPush & 7) + 5 * 8;
        } else {
            // White just pushed, EP target is rank 3 (index 2)
            engineEpTarget = (state.lastDoublePawnPush & 7) + 2 * 8;
        }
        ep = engineEpTarget ^ 7;  // convert to Fathom square
    } else {
        ep = 0;
    }

    turn = (state.friendlyColor() == WHITE);
}

int TablebaseProbe::probeWDL(const GameState& state) const {
    if (!initialized) return TB_RESULT_INVALID;

    // Cannot probe with castling rights
    if (state.castlingMask) {
        return TB_RESULT_INVALID;
    }

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
    stateToFathom(state, white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn);

    unsigned result = tb_probe_wdl(white, black, kings, queens, rooks, bishops,
                                    knights, pawns, state.rule50_count(), 0, ep, turn);

    if (result == TB_RESULT_FAILED) return TB_RESULT_INVALID;
    return static_cast<int>(result);
}

int TablebaseProbe::probeRoot(const GameState& state, Move& bestMove) const {
    if (!initialized) return TB_RESULT_INVALID;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
    stateToFathom(state, white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn);

    unsigned results[TB_MAX_MOVES];
    unsigned result = tb_probe_root(white, black, kings, queens, rooks, bishops,
                                     knights, pawns, state.rule50_count(),
                                     0, ep, turn, results);

    if (result == TB_RESULT_FAILED) return TB_RESULT_INVALID;
    if (result == TB_RESULT_CHECKMATE) return TB_RESULT_INVALID;
    if (result == TB_RESULT_STALEMATE) return TB_RESULT_INVALID;

    // Extract best move from result (Fathom squares)
    unsigned fathomFrom = TB_GET_FROM(result);
    unsigned fathomTo = TB_GET_TO(result);
    unsigned promo = TB_GET_PROMOTES(result);

    // Convert Fathom squares to engine squares
    int from = fathomFrom ^ 7;
    int to = fathomTo ^ 7;

    // Determine piece type from engine position

    // Handle en passant

    // Convert promotion piece type
    // Fathom: QUEEN=1, ROOK=2, BISHOP=3, KNIGHT=4
    // Engine uses piece type constants from Const.hpp
    int8_t promotion = -1;
    if (promo != TB_PROMOTES_NONE) {
        const int promoMap[] = {-1, QUEEN, ROOK, BISHOP, KNIGHT};
        promotion = promoMap[promo];
    }

    // Construct the move
    bestMove.moveInfo = 0;  // Clear first
    bestMove.moveInfo |= (int16_t)(to);
    bestMove.moveInfo |= (int16_t)(from << 6);
    if (promotion != -1) {
        bestMove.updatePromotion(promotion);
    }
    if (TB_GET_EP(result)) {
        bestMove.setFlag(Move::fep);
    }

    return TB_GET_WDL(result);
}

int TablebaseProbe::rootFiltering(const GameState& state, Move* moves, int& nbMoves) const {
    if (!initialized) return TB_RESULT_INVALID;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
    stateToFathom(state, white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn);

    TbRootMoves results;
    int ok = tb_probe_root_dtz(white, black, kings, queens, rooks, bishops,
                                knights, pawns, state.rule50_count(),
                                0, ep, turn, true, true, &results);
    if (!ok || results.size == 0){    
        ok = tb_probe_root_wdl(white, black, kings, queens, rooks, bishops,
                                knights, pawns, state.rule50_count(),
                                0, ep, turn, true, &results);
        if (!ok || results.size == 0) return TB_RESULT_INVALID;
    }
    // Find the best (highest) rank across all moves.
    // WdlToRank[] in Fathom: LOSS=-1000, BLESSED_LOSS=-899, DRAW=0, CURSED_WIN=899, WIN=1000
    int32_t bestRank = results.moves[0].tbRank;
    for (unsigned i = 1; i < results.size; i++) {
        if (results.moves[i].tbRank > bestRank)
            bestRank = results.moves[i].tbRank;
    }
    // Map best rank back to TB_RESULT_* constant
    int wdl;
    int lowerbound;
    if      (bestRank >= 900) wdl = TB_RESULT_WIN, lowerbound = 900;
    else if (bestRank >= 100)  wdl = TB_RESULT_CURSED_WIN, lowerbound = 100;
    else if (bestRank > -100)  wdl = TB_RESULT_DRAW, lowerbound = -100;
    else if (bestRank > -900) wdl = TB_RESULT_BLESSED_LOSS, lowerbound = -899;
    else                       wdl = TB_RESULT_LOSS, lowerbound = -1000;

    // Promotion mapping: Fathom (NONE=0,Q=1,R=2,B=3,N=4) -> engine piece type
    const int promoMap[] = {0, QUEEN, ROOK, BISHOP, KNIGHT};

    // Filter moves[] in-place: keep only moves matching a best-rank TbRootMove
    int newNb = 0;
    for (int i = 0; i < nbMoves; i++) {
        int engineFrom = moves[i].from();
        int engineTo   = moves[i].to();
        int engineProm = moves[i].promotion();

        bool keep = false;
        for (unsigned j = 0; j < results.size; j++) {
            if (results.moves[j].tbRank < lowerbound) continue;

            unsigned fathomFrom = TB_MOVE_FROM(results.moves[j].move);
            unsigned fathomTo   = TB_MOVE_TO(results.moves[j].move);
            unsigned fathomProm = TB_MOVE_PROMOTES(results.moves[j].move);

            if ((int)(fathomFrom ^ 7) != engineFrom) continue;
            if ((int)(fathomTo   ^ 7) != engineTo)   continue;
            if (promoMap[fathomProm]   != engineProm) continue;

            keep = true;
            break;
        }
        if (keep)
            moves[newNb++] = moves[i];
    }

    // Safety: if nothing matched (encoding mismatch), leave moves unchanged
    if (newNb == 0) return TB_RESULT_INVALID;

    nbMoves = newNb;
    return wdl;
}

int TablebaseProbe::wdlToScore(int wdl, int ply) {
    switch (wdl) {
        case TB_RESULT_WIN:
            return TB_WIN_SCORE - ply;  // Prefer faster wins
        case TB_RESULT_CURSED_WIN:
            return TB_CURSED_WIN_SCORE;
        case TB_RESULT_DRAW:
            return 0;
        case TB_RESULT_BLESSED_LOSS:
            return TB_BLESSED_LOSS_SCORE;
        case TB_RESULT_LOSS:
            return -TB_WIN_SCORE + ply;  // Prefer slower losses
        default:
            return 0;
    }
}
