#include "TablebaseProbe.hpp"
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
    probeDepth = depth;
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
    int count = 0;
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < 6; p++) {
            count += countbit(state.boardRepresentation[c][p]);
        }
    }
    return count;
}

bool TablebaseProbe::canProbe(const GameState& state) const {
    if (!initialized || tbLargest == 0) return false;

    int pieceCount = countPieces(state);
    if (pieceCount > (int)tbLargest || pieceCount > probeLimit) return false;

    // Cannot probe with castling rights
    if (state.castlingRights[WHITE][0] || state.castlingRights[WHITE][1] ||
        state.castlingRights[BLACK][0] || state.castlingRights[BLACK][1]) {
        return false;
    }

    return true;
}

// Helper function to convert GameState to Fathom format
static void stateToFathom(const GameState& state,
                          uint64_t& white, uint64_t& black,
                          uint64_t& kings, uint64_t& queens, uint64_t& rooks,
                          uint64_t& bishops, uint64_t& knights, uint64_t& pawns,
                          unsigned& ep, bool& turn) {
    // Combine color bitboards and convert to Fathom format using reverse_col
    white = reverse_col(
            state.boardRepresentation[WHITE][PAWN] |
            state.boardRepresentation[WHITE][KNIGHT] |
            state.boardRepresentation[WHITE][BISHOP] |
            state.boardRepresentation[WHITE][ROOK] |
            state.boardRepresentation[WHITE][QUEEN] |
            state.boardRepresentation[WHITE][KING]);

    black = reverse_col(
            state.boardRepresentation[BLACK][PAWN] |
            state.boardRepresentation[BLACK][KNIGHT] |
            state.boardRepresentation[BLACK][BISHOP] |
            state.boardRepresentation[BLACK][ROOK] |
            state.boardRepresentation[BLACK][QUEEN] |
            state.boardRepresentation[BLACK][KING]);

    // Combine piece type bitboards and convert to Fathom format
    kings   = reverse_col(state.boardRepresentation[WHITE][KING]   | state.boardRepresentation[BLACK][KING]);
    queens  = reverse_col(state.boardRepresentation[WHITE][QUEEN]  | state.boardRepresentation[BLACK][QUEEN]);
    rooks   = reverse_col(state.boardRepresentation[WHITE][ROOK]   | state.boardRepresentation[BLACK][ROOK]);
    bishops = reverse_col(state.boardRepresentation[WHITE][BISHOP] | state.boardRepresentation[BLACK][BISHOP]);
    knights = reverse_col(state.boardRepresentation[WHITE][KNIGHT] | state.boardRepresentation[BLACK][KNIGHT]);
    pawns   = reverse_col(state.boardRepresentation[WHITE][PAWN]   | state.boardRepresentation[BLACK][PAWN]);

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
    if (state.castlingRights[WHITE][0] || state.castlingRights[WHITE][1] ||
        state.castlingRights[BLACK][0] || state.castlingRights[BLACK][1]) {
        return TB_RESULT_INVALID;
    }

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
    stateToFathom(state, white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn);

    unsigned result = tb_probe_wdl(white, black, kings, queens, rooks, bishops,
                                    knights, pawns, 0, 0, ep, turn);

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
    int piece = PAWN;  // Default
    int friendlyColor = state.friendlyColor();
    for (int p = 0; p < 6; p++) {
        if (state.boardRepresentation[friendlyColor][p] & (1ULL << from)) {
            piece = p;
            break;
        }
    }
    int capture = -2;  // No capture by default

    // Check if there's a capture (using engine squares)
    int oppColor = 1 - state.friendlyColor();
    for (int p = 0; p < 6; p++) {
        if (state.boardRepresentation[oppColor][p] & (1ULL << to)) {
            capture = p;
            break;
        }
    }

    // Handle en passant
    if (TB_GET_EP(result)) {
        capture = -1;  // Engine's EP indicator
    }

    // Convert promotion piece type
    // Fathom: QUEEN=1, ROOK=2, BISHOP=3, KNIGHT=4
    // Engine uses piece type constants from Const.hpp
    int8_t promotion = -1;
    if (promo != TB_PROMOTES_NONE) {
        const int promoMap[] = {-1, QUEEN, ROOK, BISHOP, KNIGHT};
        promotion = promoMap[promo];
    }

    // Construct the move
    bestMove.piece = piece;
    bestMove.capture = capture;
    bestMove.moveInfo = -4096;  // Clear first
    bestMove.moveInfo |= (int16_t)(to);
    bestMove.moveInfo |= (int16_t)(from << 6);
    if (promotion != -1) {
        bestMove.moveInfo &= ~(-4096);
        bestMove.moveInfo |= (int16_t)(promotion << 12);
    }

    return TB_GET_WDL(result);
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
