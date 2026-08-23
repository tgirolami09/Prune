#include "Evaluator.hpp"
#include <assert.h>
#include <algorithm>
#include <cstring>
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "NNUE.hpp"
#include "TablebaseProbe.hpp"
#include "tunables.hpp"
#ifdef DEBUG_MACRO
#include "stats_helpers.hpp"
StatVar<big, 48 * 1024, 0> matScalingStats;
#endif

big get_rook_lines(big occupancy, int square) {
    return moves_table(square + 64, occupancy, mask_empty_rook(square));
}
big get_bishop_lines(big occupancy, int square) {
    return moves_table(square, occupancy, mask_empty_bishop(square));
}

inline int getLVA(int square, const GameState& state, bool stm, big occupancy,
                  int& pieceType) {  // return the square where the lva come
                                     // from, set pieceType
    // Pawns
    big mask = occupancy & state.board.getMask(PAWN, stm) & attackPawns[(!stm) * 64 + square];
    if (mask) {
        pieceType = PAWN;
        return __builtin_ctzll(mask);
    }
    // Knight
    mask = occupancy & state.board.getMask(KNIGHT, stm) & KnightMoves[square];
    if (mask) {
        pieceType = KNIGHT;
        return __builtin_ctzll(mask);
    }
    // Bishop
    big maskB = occupancy & get_bishop_lines(occupancy, square);
    mask = state.board.getMask(BISHOP, stm) & maskB;
    if (mask) {
        pieceType = BISHOP;
        return __builtin_ctzll(mask);
    }
    // Rook
    big maskR = occupancy & get_rook_lines(occupancy, square);
    mask = state.board.getMask(ROOK, stm) & maskR;
    if (mask) {
        pieceType = ROOK;
        return __builtin_ctzll(mask);
    }
    // Queen
    mask = state.board.getMask(QUEEN, stm) & (maskR | maskB);
    if (mask) {
        pieceType = QUEEN;
        return __builtin_ctzll(mask);
    }
    // KING
    mask = occupancy & state.board.getMask(KING, stm) & normalKingMoves[square];
    if (mask) {
        pieceType = KING;
        return __builtin_ctzll(mask);
    }
    return -1;
}

int fastSEE(const Move& move, const GameState& state, const int* value_pieces) {
    big occupancy = state.board.colors[WHITE] | state.board.colors[BLACK];
    occupancy ^= 1ULL << move.from();
    int square = move.to();
    bool stm = !state.friendlyColor();
    int atk;
    int pieceType;
    ubyte stack[16];
    int idStack = 0;
    int lastPiece = state.getPiece(move.from());
    while ((atk = getLVA(square, state, stm, occupancy, pieceType)) != -1) {
        stack[idStack++] = lastPiece;
        occupancy ^= 1ULL << atk;
        lastPiece = pieceType;
        stm = !stm;
    }
    int res = 0;
    idStack--;
    for (; idStack >= 0; idStack--) {
        res = max(0, value_pieces[stack[idStack]] - res);
    }
    return res;
}

big get_mask(const GameState& state, int p) {
    return state.board.pieces[p];
}

big firstTouch(int square, int square2, big occupancy) {
    big mask = fullDir[square][square2] & occupancy;
    if (!mask)
        return 0;
    if (square2 > square)
        return mask & -mask;
    else
        return 1ULL << (__builtin_clzll(mask) ^ 63);
}

bool see_ge(int born, const Move& move, const GameState& state, const int* value_pieces) {
    if (move.getMovePart() == Move::fcastle)
        return born < 0;
    int square = move.to();
    // occupancy ^= 1ULL << move.from();
    bool stm = state.friendlyColor();
    int atk = move.from();
    int lastPiece = state.board.getCapture(move);
    int pieceType = state.getPiece(move.from());
    bool sstm = stm;
    const big diagPieces = state.board.pieces[BISHOP] | state.board.pieces[QUEEN];
    const big hvPieces = state.board.pieces[ROOK] | state.board.pieces[QUEEN];
    big occupancy = state.board.occupancy() ^ (1ULL << atk);
    born = value_pieces[lastPiece] - born;
    stm = !stm;
    lastPiece = pieceType;
    if (born < 0)
        return false;
    big bishopAtk = mask_empty_bishop(square);
    big attacks = ((get_bishop_lines(occupancy, square) & diagPieces) |
                   (get_rook_lines(occupancy, square) & hvPieces) |
                   (KnightMoves[square] & state.board.pieces[KNIGHT]) |
                   (attackPawns[square] & state.board.getMask(PAWN, 1)) |
                   (attackPawns[square + 64] & state.board.getMask(PAWN, 0)) |
                   (normalKingMoves[square] & state.board.pieces[KING])) &
                  occupancy;
    bool begin2first = false;
    bool begin2second = false;
    big sideAtks;
    while ((sideAtks = (attacks & state.board.colors[stm]))) {
        pieceType = -1;
        for (int p = begin2first * 2; p < nbPieces; p++) {
            big mask = state.board.pieces[p] & sideAtks;
            if (mask) {
                atk = __builtin_ctzll(mask);
                pieceType = p;
                break;
            }
        }
        if ((pieceType == KING && (attacks & (attacks - 1))))
            break;
        occupancy ^= 1ULL << atk;
        born = value_pieces[lastPiece] - born;
        stm = !stm;
        lastPiece = pieceType;
        if (stm == sstm) {
            if (born <= 0)
                return true;
        } else if (born < 0)
            return false;
        if (pieceType == KING)
            break;
        begin2first = begin2second;
        begin2second = pieceType > KNIGHT;
        if (pieceType == QUEEN) {
            if ((1ULL << atk) & bishopAtk)
                attacks |= firstTouch(square, atk, occupancy) & diagPieces;
            else
                attacks |= firstTouch(square, atk, occupancy) & hvPieces;
        } else if (pieceType == ROOK)
            attacks |= firstTouch(square, atk, occupancy) & hvPieces;
        else if (pieceType != KNIGHT)
            attacks |= firstTouch(square, atk, occupancy) & diagPieces;
        attacks &= occupancy;
    }
    // printf("%d %d %d\n", stm, sstm, born);
    return stm != sstm || born <= 0;
}

int score_move(const Move& move, int historyScore, const GameState& state,
               const int* value_pieces) {
    int score = 0;
    if (state.board.isTactical(move)) {
        int cap = state.board.getCapture(move);
        if (cap != SPACE)
            score += cap * 6;
        if (move.getFlag() == Move::fpromo)
            score += move.promotion();
        score *= maxHistory * 2;
        if (see_ge(0, move, state, value_pieces))
            score |= 1 << 28;
        score |= 2 << 28;
    }
    score += historyScore + maxHistory;
    return score;
}

void IncrementalEvaluator::print() {
    printf("phase = %d\n", mgPhase);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            printf("piece = %d, color = %d, nbPieces = %d\n", j, i, presentPieces[i][j]);
        }
    }
}

IncrementalEvaluator::IncrementalEvaluator() {
    memset(presentPieces, 0, sizeof(presentPieces));
}

void IncrementalEvaluator::init(
    const GameState& state,
    const NNUE& nnue) {  // should be only call at the start of the search
    mgPhase = 0;
    nbMan = 0;
    stackIndex = 0;
    nnue.initAcc(stackAcc[stackIndex]);
    finny.init(nnue);
    stackAcc[stackIndex].update.nbThreats[0] = 0;
    stackAcc[stackIndex].update.nbThreats[1] = 0;
    stackAcc[stackIndex].update.dirty = false;
    stackAcc[stackIndex].Kside[WHITE] = col(__builtin_ctzll(state.board.getMask(KING, WHITE))) <= 3;
    stackAcc[stackIndex].Kside[BLACK] = col(__builtin_ctzll(state.board.getMask(KING, BLACK))) <= 3;
    stackAcc[stackIndex].idInputBucket[WHITE] =
        getInputBucket(__builtin_ctzll(state.board.getMask(KING, WHITE)), WHITE,
                       stackAcc[stackIndex].Kside[WHITE]);
    stackAcc[stackIndex].idInputBucket[BLACK] =
        getInputBucket(__builtin_ctzll(state.board.getMask(KING, BLACK)), BLACK,
                       stackAcc[stackIndex].Kside[BLACK]);
    memcpy(&stackAcc[stackIndex].board, &state.board, sizeof(stackAcc[stackIndex].board));
    nnue.calcThreats(stackAcc[stackIndex], WHITE, state.board);
    nnue.calcThreats(stackAcc[stackIndex], BLACK, state.board);
    // printf("%d %d\n", stackAcc[stackIndex].idInputBucket[WHITE],
    // stackAcc[stackIndex].idInputBucket[BLACK]);
    memset(presentPieces, 0, sizeof(presentPieces));
    for (int square = 0; square < 64; square++) {
        int piece = state.getfullPiece(square);
        if (type(piece) != SPACE) {
            changePiece<1, true>(nnue, square, type(piece), color(piece));
            // printf("intermediate eval : %d\n",
            // getScore(state.friendlyColor()));
        }
    }
}

bool IncrementalEvaluator::isInsufficientMaterial() const {
    if (mgPhase <= 1 && !presentPieces[WHITE][PAWN] && !presentPieces[BLACK][PAWN]) {
        return true;
    }
    return false;
}

bool IncrementalEvaluator::isOnlyPawns() const {
    return !mgPhase;
}

int IncrementalEvaluator::getRaw(bool c, _unused const NNUE& nnue) {
    nnue.updateStack(stackAcc, stackIndex, finny);
    return nnue.eval(stackAcc[stackIndex], c, (nbMan - 2) / DIVISOR);
}

int IncrementalEvaluator::getScore(bool c, const corrhists& ch, const GameState& state,
                                   const tunables& parameters, const NNUE& nnue) {
    int raw_eval = getRaw(c, nnue);
    return correctEval(raw_eval, ch, state, parameters);
}
int IncrementalEvaluator::correctEval(int raw_eval, const corrhists& ch, const GameState& state,
                                      _unused const tunables& parameters) const {
    raw_eval += ch.probe(state);
#if !defined(DATAGEN)
    int nbQ = presentPieces[WHITE][QUEEN] + presentPieces[BLACK][QUEEN];
    int nbR = presentPieces[WHITE][ROOK] + presentPieces[BLACK][ROOK];
    int nbB = presentPieces[WHITE][BISHOP] + presentPieces[BLACK][BISHOP];
    int nbN = presentPieces[WHITE][KNIGHT] + presentPieces[BLACK][KNIGHT];
    int nbP = presentPieces[WHITE][PAWN] + presentPieces[BLACK][PAWN];
    int mat = nbQ * parameters.mats_queen + nbR * parameters.mats_rook +
              nbB * parameters.mats_bishop + nbN * parameters.mats_knight +
              nbP * parameters.mats_pawn;
#ifdef DEBUG_MACRO
    matScalingStats.update(mat);
#endif
    int matScaling = raw_eval * (mat + parameters.mats_offset) / (48 * 1024);
    return clamp(matScaling, -TB_WIN_SCORE + 100, TB_WIN_SCORE - 100);
#else
    return clamp(raw_eval, -TB_WIN_SCORE + 100, TB_WIN_SCORE - 100);
#endif
}
void IncrementalEvaluator::undoMove(const NNUE& nnue, Move move, bool c,
                                    const PositionState& state1, const PositionState& state2) {
    playMove<-1>(nnue, move, c, state1, state2);
}

template <int f, bool updateNNUE>
void IncrementalEvaluator::changePiece(_unused const NNUE& nnue, int pos, int piece, bool c,
                                       _unused bool updateNNUE2) {
    if (updateNNUE)
        if (updateNNUE2) {
            Index index(pos, piece, c);
            nnue.change1<f>(stackAcc[stackIndex], WHITE,
                            index.mirror(stackAcc[stackIndex].Kside[WHITE]),
                            stackAcc[stackIndex].idInputBucket[WHITE]);
            nnue.change1<f>(stackAcc[stackIndex], BLACK,
                            index.mirror(stackAcc[stackIndex].Kside[BLACK]).changepov(),
                            stackAcc[stackIndex].idInputBucket[BLACK]);
        }
    mgPhase += f * gamephaseInc[piece];
    nbMan += f;
    presentPieces[c][piece] += f;
}

template <int f, bool updateNNUE>
void IncrementalEvaluator::changePiece2(_unused const NNUE& nnue, int pos, int piece, bool c) {
    if (updateNNUE) {
        Index index(pos, piece, c);
        nnue.change2<f>(stackAcc[stackIndex], stackAcc[stackIndex + 1], WHITE,
                        index.mirror(stackAcc[stackIndex].Kside[WHITE]),
                        stackAcc[stackIndex].idInputBucket[WHITE]);
        nnue.change2<f>(stackAcc[stackIndex], stackAcc[stackIndex + 1], BLACK,
                        index.mirror(stackAcc[stackIndex].Kside[BLACK]).changepov(),
                        stackAcc[stackIndex].idInputBucket[BLACK]);
        stackIndex++;
    } else {
        stackIndex--;
    }
    mgPhase += f * gamephaseInc[piece];
    nbMan += f;
    presentPieces[c][piece] += f;
}

template <int f>
void IncrementalEvaluator::playMove(const NNUE& nnue, Move move, bool c,
                                    _unused const PositionState& state1,
                                    _unused const PositionState& state2) {
    static_assert(f == -1 || f == 1, "f has to be either -1 or 1");
    const int piece = type(state1.mailbox[move.from()]);
    const int toPiece = piece | move.promotion();
    const int capture = state1.getCapture(move);
    const int toSquare = move.toMover();
    if (move.getFlag() == Move::fpromo) {
        changePiece<-f, false>(nnue, move.from(), piece, c);
        changePiece<f, false>(nnue, move.to(), toPiece, c);
    }
    Index sub1(move.from(), piece, c), add1(toSquare, toPiece, c), sub2, add2;
    bool mirror = false;
    if (capture != SPACE) {
        int posCapture = move.to();
        int pieceCapture = capture;
        if (move.getFlag() == Move::fep) {
            if (c == WHITE)
                posCapture -= 8;
            else
                posCapture += 8;
            pieceCapture = PAWN;
        }
        changePiece<-f, false>(nnue, posCapture, pieceCapture, !c);
        if (f == 1)
            sub2 = Index(posCapture, pieceCapture, !c);
    }
    if (piece == KING) {
        if ((col(move.from()) > 3) != (col(toSquare) > 3))
            mirror = true;
        if (move.getFlag() == Move::fcastle) {  // castling
            int rookStart = move.to();
            int rookEnd = toSquare + 2 * (move.from() > move.to()) - 1;
            if (f == 1)
                sub2 = Index(rookStart, ROOK, c), add2 = Index(rookEnd, ROOK, c);
        }
    }
    if (f == 1) {
        stackAcc[stackIndex + 1].reinit(move, state1, state2, stackAcc[stackIndex], c, mirror, sub1,
                                        add1, sub2, add2);
        stackIndex++;
    } else
        stackIndex--;
}

void IncrementalEvaluator::backStack() {
    stackIndex--;
}

void IncrementalEvaluator::playNoBack(_unused const GameState& state, Move move, bool c,
                                      _unused const NNUE& nnue) {
    int piece = state.getPiece(move.from());
    int toPiece = piece | move.promotion();  // for promotion
    int capture = state.board.getCapture(move);
    int toSquare = move.toMover();
    bool mirror = false;
    if (piece == KING && (col(move.from()) > 3) != (col(toSquare) > 3))
        mirror = true;
    changePiece<-1, true>(nnue, move.from(), piece, c, !mirror);
    changePiece<1, true>(nnue, toSquare, toPiece, c, !mirror);
    if (capture != SPACE) {
        int posCapture = move.to();
        int pieceCapture = capture;
        if (move.getFlag() == Move::fep) {  // for en passant
            if (c == WHITE)
                posCapture -= 8;
            else
                posCapture += 8;
            pieceCapture = PAWN;
        }
        changePiece<-1, true>(nnue, posCapture, pieceCapture, !c, !mirror);
    }
    if (move.getFlag() == Move::fcastle) {  // castling
        int rookStart = move.to();
        int rookEnd = toSquare + 2 * (move.from() > move.to()) - 1;
        changePiece<-1, true>(nnue, rookStart, ROOK, c, !mirror);
        changePiece<1, true>(nnue, rookEnd, ROOK, c, !mirror);
    }
    if (mirror) {
        stackAcc[stackIndex].Kside[state.enemyColor()] ^= 1;
        init(state, nnue);
    }
}
const Accumulator& IncrementalEvaluator::operator[](int idx) const {
    return stackAcc[idx];
}

template void IncrementalEvaluator::playMove<-1>(const NNUE&, Move, bool, const PositionState&,
                                                 const PositionState&);
template void IncrementalEvaluator::playMove<1>(const NNUE&, Move, bool, const PositionState&,
                                                const PositionState&);
template void IncrementalEvaluator::changePiece2<-1, true>(const NNUE&, int, int, bool);
template void IncrementalEvaluator::changePiece2<1, true>(const NNUE&, int, int, bool);
template void IncrementalEvaluator::changePiece2<-1, false>(const NNUE&, int, int, bool);
template void IncrementalEvaluator::changePiece2<1, false>(const NNUE&, int, int, bool);
