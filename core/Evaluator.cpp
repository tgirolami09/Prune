#include "Evaluator.hpp"
#include <assert.h>
#include <algorithm>
#include <bit>
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
StatVar<u64, 48 * 1024, 0> matScalingStats;
#endif

u64 get_rook_lines(u64 occupancy, int square) {
    return moves_table(square + 64, occupancy, mask_empty_rook(square));
}
u64 get_bishop_lines(u64 occupancy, int square) {
    return moves_table(square, occupancy, mask_empty_bishop(square));
}

inline int getLVA(int square, const GameState& state, bool stm, u64 occupancy,
                  int& pieceType) {  // return the square where the lva come
                                     // from, set pieceType
    // Pawns
    u64 mask = occupancy & state.board.getMask(Pawn, stm) & attackPawns[(!stm) * 64 + square];
    if (mask) {
        pieceType = Pawn;
        return __builtin_ctzll(mask);
    }
    // Knight
    mask = occupancy & state.board.getMask(Knight, stm) & KnightMoves[square];
    if (mask) {
        pieceType = Knight;
        return __builtin_ctzll(mask);
    }
    // Bishop
    u64 maskB = occupancy & get_bishop_lines(occupancy, square);
    mask = state.board.getMask(Bishop, stm) & maskB;
    if (mask) {
        pieceType = Bishop;
        return __builtin_ctzll(mask);
    }
    // Rook
    u64 maskR = occupancy & get_rook_lines(occupancy, square);
    mask = state.board.getMask(Rook, stm) & maskR;
    if (mask) {
        pieceType = Rook;
        return __builtin_ctzll(mask);
    }
    // Queen
    mask = state.board.getMask(Queen, stm) & (maskR | maskB);
    if (mask) {
        pieceType = Queen;
        return __builtin_ctzll(mask);
    }
    // King
    mask = occupancy & state.board.getMask(King, stm) & normalKingMoves[square];
    if (mask) {
        pieceType = King;
        return __builtin_ctzll(mask);
    }
    return -1;
}

int fastSEE(const Move& move, const GameState& state, const int* value_pieces) {
    u64 occupancy = state.board.colors[White] | state.board.colors[Black];
    occupancy ^= 1ULL << move.from();
    int square = move.to();
    bool stm = !state.friendlyColor();
    int atk;
    int pieceType;
    u8 stack[16];
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

u64 get_mask(const GameState& state, int p) {
    return state.board.pieces[p];
}

u64 firstTouch(int square, int square2, u64 occupancy) {
    u64 mask = fullDir[square][square2] & occupancy;
    if (!mask)
        return 0;
    if (square2 > square)
        return mask & -mask;
    else
        return 1ULL << (__builtin_clzll(mask) ^ 63);
}

bool see_ge(int born, const Move& move, const GameState& state, const int* value_pieces) {
    if (move.getFlag() == Move::fcastle)
        return born < 0;
    int square = move.to();
    // occupancy ^= 1ULL << move.from();
    bool stm = state.friendlyColor();
    int atk = move.from();
    int lastPiece = state.board.getCapture(move);
    int pieceType = state.getPiece(move.from());
    bool sstm = stm;
    const u64 diagPieces = state.board.pieces[Bishop] | state.board.pieces[Queen];
    const u64 hvPieces = state.board.pieces[Rook] | state.board.pieces[Queen];
    u64 occupancy = state.board.occupancy() ^ (1ULL << atk);
    born = value_pieces[lastPiece] - born;
    stm = !stm;
    lastPiece = pieceType;
    if (born < 0)
        return false;
    u64 bishopAtk = mask_empty_bishop(square);
    u64 attacks = ((get_bishop_lines(occupancy, square) & diagPieces) |
                   (get_rook_lines(occupancy, square) & hvPieces) |
                   (KnightMoves[square] & state.board.pieces[Knight]) |
                   (attackPawns[square] & state.board.getMask(Pawn, 1)) |
                   (attackPawns[square + 64] & state.board.getMask(Pawn, 0)) |
                   (normalKingMoves[square] & state.board.pieces[King])) &
                  occupancy;
    bool begin2first = false;
    bool begin2second = false;
    u64 sideAtks;
    while ((sideAtks = (attacks & state.board.colors[stm]))) {
        pieceType = -1;
        for (int p = begin2first * 2; p < nbPieces; p++) {
            u64 mask = state.board.pieces[p] & sideAtks;
            if (mask) {
                atk = __builtin_ctzll(mask);
                pieceType = p;
                break;
            }
        }
        if ((pieceType == King && (attacks & (attacks - 1))))
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
        if (pieceType == King)
            break;
        begin2first = begin2second;
        begin2second = pieceType > Knight;
        if (pieceType == Queen) {
            if ((1ULL << atk) & bishopAtk)
                attacks |= firstTouch(square, atk, occupancy) & diagPieces;
            else
                attacks |= firstTouch(square, atk, occupancy) & hvPieces;
        } else if (pieceType == Rook)
            attacks |= firstTouch(square, atk, occupancy) & hvPieces;
        else if (pieceType != Knight)
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
        if (cap != Void)
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
}

IncrementalEvaluator::IncrementalEvaluator() {}

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
    stackAcc[stackIndex].Kside[White] = col(__builtin_ctzll(state.board.getMask(King, White))) <= 3;
    stackAcc[stackIndex].Kside[Black] = col(__builtin_ctzll(state.board.getMask(King, Black))) <= 3;
    stackAcc[stackIndex].idInputBucket[White] =
        getInputBucket(__builtin_ctzll(state.board.getMask(King, White)), White,
                       stackAcc[stackIndex].Kside[White]);
    stackAcc[stackIndex].idInputBucket[Black] =
        getInputBucket(__builtin_ctzll(state.board.getMask(King, Black)), Black,
                       stackAcc[stackIndex].Kside[Black]);
    memcpy(&stackAcc[stackIndex].board, &state.board, sizeof(stackAcc[stackIndex].board));
    nnue.calcThreats(stackAcc[stackIndex], White, state.board);
    nnue.calcThreats(stackAcc[stackIndex], Black, state.board);
    // printf("%d %d\n", stackAcc[stackIndex].idInputBucket[White],
    // stackAcc[stackIndex].idInputBucket[Black]);
    for (int square = 0; square < 64; square++) {
        int piece = state.getfullPiece(square);
        if (type(piece) != Void) {
            changePiece<1, true>(nnue, square, type(piece), color(piece));
            // printf("intermediate eval : %d\n",
            // getScore(state.friendlyColor()));
        }
    }
}

bool IncrementalEvaluator::isInsufficientMaterial(const GameState& state) const {
    if (mgPhase <= 1 && !state.board.pieces[Pawn]) {
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
    int nbQ = popcount(state.board.pieces[Queen]);
    int nbR = popcount(state.board.pieces[Rook]);
    int nbB = popcount(state.board.pieces[Bishop]);
    int nbN = popcount(state.board.pieces[Knight]);
    int nbP = popcount(state.board.pieces[Pawn]);
    int mat = nbQ * parameters.mats_queen + nbR * parameters.mats_rook +
              nbB * parameters.mats_bishop + nbN * parameters.mats_knight +
              nbP * parameters.mats_pawn;
#ifdef DEBUG_MACRO
    matScalingStats.update(mat);
#endif
    int matScaling = raw_eval * (mat + parameters.mats_offset) / (48 * 1024);
    matScaling = matScaling * (200 - state.rule50_count()) / 200;
    return clamp(matScaling, -TB_WIN_SCORE + 100, TB_WIN_SCORE - 100);
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
            nnue.change1<f>(stackAcc[stackIndex], White,
                            (int)index.mirror(stackAcc[stackIndex].Kside[White]),
                            stackAcc[stackIndex].idInputBucket[White]);
            nnue.change1<f>(stackAcc[stackIndex], Black,
                            (int)index.mirror(stackAcc[stackIndex].Kside[Black]).changepov(),
                            stackAcc[stackIndex].idInputBucket[Black]);
        }
    mgPhase += f * gamephaseInc[piece];
    nbMan += f;
}

template <int f, bool updateNNUE>
void IncrementalEvaluator::changePiece2(_unused const NNUE& nnue, int pos, int piece, bool c) {
    if (updateNNUE) {
        Index index(pos, piece, c);
        nnue.change2<f>(stackAcc[stackIndex], stackAcc[stackIndex + 1], White,
                        (int)index.mirror(stackAcc[stackIndex].Kside[White]),
                        stackAcc[stackIndex].idInputBucket[White]);
        nnue.change2<f>(stackAcc[stackIndex], stackAcc[stackIndex + 1], Black,
                        (int)index.mirror(stackAcc[stackIndex].Kside[Black]).changepov(),
                        stackAcc[stackIndex].idInputBucket[Black]);
        stackIndex++;
    } else {
        stackIndex--;
    }
    mgPhase += f * gamephaseInc[piece];
    nbMan += f;
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
    if (capture != Void) {
        int posCapture = move.to();
        int pieceCapture = capture;
        if (move.getFlag() == Move::fep) {
            if (c == White)
                posCapture -= 8;
            else
                posCapture += 8;
            pieceCapture = Pawn;
        }
        changePiece<-f, false>(nnue, posCapture, pieceCapture, !c);
        if (f == 1)
            sub2 = Index(posCapture, pieceCapture, !c);
    }
    if (piece == King) {
        if ((col(move.from()) > 3) != (col(toSquare) > 3))
            mirror = true;
        if (move.getFlag() == Move::fcastle) {  // castling
            int rookStart = move.to();
            int rookEnd = toSquare + 2 * (move.from() > move.to()) - 1;
            if (f == 1)
                sub2 = Index(rookStart, Rook, c), add2 = Index(rookEnd, Rook, c);
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
    if (piece == King && (col(move.from()) > 3) != (col(toSquare) > 3))
        mirror = true;
    changePiece<-1, true>(nnue, move.from(), piece, c, !mirror);
    changePiece<1, true>(nnue, toSquare, toPiece, c, !mirror);
    if (capture != Void) {
        int posCapture = move.to();
        int pieceCapture = capture;
        if (move.getFlag() == Move::fep) {  // for en passant
            if (c == White)
                posCapture -= 8;
            else
                posCapture += 8;
            pieceCapture = Pawn;
        }
        changePiece<-1, true>(nnue, posCapture, pieceCapture, !c, !mirror);
    }
    if (move.getFlag() == Move::fcastle) {  // castling
        int rookStart = move.to();
        int rookEnd = toSquare + 2 * (move.from() > move.to()) - 1;
        changePiece<-1, true>(nnue, rookStart, Rook, c, !mirror);
        changePiece<1, true>(nnue, rookEnd, Rook, c, !mirror);
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
