#include "LegalMoveGenerator.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include <cstring>
#include <cassert>
using namespace std;

big KnightMoves[64]; //Knight moves for each position of the board
big pieceCastlingMasks[2][2];
big attackCastlingMasks[2][2];
big normalKingMoves[64];
big attackPawns[128];
big directions[64][64];
big fullDir[64][64];
big* tableMagic;
int indexesTable[128];
const constTable* constantsMagic = (const constTable*)magicsData;

void PrecomputeKnightMoveData(){
    const pair<int, int> moves[8] = {
        {-2,  1},
        {-2, -1},
        {-1,  2},
        {-1, -2},
        { 1,  2},
        { 1, -2},
        { 2,  1},
        { 2, -1}};
    for (int row = 0; row<8; ++row){
        for (int col = 0; col<8; ++col){
            int squareIndex = row * 8 + col;
            //Precompute knight moves
            big knightMoveMask = 0;
            for(pair<int, int> move:moves){
                //0 is up and 1 is down
                int square = squareIndex;
                if(row+move.first < 8 && row+move.first >= 0)
                    square += 8*move.first;
                else continue;
                if(col+move.second < 8 && col+move.second >= 0)
                    square += move.second;
                else continue;
                knightMoveMask |= 1ULL << square;
            }
            KnightMoves[squareIndex] = knightMoveMask;
        }
    }  
}
void precomputeCastlingMasks(){
    pieceCastlingMasks[0][1] = 0b00000110;
    pieceCastlingMasks[0][0] = 0b01110000;
    pieceCastlingMasks[1][1] = pieceCastlingMasks[0][1] << 56;
    pieceCastlingMasks[1][0] = pieceCastlingMasks[0][0] << 56;

    attackCastlingMasks[0][1] = 0b00001110;
    attackCastlingMasks[0][0] = 0b00111000;
    attackCastlingMasks[1][1] = attackCastlingMasks[0][1] << 56;
    attackCastlingMasks[1][0] = attackCastlingMasks[0][0] << 56;
}

void precomputeNormlaKingMoves(){
    for (int kingPosition = 0; kingPosition < 64 ; ++ kingPosition){
        big kingEndMask = 0;

        int transitionsRow[2] = {8, -8};
        int transitionsCol[2] = {-1, 1};

        int nbCol=2;
        int nbRow=2;

        if (row(kingPosition)==7){
            swap(transitionsRow[0], transitionsRow[1]);
            nbRow--;
        }else if (row(kingPosition)==0){
            nbRow--;
        }

        if (col(kingPosition)==0){
            swap(transitionsCol[0], transitionsCol[1]);
            nbCol--;
        }else if (col(kingPosition)==7){
            nbCol--;
        }

        for (int i = 0; i < nbRow;++i){
            int trans1 = transitionsRow[i];
            int cardEnd = kingPosition + trans1;
            kingEndMask |= (1ull << cardEnd);
            for (int j = 0; j < nbCol;++j){
                int trans2 = transitionsCol[j];

                int diagEnd = cardEnd + trans2;
                kingEndMask |= (1ull << diagEnd);

                int secondCardEnd = kingPosition + trans2;
                kingEndMask |= (1ull << secondCardEnd);
            }
        }
        normalKingMoves[kingPosition] = kingEndMask;
    }
}

void precomputePawnsAttack(){
    for(int c=0; c<2; c++){
        int leftLimit = c ?  7 : 0;
        int rightLimit = leftLimit^7;
        int moveFactor = c ? -1 : 1;
        for(int square=0; square<64; square++){
            int key = c*64+square;
            attackPawns[key] = 0;
            if(square+8*moveFactor < 0 || square+8*moveFactor >= 64)continue;
            int pieceCol = col(square);
            if(pieceCol != leftLimit)
                attackPawns[key] |= 1ull<<(square + 7 * moveFactor);
            if(pieceCol != rightLimit)
                attackPawns[key] |= 1ull<<(square + 9 * moveFactor);
        }
    }
}
void precomputeDirections(){
    //Set everything to 0 first just to be sure
    for (int i = 0; i < 64; ++i){
        for (int j = 0; j < 64; ++j){
            directions[i][j] = 0;
            fullDir[i][j] = 0;
        }    
    }
    for(int row=0; row<8; row++){
        for(int col=0; col<8; col++){
            int square = row*8+col;
            for(int idDir=0; idDir<8; idDir++){
                int r=row+dirs[idDir][0];
                int c=col+dirs[idDir][1];
                big mask = 0;
                while(r >= 0 && r < 8 && c >= 0 && c < 8){
                    int sq = (r*8+c);
                    mask |= 1ULL << sq;
                    directions[square][sq] = mask; // line of 1 between square and sq
                    r += dirs[idDir][0];
                    c += dirs[idDir][1];
                }
                r=row+dirs[idDir][0];
                c=col+dirs[idDir][1];
                while(r >= 0 && r < 8 && c >= 0 && c < 8){
                    int sq = (r*8+c);
                    fullDir[square][sq] = mask; // line of 1 from square in the direction of sq
                    r += dirs[idDir][0];
                    c += dirs[idDir][1];
                }
            }
        }
    }
}

big go_dir(big mask, int square, int dir, big clipped){
    int cur_square = square;
    big cur_mask=0;
    do{
        cur_square += dir;
        if(cur_square < 0 || cur_square >= 64)break;
        big p = 1ULL << cur_square;
        cur_mask |= p;
        if((clipped&p) == 0 || (p&mask) != 0)
            break;
    }while(1);
    return cur_mask;
}

big usefull_rook(big mask, int square){
    int col = square&7;
    int row = square >> 3;
    big cur_mask = 0;
    if(col != 7)
        cur_mask |= go_dir(mask, square, 1, clipped_bcol);
    if(col != 0)
        cur_mask |= go_dir(mask, square, -1, clipped_bcol);
    if(row != 7)
        cur_mask |= go_dir(mask, square, 8, clipped_brow);
    if(row != 0)
        cur_mask |= go_dir(mask, square, -8, clipped_brow);
    return cur_mask;
}

big usefull_bishop(big mask, int square){
    big cur_mask=0;
    int col=square&7;
    int row=square >> 3;
    if(col != 0){
        if(row != 7)
            cur_mask |= go_dir(mask, square, +7, clipped_mask);
        if(row != 0)
            cur_mask |= go_dir(mask, square, -9, clipped_mask);
    }if(col != 7){
        if(row != 7)
            cur_mask |= go_dir(mask, square, +9, clipped_mask);
        if(row != 0)
            cur_mask |= go_dir(mask, square, -7, clipped_mask);
    }
    return cur_mask&(~(1ULL << square));
}

static big get_usefull(bool is_rook, big mask, int square){
    return (is_rook?usefull_rook:usefull_bishop)(mask, square);
}

static big apply_id(big id, big mask){
    big new_mask=0;
    while(mask){
        int bit=__builtin_ctzll(mask);
        big m=1ULL << bit;
        if((id&1) == 1)
            new_mask |= m;
        id >>= 1;
        mask ^= m;
    }
    return new_mask;
}

static big rook_mask(big id, big square){
    big mask = (clipped_row[square>>3]|clipped_col[square&7])&(~(1ULL<<square));
    return apply_id(id, mask);
}

static big bishop_mask(big id, big square){
    int col = square&7;
    int row = square>>3;
    big mask = (clipped_diag[col+row]|clipped_idiag[row-col+7])&(~(1ULL<<square));
    return apply_id(id, mask);
}

static big get_mask(bool is_rook, big id, big square){
    return (is_rook?rook_mask:bishop_mask)(id, square);
}

void load_table(){
    big magic=0;
    int decR=0, minimum=0, size=0;
    int total = 0;
    for(int i=0; i<128; i++){
        indexesTable[i] = i ? indexesTable[i-1]+(1<<constantsMagic[i-1].bits) : 0;
        total += 1<<constantsMagic[i].bits;
    }
    tableMagic = (big*)calloc(total, sizeof(big));
    for(int current = 0; current<128; current++){
        magic = constantsMagic[current].magic;
        decR = constantsMagic[current].decR;
        minimum = constantsMagic[current].bits;
        size = 1ul << minimum;
        const bool is_rook = current >= 64;
        const int square = current%64;
        int nbBits = __builtin_popcountll(get_mask(is_rook, MAX_BIG, square));
        const big nbIds = 1ul << nbBits;
        for(big id=0; id<nbIds; id++){
            const big mask = get_mask(is_rook, id, square);
            const big res = mask*magic;
            const big res_mask = get_usefull(is_rook, mask, square);
            const big key = (res&(MAX_BIG>>decR)) >> (64-decR-minimum);
            assert(key < (big)size);
            tableMagic[indexesTable[current]+key] = res_mask;
        }
    }
}

void clear_table(){
    free(tableMagic);
}

__attribute__((constructor(102)))
void init_consts_legalMove(){
    PrecomputeKnightMoveData();
    precomputeDirections();
    load_table();
    precomputeCastlingMasks();
    precomputeNormlaKingMoves();
    precomputePawnsAttack();
}

template<bool isPawn>
void LegalMoveGenerator::maskToMoves(int start, big mask, Move* moves, int& nbMoves, int8_t piece, bool promotQueen){
    while(mask){
        int bit = __builtin_ctzll(mask);
        mask &= mask-1;
        Move base;// = {(int8_t)start, (int8_t)bit, piece};
        base.updateFrom(start);
        base.updateTo(bit);
        base.piece = piece;
        big _mask = 1ULL << bit;
        //There is a capture
        if(_mask&allEnemies)
            for(int i=0; i<6; i++)
                if(enemyPieces[i]&_mask){ base.capture = i; break; }
        if(isPawn && (row(bit) == 7 || row(bit) == 0)){
            int8_t piecesPromot[4] = {KNIGHT, BISHOP, ROOK, QUEEN};
            int _start=0;
            if(promotQueen)
                _start=3;
            for(int i=_start; i<4; i++){
                moves[nbMoves] = base;
                // moves[nbMoves].promoteTo = typePiece;
                moves[nbMoves].updatePromotion(piecesPromot[i]);
                nbMoves++;
            }
        }else{
            if(isPawn && (col(start) != col(bit)) && base.capture == -2){
                base.capture = -1;
            }
            moves[nbMoves] = base;
            nbMoves++;
        }
    }
}

big moves_table(int index, big mask_pieces){
    int tIndex = (mask_pieces*constantsMagic[index].magic & (MAX_BIG >> constantsMagic[index].decR)) >> (64-constantsMagic[index].decR-constantsMagic[index].bits);
    return tableMagic[indexesTable[index]+tIndex];
}

inline big LegalMoveGenerator::pseudoLegalBishopMoves(int bishopPosition, big Pieces){
    // big bishopMoveMask=moves_table(bishopPosition, allPieces&mask_empty_bishop(bishopPosition));
    // return bishopMoveMask;
    return moves_table(bishopPosition, Pieces&mask_empty_bishop(bishopPosition));
}

inline big LegalMoveGenerator::pseudoLegalRookMoves(int rookPosition, big Pieces){
    // big rookMoveMask=moves_table(rookPosition+64, allPieces&mask_empty_rook(rookPosition));
    // return rookMoveMask;
    return moves_table(rookPosition+64, Pieces&mask_empty_rook(rookPosition));
}


inline big LegalMoveGenerator::pseudoLegalKnightMoves(int knightPosition){
    // big knightEndMask = KnightMoves[knightPosition];
    // return knightEndMask;
    return KnightMoves[knightPosition];
}

template<bool IsWhite>
big LegalMoveGenerator::pseudoLegalPawnMoves(int pawnPosition, big Pieces, int friendKingPos, big moveMask, big captureMask, big enPieces, int enPassant, big enemyRooks){
    big pawnMoveMask = 0;
    big pawnAttackMask = 0;

    if (moveMask != 0){
        int pieceRow = row(pawnPosition);
        constexpr int startRow = IsWhite ? 1 : 6;

        //Single pawn push (check there are no pieces on target square)
        big pushMask = IsWhite ? (1ULL << (pawnPosition + 8)) : (1ULL << (pawnPosition - 8));
        pawnMoveMask |= pushMask & (~Pieces);

        //Double pawn push
        if (pieceRow == startRow && pawnMoveMask){
            if constexpr (IsWhite) pushMask <<= 8;
            else pushMask >>= 8;
            pawnMoveMask |= pushMask & (~Pieces);
        }
    }

    //Just dangers created by pawns (no filtering by actual pieces that could be taken)
    constexpr int colorIdx = IsWhite ? 0 : 1;
    pawnAttackMask = attackPawns[colorIdx * 64 + pawnPosition];

    big finalMoveMask = (pawnMoveMask & moveMask) | (pawnAttackMask & captureMask & enPieces);

    // En passant: the captured pawn is one rank behind the EP square
    constexpr int epCapturedOffset = IsWhite ? -8 : 8;
    if (enPassant != -1 && ((pawnAttackMask & (1ull << enPassant)) != 0) && ((captureMask & (1ull << (enPassant + epCapturedOffset))) != 0)){
        big kingAsRook = pseudoLegalRookMoves(friendKingPos, Pieces ^ ((1ull << pawnPosition) | (1ull << (enPassant + epCapturedOffset))));
        if((row(friendKingPos) != row(enPassant + epCapturedOffset)) | ((kingAsRook&enemyRooks) == 0)){
            finalMoveMask |= (1ull << enPassant);
        }
    }

    return finalMoveMask;
}  

template<bool IsWhite>
big LegalMoveGenerator::pseudoLegalKingMoves(int kingPosition, big Pieces, bool kingCastling, bool queenCastling){
    constexpr int color = IsWhite ? 0 : 1;
    big kingEndMask = normalKingMoves[kingPosition];

    if(kingCastling && !(pieceCastlingMasks[color][1]&(Pieces)) && !(attackCastlingMasks[color][1] & allDangerSquares)){
        constexpr int posKingSideCastle = color * 56 + 1;
        kingEndMask |= 1ULL << posKingSideCastle;
    }

    if(queenCastling && !(pieceCastlingMasks[color][0]&(Pieces)) && !(attackCastlingMasks[color][0] & allDangerSquares)){
        constexpr int posQueenSideCastle = color * 56 + 5;
        kingEndMask |= 1ULL << posQueenSideCastle;
    }
    return kingEndMask;
}

template<bool IsWhite>
int LegalMoveGenerator::dealWithEnemyPawns(big enemyPawnPositions, int friendKingPos){
    // IsWhite means friendly is white, so enemy is black
    // Bit layout: bit 0 = a1, bit 7 = h1, bit 8 = a2, etc.
    // White pawns: +7 = up-left, +9 = up-right
    // Black pawns: -7 = down-right, -9 = down-left
    constexpr big FileA = 0x0101010101010101ULL;  // col 0
    constexpr big FileH = 0x8080808080808080ULL;  // col 7
    checkerPos = -1;

    // Single diagonal shift instead of vertical + horizontal
    big attacks7;  // the +7/-7 attack direction
    big attacks9;  // the +9/-9 attack direction
    if constexpr (IsWhite){
        // Enemy is black: attacks via sq-7 (down-right) and sq-9 (down-left)
        attacks7 = (enemyPawnPositions & ~FileH) >> 7;  // down-right, clip H file source
        attacks9 = (enemyPawnPositions & ~FileA) >> 9;  // down-left, clip A file source
    } else {
        // Enemy is white: attacks via sq+7 (up-left) and sq+9 (up-right)
        attacks7 = (enemyPawnPositions & ~FileA) << 7;  // up-left, clip A file source
        attacks9 = (enemyPawnPositions & ~FileH) << 9;  // up-right, clip H file source
    }
    allDangerSquares |= attacks7 | attacks9;

    if (attacks7 & (1ull << friendKingPos)){
        if constexpr (IsWhite)
            checkerPos = friendKingPos + 7;   // checker is 7 above (the black pawn)
        else
            checkerPos = friendKingPos - 7;   // checker is 7 below (the white pawn)
    }
    if (attacks9 & (1ull << friendKingPos)){
        if constexpr (IsWhite)
            checkerPos = friendKingPos + 9;
        else
            checkerPos = friendKingPos - 9;
    }

    return checkerPos;
}

int LegalMoveGenerator::dealWithEnemyKnights(big enemyKnightPositions, int friendKingPos){
    int checkerK = -1;
    for (big bb = enemyKnightPositions; bb; bb &= bb - 1){
        int currentKnightPos = __builtin_ctzll(bb);
        big dangerSquares = pseudoLegalKnightMoves(currentKnightPos);

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            //Knight is giving check (only 1 knight can give check)
            checkerK = currentKnightPos;
        }
    }
    return checkerK;
}

int LegalMoveGenerator::dealWithEnemyBishops(big enemyBishopPositions, big Pieces, int friendKingPos){
    int checkerB = -1;
    for (big bb = enemyBishopPositions; bb; bb &= bb - 1){
        int currentBishopPos = __builtin_ctzll(bb);
        big dangerSquares = pseudoLegalBishopMoves(currentBishopPos, Pieces ^ (1ull << friendKingPos));

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            if (checkerB != -1){
                //Already a piece giving check
                //Const value just to know two of same type are giving check
                checkerB = doubleCheckFromSameType;
            }
            else{
                checkerB = currentBishopPos;
            }
        }

        int kingRow = row(friendKingPos), kingCol = col(friendKingPos);
        int bishopRow = row(currentBishopPos), bishopCol = col(currentBishopPos);

        //It makes sense for a bishop to pin a piece if its in the same diagonal as the king
        if (abs(kingRow - bishopRow) == abs(kingCol - bishopCol)){
            big ray = directions[friendKingPos][currentBishopPos];
            big pinnedPieceMask = (ray ^ (1ull << currentBishopPos)) & Pieces;
            //There is a pinned piece;
            if (countbit(pinnedPieceMask) == 1){
                pinD12 |= ray;
            }
        }
    }
    return checkerB; 
}

int LegalMoveGenerator::dealWithEnemyRooks(big enemyRookPositions, big Pieces, int friendKingPos){
    int checkerR = -1;
    for (big bb = enemyRookPositions; bb; bb &= bb - 1){
        int currentRookPos = __builtin_ctzll(bb);
        big dangerSquares = pseudoLegalRookMoves(currentRookPos, Pieces ^ (1ull << friendKingPos));

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            if (checkerR != -1){
                //Already a piece giving check
                //Const value just to know two of same type are giving check
                checkerR = doubleCheckFromSameType;
            }
            else{
                checkerR = currentRookPos;
            }
        }

        int kingRow = row(friendKingPos), kingCol = col(friendKingPos);
        int rookRow = row(currentRookPos), rookCol = col(currentRookPos);

        //It makes sense for a rook to pin a piece if its on the same row or column as the king
        if (kingRow == rookRow || kingCol == rookCol){
            big ray = directions[friendKingPos][currentRookPos];
            big pinnedPieceMask = (ray ^ (1ull << currentRookPos)) & Pieces;
            //There is a pinned piece;
            if (countbit(pinnedPieceMask) == 1){
                pinHV |= ray;
            }
        }
    }
    return checkerR; 
}

void LegalMoveGenerator::dealWithEnemyKing(int enemyKingPos){
    //Castling args are false so template param is irrelevant
    big dangerSquares = pseudoLegalKingMoves<false>(enemyKingPos, 0, false, false);
    allDangerSquares |= dangerSquares;
}

template<bool IsWhite>
void LegalMoveGenerator::legalKingMoves(const GameState& state, Move* moves, int& nbMoves, big Pieces, big captureMask){
    constexpr int color = IsWhite ? 0 : 1;
    int kingPos = __builtin_ctzll(state.friendlyPieces()[KING]);
    big kingEndMask = pseudoLegalKingMoves<IsWhite>(kingPos, Pieces, state.castlingRights[color][1], state.castlingRights[color][0]);
    kingEndMask &= (~allFriends);
    kingEndMask &= (~allDangerSquares);
    //In case only captures need to be generated
    kingEndMask &= (captureMask);

    maskToMoves<false>(kingPos, kingEndMask, moves, nbMoves, KING);
}

template<bool IsWhite>
void LegalMoveGenerator::legalPawnMoves(big pawnMask, int lastDoublePawnPush, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves, big Pieces, big enemyRooks, bool promotQueen){
    for (big bb = pawnMask; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big sqBit = 1ULL << sq;
        big pawnMoveMask = pseudoLegalPawnMoves<IsWhite>(sq, Pieces, friendlyKingPosition, moveMask, captureMask, allEnemies, lastDoublePawnPush, enemyRooks);

        // Apply pin restrictions
        if (sqBit & pinD12)
            pawnMoveMask &= pinD12;     // diag-pinned: can only capture along pin ray
        else if (sqBit & pinHV)
            pawnMoveMask &= pinHV;      // HV-pinned: can only push along pin file

        maskToMoves<true>(sq, pawnMoveMask, pawnMoves, nbMoves, PAWN, promotQueen);
    }
}

void LegalMoveGenerator::legalKnightMoves(big knightMask, big moveMask, big captureMask, Move* knightMoves, int& nbMoves){
    // Pinned knights can never move (no knight move stays on a pin ray)
    big movableKnights = knightMask & ~(pinHV | pinD12);
    big target = moveMask | captureMask;
    for (big bb = movableKnights; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big knightEndMask = pseudoLegalKnightMoves(sq) & target;
        maskToMoves<false>(sq, knightEndMask, knightMoves, nbMoves, KNIGHT);
    }
}

void LegalMoveGenerator::legalSlidingMoves(big moveMask, big captureMask, Move* slidingMoves, int& nbMoves, big Pieces){
    big target = moveMask | captureMask;
    big pinned = pinHV | pinD12;

    // --- Bishop-like pieces (bishops + queens using diagonal moves) ---
    big bishopLike = friendlyPieces[BISHOP] | friendlyPieces[QUEEN];

    // Unpinned bishop-like: full freedom (within target)
    for (big bb = bishopLike & ~pinned; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big moves = pseudoLegalBishopMoves(sq, Pieces) & target;
        int8_t piece = (friendlyPieces[BISHOP] & (1ULL << sq)) ? BISHOP : QUEEN;
        maskToMoves<false>(sq, moves, slidingMoves, nbMoves, piece);
    }
    // Diag-pinned bishop-like: restricted to pinD12 ray
    for (big bb = bishopLike & pinD12; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big moves = pseudoLegalBishopMoves(sq, Pieces) & target & pinD12;
        int8_t piece = (friendlyPieces[BISHOP] & (1ULL << sq)) ? BISHOP : QUEEN;
        maskToMoves<false>(sq, moves, slidingMoves, nbMoves, piece);
    }
    // HV-pinned bishop-like pieces cannot move diagonally, skip them

    // --- Rook-like pieces (rooks + queens using HV moves) ---
    big rookLike = friendlyPieces[ROOK] | friendlyPieces[QUEEN];

    // Unpinned rook-like: full freedom (within target)
    for (big bb = rookLike & ~pinned; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big moves = pseudoLegalRookMoves(sq, Pieces) & target;
        int8_t piece = (friendlyPieces[ROOK] & (1ULL << sq)) ? ROOK : QUEEN;
        maskToMoves<false>(sq, moves, slidingMoves, nbMoves, piece);
    }
    // HV-pinned rook-like: restricted to pinHV ray
    for (big bb = rookLike & pinHV; bb; bb &= bb - 1){
        int sq = __builtin_ctzll(bb);
        big moves = pseudoLegalRookMoves(sq, Pieces) & target & pinHV;
        int8_t piece = (friendlyPieces[ROOK] & (1ULL << sq)) ? ROOK : QUEEN;
        maskToMoves<false>(sq, moves, slidingMoves, nbMoves, piece);
    }
    // Diag-pinned rook-like pieces cannot move along HV, skip them
}
bool LegalMoveGenerator::isCheck() const{
    return nbCheckers >= 1;
}
template<bool IsWhite>
bool LegalMoveGenerator::initDangersImpl(const GameState& state){
    pinHV = 0;
    pinD12 = 0;
    nbCheckers = 0;
    checkerPos = -1;
    friendlyPieces = state.friendlyPieces();
    enemyPieces = state.enemyPieces();

    friendlyKingPosition = __builtin_ctzll(friendlyPieces[KING]);
    enemyKingPosition = __builtin_ctzll(enemyPieces[KING]);

    allFriends = 0;
    allEnemies = 0;
    for (int i = 0; i < 6; ++i){
        big boardFriend = friendlyPieces[i];
        allFriends |= boardFriend;
        big boardEnemy = enemyPieces[i];
        allEnemies |= boardEnemy;
    }

    allPieces = allFriends | allEnemies;
    allDangerSquares = 0;
    dealWithEnemyKing(enemyKingPosition);

    //Updates the danger squares and retrieves the possibe pawn checker
    int pawnCheckerPos = dealWithEnemyPawns<IsWhite>(enemyPieces[PAWN], friendlyKingPosition);
    if (pawnCheckerPos != -1){
        nbCheckers += 1;
        checkerPos = pawnCheckerPos;
    }

    //Updates the danger squares and retrieves the possibe knight checker
    int knightCheckerPos = dealWithEnemyKnights(enemyPieces[KNIGHT], friendlyKingPosition);
    if (knightCheckerPos != -1){
        nbCheckers += 1;
        checkerPos = knightCheckerPos;
    }

    //Now pieces can pin and have multiple of a type attacking the king

    //Add the queen for its bishop rays
    int bishopCheckerPos = dealWithEnemyBishops(enemyPieces[BISHOP] | enemyPieces[QUEEN], allPieces, friendlyKingPosition);
    if (bishopCheckerPos != -1){
        nbCheckers += 1;
        if (bishopCheckerPos == doubleCheckFromSameType){
            nbCheckers += 1;
        }
        else{
            checkerPos = bishopCheckerPos;
        }
    }

    //Add the queen for its rook rays
    int rookCheckerPos = dealWithEnemyRooks(enemyPieces[ROOK] | enemyPieces[QUEEN], allPieces, friendlyKingPosition);
    if (rookCheckerPos != -1){
        nbCheckers += 1;
        if (rookCheckerPos == doubleCheckFromSameType){
            nbCheckers += 1;
        }
        else{
            checkerPos = rookCheckerPos;
        }
    }
    return nbCheckers >= 1;
}

bool LegalMoveGenerator::initDangers(const GameState& state){
    return state.friendlyColor() ? initDangersImpl<false>(state) : initDangersImpl<true>(state);
}

template<bool IsWhite, bool InCheck>
int LegalMoveGenerator::generateLegalMovesImpl(const GameState& state, bool& inCheck, Move* legalMoves, big& dangerPositions, bool onlyCapture){
    big moveMask = (~allFriends);
    big captureMask = allEnemies;
    inCheck = InCheck;
    int nbMoves = 0;

    dangerPositions = allDangerSquares;

    //From here we have the pinned pieces, the number of checkers, and the danger squares
    if(onlyCapture){
        moveMask = 0;
        legalKingMoves<IsWhite>(state, legalMoves, nbMoves, allPieces, captureMask);
    }
    else{
        legalKingMoves<IsWhite>(state, legalMoves, nbMoves, allPieces);
    }
    big pawnMoveMask = ~allFriends;

    if constexpr (InCheck){
        if (nbCheckers >= 2){
            //Because if there are two checkers than only king moves are interesting
            return nbMoves;
        }
        // Single check: restrict to ray + checker
        captureMask = (1ull << checkerPos);
        big rayToChecker = directions[friendlyKingPosition][checkerPos];
        moveMask &= rayToChecker;
        pawnMoveMask &= rayToChecker;
    }

    pawnMoveMask = (onlyCapture?0:pawnMoveMask)|(pawnMoveMask&(~clipped_brow));
    legalPawnMoves<IsWhite>(friendlyPieces[PAWN], state.lastDoublePawnPush, pawnMoveMask, captureMask, legalMoves, nbMoves, allPieces, enemyPieces[ROOK] | enemyPieces[QUEEN], onlyCapture);
    legalKnightMoves(friendlyPieces[KNIGHT], moveMask, captureMask, legalMoves, nbMoves);
    legalSlidingMoves(moveMask, captureMask, legalMoves, nbMoves, allPieces);
    return nbMoves;
}

int LegalMoveGenerator::generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves, big& dangerPositions, bool onlyCapture){
    bool check = nbCheckers >= 1;
    if (state.friendlyColor())
        return check ? generateLegalMovesImpl<false, true>(state, inCheck, legalMoves, dangerPositions, onlyCapture)
                     : generateLegalMovesImpl<false, false>(state, inCheck, legalMoves, dangerPositions, onlyCapture);
    else
        return check ? generateLegalMovesImpl<true, true>(state, inCheck, legalMoves, dangerPositions, onlyCapture)
                     : generateLegalMovesImpl<true, false>(state, inCheck, legalMoves, dangerPositions, onlyCapture);
}

template<bool IsWhite>
Move LegalMoveGenerator::getLVAImpl(int posCapture, GameState& state){
    pinHV = 0;
    pinD12 = 0;
    Move LVAmove;
    big captureMask = -1;

    nbCheckers = 0;
    checkerPos = -1;

    friendlyPieces = state.friendlyPieces();
    enemyPieces = state.enemyPieces();

    friendlyKingPosition = __builtin_ctzll(friendlyPieces[KING]);
    enemyKingPosition = __builtin_ctzll(enemyPieces[KING]);

    allFriends = 0;
    allEnemies = 0;
    for (int i = 0; i < 6; ++i){
        big boardFriend = friendlyPieces[i];
        allFriends |= boardFriend;
        big boardEnemy = enemyPieces[i];
        allEnemies |= boardEnemy;
    }

    allPieces = allFriends | allEnemies;

    captureMask = allEnemies;

    allDangerSquares = 0;

    dealWithEnemyKing(enemyKingPosition);

    //Updates the danger squares and retrieves the possibe pawn checker
    int pawnCheckerPos = dealWithEnemyPawns<IsWhite>(enemyPieces[PAWN], friendlyKingPosition);
    if (pawnCheckerPos != -1){
        nbCheckers++;
        checkerPos = pawnCheckerPos;
        if(checkerPos != posCapture)return nullMove;
    }

    //Updates the danger squares and retrieves the possibe knight checker
    int knightCheckerPos = dealWithEnemyKnights(enemyPieces[KNIGHT], friendlyKingPosition);
    if (knightCheckerPos != -1){
        if(nbCheckers++)return nullMove;
        checkerPos = knightCheckerPos;
        if(checkerPos != posCapture)return nullMove;
    }

    //Now pieces can pin and have multiple of a type attacking the king

    //Add the queen for its bishop rays
    int bishopCheckerPos = dealWithEnemyBishops(enemyPieces[BISHOP] | enemyPieces[QUEEN], allPieces, friendlyKingPosition);
    if (bishopCheckerPos != -1){
        if(nbCheckers++)return nullMove;
        if (bishopCheckerPos == doubleCheckFromSameType){
            return nullMove;
        }
        else{
            checkerPos = bishopCheckerPos;
            if(checkerPos != posCapture)return nullMove;
        }
    }

    //Add the queen for its rook rays
    int rookCheckerPos = dealWithEnemyRooks(enemyPieces[ROOK] | enemyPieces[QUEEN], allPieces, friendlyKingPosition);
    if (rookCheckerPos != -1){
        if(nbCheckers++)return nullMove;
        if (rookCheckerPos == doubleCheckFromSameType){
            return nullMove;
        }
        else{
            checkerPos = rookCheckerPos;
            if(checkerPos != posCapture)return nullMove;
        }
    }

    captureMask = 1ULL << posCapture;

    int capturedPiece = PAWN;
    for(int i=1; i<6; i++){
        if(enemyPieces[i]&captureMask){
            capturedPiece = i;
            break;
        }
    }
    LVAmove.capture = capturedPiece;
    LVAmove.updateTo(posCapture);
    //From here we have the pinned pieces, the number of checkers, and the danger squares
    big kingEndMask = pseudoLegalKingMoves<IsWhite>(friendlyKingPosition, allPieces, false, false);
    kingEndMask &= (~allDangerSquares);
    if(kingEndMask & captureMask){
        LVAmove.updateFrom(friendlyKingPosition);
        LVAmove.piece = KING;
        return LVAmove;
    }
    big fromCaseBishop = moves_table(posCapture, allPieces&mask_empty_bishop(posCapture));
    big fromCaseRook = moves_table(posCapture+64, allPieces&mask_empty_rook(posCapture));
    big pinned = pinHV | pinD12;
    constexpr int enemyColorIdx = IsWhite ? 1 : 0;
    big possiblePieces[5] = {
        friendlyPieces[PAWN]   & attackPawns[enemyColorIdx*64+posCapture],
        friendlyPieces[KNIGHT] & KnightMoves[posCapture] & ~pinned,
        friendlyPieces[BISHOP] & fromCaseBishop,
        friendlyPieces[ROOK]   & fromCaseRook,
        friendlyPieces[QUEEN]  & (fromCaseBishop | fromCaseRook),
    };
    for(int piece=0; piece<KING; piece++){
        for(big bb = possiblePieces[piece]; bb; bb &= bb - 1){
            int sq = __builtin_ctzll(bb);
            big sqBit = 1ULL << sq;
            if ((sqBit & pinned) && !(pinned & captureMask)) continue;
            LVAmove.piece = piece;
            LVAmove.updateFrom(sq);
            if(piece == PAWN && (row(posCapture) == 0 || row(posCapture) == 7)){
                LVAmove.updatePromotion(QUEEN);
            }
            return LVAmove;
        }
    }
    return nullMove;
}

Move LegalMoveGenerator::getLVA(int posCapture, GameState& state){
    return state.friendlyColor() ? getLVAImpl<false>(posCapture, state) : getLVAImpl<true>(posCapture, state);
}