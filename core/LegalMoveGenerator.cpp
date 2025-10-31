#include "LegalMoveGenerator.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include <utility>
#include <cstring>
using namespace std;

BINARY_ASM_INCLUDE("magics.out", magicsData);

big parseInt(int& pointer){
    big current = 0;
    if(pointer >= magicsData_size)return 0;
    while(transform(magicsData[pointer]) > '9' || transform(magicsData[pointer]) < '0'){
        pointer++; //remove useless char
        if(pointer >= magicsData_size)return 0;
    }
    while(transform(magicsData[pointer]) <= '9' && transform(magicsData[pointer]) >= '0'){
        current = current*10+(transform(magicsData[pointer])-'0');
        pointer++;
        if(pointer == magicsData_size)return 0;
    }
    return current;
}

void LegalMoveGenerator::PrecomputeKnightMoveData(){
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
void LegalMoveGenerator::precomputeCastlingMasks(){
    pieceCastlingMasks[0][1] = 0b00000110;
    pieceCastlingMasks[0][0] = 0b01110000;
    pieceCastlingMasks[1][1] = pieceCastlingMasks[0][1] << 56;
    pieceCastlingMasks[1][0] = pieceCastlingMasks[0][0] << 56;

    attackCastlingMasks[0][1] = 0b00001110;
    attackCastlingMasks[0][0] = 0b00111000;
    attackCastlingMasks[1][1] = attackCastlingMasks[0][1] << 56;
    attackCastlingMasks[1][0] = attackCastlingMasks[0][0] << 56;
}

void LegalMoveGenerator::precomputeNormlaKingMoves(){
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

void LegalMoveGenerator::precomputePawnsAttack(){
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
void LegalMoveGenerator::precomputeDirections(){
    //Set everything to 0 first just to be sure
    for (int i = 0; i < 64; ++i){
        for (int j = 0; j < 64; ++j){
            directions[i][j] = 0;
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
                // fullDir[square][idDir] = mask;
            }
        }
    }
}

void LegalMoveGenerator::load_table(){
    big magic;
    int decR, minimum, size;
    int current = 0;
    int pointer = 0;
    //printf("%d\n", magicsData_size);
    while(pointer != magicsData_size){
        magic = parseInt(pointer);
        if(!magic)break;
        decR = parseInt(pointer);
        minimum = parseInt(pointer);
        size = parseInt(pointer);
        //printf("%d\n", size);
        constantsMagic[current] = {minimum, decR, magic};
        tableMagic[current] = (big*)calloc(size, sizeof(big));
        for(int i=0; i<size; i++){
            tableMagic[current][i] = parseInt(pointer);
        }
        current++;
    }
}
LegalMoveGenerator::LegalMoveGenerator(){
    PrecomputeKnightMoveData();
    load_table();
    init_lines();
    precomputePawnsAttack();
    precomputeCastlingMasks();
    precomputeNormlaKingMoves();
    precomputeDirections();
}
LegalMoveGenerator::~LegalMoveGenerator(){
    for(int i=0; i<128; i++){
        free(tableMagic[i]);
    }
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
        big mask = 1ULL << bit;
        //There is a capture
        if(mask&allEnemies)
            for(int i=0; i<6; i++)
                if(enemyPieces[i]&mask)base.capture = i;
        if(isPawn && (row(bit) == 7 || row(bit) == 0)){
            int8_t piecesPromot[4] = {KNIGHT, BISHOP, ROOK, QUEEN};
            int start=0;
            if(promotQueen)
                start=3;
            for(int i=start; i<4; i++){
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

inline big LegalMoveGenerator::moves_table(int index, big mask_pieces){
    return tableMagic[index][(mask_pieces*constantsMagic[index].magic & (MAX_BIG >> constantsMagic[index].decR)) >> (64-constantsMagic[index].decR-constantsMagic[index].bits)];
}

inline big LegalMoveGenerator::pseudoLegalBishopMoves(int bishopPosition, big allPieces){
    // big bishopMoveMask=moves_table(bishopPosition, allPieces&mask_empty_bishop(bishopPosition));
    // return bishopMoveMask;
    return moves_table(bishopPosition, allPieces&mask_empty_bishop(bishopPosition));
}

inline big LegalMoveGenerator::pseudoLegalRookMoves(int rookPosition, big allPieces){
    // big rookMoveMask=moves_table(rookPosition+64, allPieces&mask_empty_rook(rookPosition));
    // return rookMoveMask;
    return moves_table(rookPosition+64, allPieces&mask_empty_rook(rookPosition));
}

inline big LegalMoveGenerator::pseudoLegalQueenMoves(int queenPositions, big allPieces){
    // big bishopMovesFromQueen = pseudoLegalBishopMoves(queenPositions, allPieces);
    // big rookMovesFromQueen = pseudoLegalRookMoves(queenPositions ,allPieces);
    
    // return bishopMovesFromQueen | rookMovesFromQueen;

    return pseudoLegalBishopMoves(queenPositions, allPieces) | pseudoLegalRookMoves(queenPositions ,allPieces);;
}

inline big LegalMoveGenerator::pseudoLegalKnightMoves(int knightPosition){
    // big knightEndMask = KnightMoves[knightPosition];
    // return knightEndMask;
    return KnightMoves[knightPosition];
}

big LegalMoveGenerator::pseudoLegalPawnMoves(int pawnPosition, bool color, big& allPieces, int friendKingPos, big moveMask, big captureMask, big enemyPieces, int enPassant, big enemyRooks){
    big finalMoveMask;
    
    //If color == 0 -> white (add to move)
    //If color == 1 -> black (subtract to move)

    //Should be if color is 1 (true) then moveFactor is -1
    int moveFactor = color ? -1 : 1;

    // int leftLimit = color ?  7 : 0;
    // int rightLimit = leftLimit^7;
    // int rowPassant = color ? 3 : 4;

    big pawnMoveMask = 0;
    big pawnAttackMask = 0;

    if (moveMask != 0){
        int pieceRow = row(pawnPosition);
        int startRow = color ? 6 : 1;

        //Single pawn push (check there are no pieces on target square)
        big pushMask = 1ULL << (pawnPosition+8*moveFactor);
        pawnMoveMask |= (pushMask) & (~allPieces);

        //Double pawn push
        if (pieceRow == startRow && pawnMoveMask){
            if(color)pushMask >>= 8;
            else pushMask <<= 8;
            pawnMoveMask |= pushMask & (~allPieces);
        }
    }

    //Just dangers created by pawns (no filtering by actual pieces that could be taken)
    pawnAttackMask = attackPawns[color * 64 + pawnPosition];

    finalMoveMask = (pawnMoveMask & moveMask) | (pawnAttackMask & captureMask & enemyPieces);

    if (enPassant != -1 && ((pawnAttackMask & (1ull << enPassant)) != 0) && ((captureMask & (1ull << (enPassant + (-1 * 8 * moveFactor)))) != 0)){
        //This means en-passant actually captures the pawn that is causing check

        big kingAsRook = pseudoLegalRookMoves(friendKingPos,allPieces ^ ((1ull << pawnPosition) | (1ull << (enPassant + (-1 * 8 * moveFactor)))));
        if((row(friendKingPos) != row(enPassant + (8 * moveFactor * -1))) | ((kingAsRook&enemyRooks) == 0)){
            finalMoveMask |= (1ull << enPassant);
        }
    }

    return finalMoveMask;
}  

big LegalMoveGenerator::pseudoLegalKingMoves(int kingPosition,const big& allPieces, bool color, bool kingCastling, bool queenCastling, big& dangerSquares){
    big kingEndMask = 0;
    kingEndMask = normalKingMoves[kingPosition];
    
    if(kingCastling && !(pieceCastlingMasks[color][1]&(allPieces)) && !(attackCastlingMasks[color][1] & dangerSquares)){
        int posKingSideCastle=color*56+1;
        kingEndMask |= 1ULL << posKingSideCastle;
    }
    
    if(queenCastling  && !(pieceCastlingMasks[color][0]&(allPieces)) && !(attackCastlingMasks[color][0] & dangerSquares)){
        int posQueenSideCastle = color*56+5;
        kingEndMask |= 1ULL << posQueenSideCastle;
    }
    return kingEndMask;
}

int LegalMoveGenerator::dealWithEnemyPawns(big enemyPawnPositions, int friendKingPos, int enemyColor ,big& allDangerSquares){
    int moveFactor = enemyColor ? -1 : 1;

    int checkerPos = -1;

    //Attacks to the left
    const big col1 = (1ull << 7) + (1ull << 15) + (1ull << 23) + (1ull << 31) + (1ull << 39) + (1ull << 47) + (1ull << 55) + (1ull << 63);
    big possibleToTheLeft = enemyPawnPositions & (~col1);
    big attacksToTheLeft;
    if (moveFactor == 1){
        attacksToTheLeft = possibleToTheLeft << (8);
    }
    else{
        attacksToTheLeft = possibleToTheLeft >> (8);
    }
    attacksToTheLeft <<= 1;
    allDangerSquares |= attacksToTheLeft;

    if (attacksToTheLeft & (1ull << friendKingPos)){
        checkerPos = friendKingPos + (8 * moveFactor * -1) - 1;
    }

    //Attacks to the right
    const big col8 = col1 >> 7;
    big possibleToTheRight = enemyPawnPositions & (~col8);
    big attacksToTheRight;
    if (moveFactor == 1){
        attacksToTheRight = possibleToTheRight << (8);
    }
    else{
        attacksToTheRight = possibleToTheRight >> (8);
    }
    attacksToTheRight >>= 1;
    allDangerSquares |= attacksToTheRight;
    if (attacksToTheRight & (1ull << friendKingPos)){
        checkerPos = friendKingPos + (8 * moveFactor * -1) + 1;
    }

    return checkerPos;
}

int LegalMoveGenerator::dealWithEnemyKnights(big enemyKnightPositions, int friendKingPos, big& allDangerSquares){
    int checkerPos = -1;
    ubyte positions[10];
    int nbPositions = places(enemyKnightPositions, positions);
    for (int i = 0; i < nbPositions; ++i){
        int currentKnightPos = positions[i];
        big dangerSquares = pseudoLegalKnightMoves(currentKnightPos);

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            //Knight is giving check (only 1 knight can give check)
            checkerPos = currentKnightPos;
        }
    }
    return checkerPos;
}

int LegalMoveGenerator::dealWithEnemyBishops(big enemyBishopPositions, big& allPieces, int friendKingPos, big& allDangerSquares){
    int checkerPos = -1;
    ubyte positions[11];
    int nbPositions = places(enemyBishopPositions, positions);
    for (int i = 0; i < nbPositions; ++i){
        int currentBishopPos = positions[i];
        big dangerSquares = pseudoLegalBishopMoves(currentBishopPos, allPieces ^ (1ull << friendKingPos));

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            if (checkerPos != -1){
                //Already a piece giving check
                //Const value just to know two of same type are giving check
                checkerPos = doubleCheckFromSameType;
            }
            else{
                checkerPos = currentBishopPos;
            }
        }

        int kingRow = row(friendKingPos), kingCol = col(friendKingPos);
        int bishopRow = row(currentBishopPos), bishopCol = col(currentBishopPos);

        //It makes sense for a bishop to pin a piece if its in the same diagonal as the king
        if (abs(kingRow - bishopRow) == abs(kingCol - bishopCol)){
            big ray = directions[friendKingPos][currentBishopPos];
            big pinnedPieceMask = (ray ^ (1ull << currentBishopPos)) & allPieces;
            //There is a pinned piece;
            if (countbit(pinnedPieceMask) == 1){
                int pinnedPiecePos = __builtin_ctzll(pinnedPieceMask);
                // printf("Bishop pin ray is (bishop excluded) :\n");
                // print_mask(ray);
                pinnedMasks[pinnedPiecePos] = ray;
            }
        }
    }
    return checkerPos; 
}

int LegalMoveGenerator::dealWithEnemyRooks(big enemyRookPositions, big& allPieces, int friendKingPos, big& allDangerSquares){
    int checkerPos = -1;
    ubyte positions[11];
    int nbPositions = places(enemyRookPositions, positions);
    for (int i = 0; i < nbPositions; ++i){
        int currentRookPos = positions[i];
        big dangerSquares = pseudoLegalRookMoves(currentRookPos, allPieces ^ (1ull << friendKingPos));

        allDangerSquares |= dangerSquares;

        if (dangerSquares & (1ull << friendKingPos)){
            if (checkerPos != -1){
                //Already a piece giving check
                //Const value just to know two of same type are giving check
                checkerPos = doubleCheckFromSameType;
            }
            else{
                checkerPos = currentRookPos;
            }
        }

        int kingRow = row(friendKingPos), kingCol = col(friendKingPos);
        int rookRow = row(currentRookPos), rookCol = col(currentRookPos);

        //It makes sense for a rook to pin a piece if its on the same row or column as the king
        if (kingRow == rookRow || kingCol == rookCol){
            big ray = directions[friendKingPos][currentRookPos];
            big pinnedPieceMask = (ray ^ (1ull << currentRookPos)) & allPieces;
            //There is a pinned piece;
            if (countbit(pinnedPieceMask) == 1){
                int pinnedPiecePos = __builtin_ctzll(pinnedPieceMask);
                pinnedMasks[pinnedPiecePos] = ray;  
            }
        }
    }
    return checkerPos; 
}

void LegalMoveGenerator::dealWithEnemyKing(int enemyKingPos, big& allDangerSquares){
    //The 4 0's are to prevent from generating castling
    big dangerSquares = pseudoLegalKingMoves(enemyKingPos,0,0,0,0,allDangerSquares);
    allDangerSquares |= dangerSquares;
}

void LegalMoveGenerator::legalKingMoves(const GameState& state, Move* moves, int& nbMoves, big friendlyPieces, big allPieces, big dangerSquares, big captureMask){
    big kingMask = state.friendlyPieces()[KING];
    int kingPos = __builtin_ctzll(kingMask);
    bool curColor = state.friendlyColor();
    big kingEndMask = pseudoLegalKingMoves(kingPos, allPieces, curColor, state.castlingRights[curColor][1], state.castlingRights[curColor][0],dangerSquares);
    // printf("King wants to go here :\n");
    // print_mask(kingEndMask);
    kingEndMask &= (~friendlyPieces);
    kingEndMask &= (~dangerSquares);
    //In case only captures need to be generated
    kingEndMask &= (captureMask);
    // printf("After ands :\n");
    // print_mask(kingEndMask);

    maskToMoves<false>(kingPos,kingEndMask, moves, nbMoves, KING);
}

void LegalMoveGenerator::legalPawnMoves(big pawnMask, bool friendlyColor, int lastDoublePawnPush, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves, big allPieces, big allEnemies, big enemyRooks, bool promotQueen){
    ubyte pos[8];
    int nbPos=places(pawnMask, pos);
    for (int p = 0;p<nbPos;++p){
        big pawnMoveMask = pseudoLegalPawnMoves(pos[p], friendlyColor, allPieces, friendlyKingPosition, moveMask, captureMask, allEnemies, lastDoublePawnPush, enemyRooks);
        // printf("Pawn at pos %d wants to move to :\n",pos[p]);
        // print_mask(pawnMoveMask);
        pawnMoveMask &= pinnedMasks[pos[p]];
        // if(pinnedMasks[pos[p]] != -1ull){
        //     printf("Pinn mask for piece:\n");
        //     print_mask(pinnedMasks[pos[p]]);
        //     printf("After pin and:\n");
        //     print_mask(pawnMoveMask);
        // }

        maskToMoves<true>(pos[p], pawnMoveMask, pawnMoves, nbMoves, PAWN, promotQueen);
    }
}

void LegalMoveGenerator::legalKnightMoves(big knightMask, big moveMask, big captureMask, Move* knightMoves, int& nbMoves){
    ubyte pos[10];
    int nbPos=places(knightMask, pos);
    for (int p = 0;p<nbPos;++p){
        big knightEndMask = pseudoLegalKnightMoves(pos[p]);
        // printf("Knight at position %d want to go to:\n",pos[p]);
        // print_mask(knightEndMask);

        knightEndMask &= (moveMask | captureMask);
        knightEndMask &= pinnedMasks[pos[p]];
        // printf("After ands : \n");
        // print_mask(knightEndMask);


        maskToMoves<false>(pos[p], knightEndMask, knightMoves, nbMoves, KNIGHT);
    }
}

void LegalMoveGenerator::legalSlidingMoves(big moveMask, big captureMask, Move* slidingMoves, int& nbMoves, big allPieces){
    int types[3] = {BISHOP,ROOK,QUEEN};
    for (int pieceType : types){
        //Change function
        big typeMask = friendlyPieces[pieceType];
        ubyte pos[11];
        int nbPos=places(typeMask, pos);
        for (int p = 0;p<nbPos;++p){
            big typeEndMask = 0;
            if (pieceType == BISHOP){
                typeEndMask = pseudoLegalBishopMoves(pos[p], allPieces);
                // printf("Bishop at pos %d wants to go to :\n",pos[p]);
                // print_mask(typeEndMask);
            }
            else if (pieceType == ROOK){
                typeEndMask = pseudoLegalRookMoves(pos[p], allPieces);
            }
            else if (pieceType == QUEEN){
                typeEndMask = pseudoLegalQueenMoves(pos[p], allPieces);
            }

            typeEndMask &= (moveMask | captureMask);
            typeEndMask &= pinnedMasks[pos[p]];
            // if (pieceType == BISHOP){
            //     printf("After ands :\n");
            //     print_mask(typeEndMask);
            // }
            
            maskToMoves<false>(pos[p], typeEndMask, slidingMoves, nbMoves, pieceType);
        }
    }
}
bool LegalMoveGenerator::isCheck() const{
    return nbCheckers >= 1;
}
bool LegalMoveGenerator::initDangers(const GameState& state){
    memset(pinnedMasks, 0xFF, sizeof(pinnedMasks));
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
    dealWithEnemyKing(enemyKingPosition,allDangerSquares);

    //Updates the danger squares and retrieves the possibe pawn checker
    int pawnCheckerPos = dealWithEnemyPawns(enemyPieces[PAWN],friendlyKingPosition,state.enemyColor(),allDangerSquares);
    if (pawnCheckerPos != -1){
        nbCheckers += 1;
        checkerPos = pawnCheckerPos;
    }

    //Updates the danger squares and retrieves the possibe knight checker
    int knightCheckerPos = dealWithEnemyKnights(enemyPieces[KNIGHT],friendlyKingPosition,allDangerSquares);
    if (knightCheckerPos != -1){
        nbCheckers += 1;
        checkerPos = knightCheckerPos;
    }

    //Now pieces can pin and have multiple of a type attacking the king

    //Add the queen for its bishop rays
    int bishopCheckerPos = dealWithEnemyBishops(enemyPieces[BISHOP] | enemyPieces[QUEEN], allPieces,friendlyKingPosition,allDangerSquares);
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
    int rookCheckerPos = dealWithEnemyRooks(enemyPieces[ROOK] | enemyPieces[QUEEN], allPieces, friendlyKingPosition,allDangerSquares);
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

int LegalMoveGenerator::generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves,big& dangerPositions, bool onlyCapture){
    //Set all pinned masks to -1 (= no pinning)

    //All allowed spots for a piece to move (not allowed if king is in check)
    big moveMask = -1; //Totaly true
    //All allowed spots for a piece to capture another one (not allowed if there is a checker)
    big captureMask = -1; //Totaly false
    inCheck = false;
    int nbMoves = 0;
    moveMask = (~allFriends);
    captureMask = allEnemies;

    dangerPositions = allDangerSquares;

    //From here we have the pinned pieces, the number of checkers, and the danger squares
    if(onlyCapture){
        moveMask = 0;
        legalKingMoves(state, legalMoves, nbMoves, allFriends, allPieces, allDangerSquares, captureMask);
    }
    else{
        legalKingMoves(state, legalMoves, nbMoves, allFriends, allPieces, allDangerSquares);
    }
    big pawnMoveMask = ~allFriends;
    if (nbCheckers >= 2){
        inCheck = true;
        //Because if there are two checkers than only king moves are interesting
        return nbMoves;
    }else if (nbCheckers == 1){
        inCheck = true;
        captureMask = (1ull << checkerPos);
        //If checker is a slider this is perfect logic
        //If it's a pawn then logic is fine too (because king is already excluded and the pawn is already in the captureMask)
        //If it's a knight then no valid ray exists so the mask is 0 (which is what we want)
        big rayToChecker = directions[friendlyKingPosition][checkerPos];
        moveMask &= rayToChecker;
        pawnMoveMask &= rayToChecker;
    }
    pawnMoveMask = (onlyCapture?0:pawnMoveMask)|(pawnMoveMask&(~clipped_brow));
    legalPawnMoves(friendlyPieces[PAWN], state.friendlyColor(), state.lastDoublePawnPush, pawnMoveMask, captureMask, legalMoves, nbMoves, allPieces, allEnemies, enemyPieces[ROOK] | enemyPieces[QUEEN], onlyCapture);
    legalKnightMoves(friendlyPieces[KNIGHT], moveMask, captureMask, legalMoves, nbMoves);
    legalSlidingMoves(moveMask, captureMask, legalMoves, nbMoves, allPieces);
    return nbMoves;
}

Move LegalMoveGenerator::getLVA(int posCapture, GameState& state){
    memset(pinnedMasks, 0xFF, sizeof(pinnedMasks));
    Move LVAmove;
    //All allowed spots for a piece to move (not allowed if king is in check)
    //All allowed spots for a piece to capture another one (not allowed if there is a checker)
    big captureMask = -1; //Totaly false

    int nbCheckers = 0;
    int checkerPos = -1;

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

    big allDangerSquares = 0;

    dealWithEnemyKing(enemyKingPosition,allDangerSquares);

    //Updates the danger squares and retrieves the possibe pawn checker
    int pawnCheckerPos = dealWithEnemyPawns(enemyPieces[PAWN],friendlyKingPosition,state.enemyColor(),allDangerSquares);
    if (pawnCheckerPos != -1){
        nbCheckers++;
        checkerPos = pawnCheckerPos;
        if(checkerPos != posCapture)return nullMove;
    }

    //Updates the danger squares and retrieves the possibe knight checker
    int knightCheckerPos = dealWithEnemyKnights(enemyPieces[KNIGHT],friendlyKingPosition,allDangerSquares);
    if (knightCheckerPos != -1){
        if(nbCheckers++)return nullMove;
        checkerPos = knightCheckerPos;
        if(checkerPos != posCapture)return nullMove;
    }

    //Now pieces can pin and have multiple of a type attacking the king

    //Add the queen for its bishop rays
    int bishopCheckerPos = dealWithEnemyBishops(enemyPieces[BISHOP] | enemyPieces[QUEEN], allPieces,friendlyKingPosition,allDangerSquares);
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
    int rookCheckerPos = dealWithEnemyRooks(enemyPieces[ROOK] | enemyPieces[QUEEN], allPieces, friendlyKingPosition,allDangerSquares);
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
    bool curColor = state.friendlyColor();
    big kingEndMask = pseudoLegalKingMoves(friendlyKingPosition, allPieces, curColor, false, false, allDangerSquares);
    kingEndMask &= (~allDangerSquares);
    if(kingEndMask & captureMask){//because this is for SEE, we do not care if we take with a king or a pawn if the opponent cannot retake
        LVAmove.updateFrom(friendlyKingPosition);
        LVAmove.piece = KING;
        return LVAmove;
    }
    /*assert(nbCheckers <= 1);
    if (nbCheckers >= 2){
        return nullMove;
    }else if (nbCheckers == 1){
        captureMask &= (1ULL << checkerPos);
        if(!captureMask)return nullMove;
    }*/
    big fromCaseBishop = moves_table(posCapture, allPieces&mask_empty_bishop(posCapture));
    big fromCaseRook = moves_table(posCapture+64, allPieces&mask_empty_rook(posCapture));
    big possiblePieces[5] = {
        friendlyPieces[PAWN]   & attackPawns[state.enemyColor()*64+posCapture],
        friendlyPieces[KNIGHT] & KnightMoves[posCapture],
        friendlyPieces[BISHOP] & fromCaseBishop,
        friendlyPieces[ROOK]   & fromCaseRook,
        friendlyPieces[QUEEN]  & (fromCaseBishop | fromCaseRook),
    };
    for(int piece=0; piece<KING; piece++){
        if(possiblePieces[piece]){
            ubyte pos[8];
            int nbPiece = places(possiblePieces[piece], pos);
            for(int p=0; p<nbPiece; p++){
                if(pinnedMasks[pos[p]] & captureMask){
                    LVAmove.piece = piece;
                    LVAmove.updateFrom(pos[p]);
                    if(piece == PAWN && (row(posCapture) == 0 || row(posCapture) == 7)){
                        LVAmove.updatePromotion(QUEEN);
                    }
                    return LVAmove;
                }
            }
        }
    }
    return nullMove;
}