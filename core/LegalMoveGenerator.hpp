#ifndef LEGALMOVEGENERATOR_HPP
#define LEGALMOVEGENERATOR_HPP
#include "Const.hpp"
#include "Functions.hpp"
#include "GameState.hpp"
#include <fstream>
#include <utility>
#include <cstring>
using namespace std;

big KnightMoves[64]; //Knight moves for each position of the board

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

big pieceCastlingMasks[2][2];

big attackCastlingMasks[2][2];

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

big normalKingMoves[64];

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
            kingEndMask |= (1ul << cardEnd);
            for (int j = 0; j < nbCol;++j){
                int trans2 = transitionsCol[j];

                int diagEnd = cardEnd + trans2;
                kingEndMask |= (1ul << diagEnd);

                int secondCardEnd = kingPosition + trans2;
                kingEndMask |= (1ul << secondCardEnd);
            }
        }
        normalKingMoves[kingPosition] = kingEndMask;
    }
}

big attackPawns[128];

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
                attackPawns[key] |= 1ul<<(square + 7 * moveFactor);
            if(pieceCol != rightLimit)
                attackPawns[key] |= 1ul<<(square + 9 * moveFactor);
        }
    }
}

big directions[64][64];
// big fullDir[64][8];
static constexpr int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
void precomputeDirections(){
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

class constTable{
public:
    int bits;
    int decR;
    big magic;
};

class LegalMoveGenerator{
    big* tableMagic[129];

    constTable constantsMagic[128];

    void load_table(string name){
        ifstream file(name);
        big magic;
        int decR, minimum, size;
        big mask;
        int current = 0;
        while(file >> magic){
            file >> decR >> minimum >> size;
            constantsMagic[current] = {minimum, decR, magic};
            tableMagic[current] = (big*)calloc(size, sizeof(big));
            for(int i=0; i<size; i++){
                file >> mask;
                tableMagic[current][i] = mask;
            }
            current++;
        }
    }
    
    public :

    LegalMoveGenerator(){
        PrecomputeKnightMoveData();
        load_table("magics.out");
        init_lines();
        precomputePawnsAttack();
        precomputeCastlingMasks();
        precomputeNormlaKingMoves();
        precomputeDirections();
    }
    ~LegalMoveGenerator(){
        for(int i=0; i<128; i++){
            free(tableMagic[i]);
        }
    }

    private:

    const int doubleCheckFromSameType = -100;

    //A mask for allowed squares given pins for every piece on the board
    big pinnedMasks[64];

    template<bool isPawn>
    void maskToMoves(int start, big mask, Move* moves, int& nbMoves, int8_t piece){
        while(mask){
            int bit = __builtin_ctzll(mask);
            mask &= mask-1;
            Move base = {(int8_t)start, (int8_t)bit, piece};
            big mask = 1ULL << bit;
            //There is a capture
            if(mask&allEnemies)
                for(int i=0; i<6; i++)
                    if(enemyPieces[i]&mask)base.capture = i;
            if(isPawn && (row(bit) == 7 || row(bit) == 0)){
                for(int8_t typePiece:{KNIGHT, BISHOP, ROOK, QUEEN}){
                    moves[nbMoves] = base;
                    moves[nbMoves].promoteTo = typePiece;
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

    inline big moves_table(int index, big mask_pieces){
        return tableMagic[index][(mask_pieces*constantsMagic[index].magic & (MAX_BIG >> constantsMagic[index].decR)) >> (64-constantsMagic[index].decR-constantsMagic[index].bits)];
    }

    inline big pseudoLegalBishopMoves(int bishopPosition, big allPieces){
        // big bishopMoveMask=moves_table(bishopPosition, allPieces&mask_empty_bishop(bishopPosition));
        // return bishopMoveMask;
        return moves_table(bishopPosition, allPieces&mask_empty_bishop(bishopPosition));
    }

    inline big pseudoLegalRookMoves(int rookPosition, big allPieces){
        // big rookMoveMask=moves_table(rookPosition+64, allPieces&mask_empty_rook(rookPosition));
        // return rookMoveMask;
        return moves_table(rookPosition+64, allPieces&mask_empty_rook(rookPosition));
    }

    inline big pseudoLegalQueenMoves(int queenPositions, big allPieces){
        // big bishopMovesFromQueen = pseudoLegalBishopMoves(queenPositions, allPieces);
        // big rookMovesFromQueen = pseudoLegalRookMoves(queenPositions ,allPieces);
       
        // return bishopMovesFromQueen | rookMovesFromQueen;

        return pseudoLegalBishopMoves(queenPositions, allPieces) | pseudoLegalRookMoves(queenPositions ,allPieces);;
    }

    inline big pseudoLegalKnightMoves(int knightPosition){
        // big knightEndMask = KnightMoves[knightPosition];
        // return knightEndMask;
        return KnightMoves[knightPosition];
    }

    big pseudoLegalPawnMoves(int pawnPosition, bool color, big& allPieces, int friendKingPos, big moveMask = -1, big captureMask = -1, big enemyPieces = -1, int enPassant = -1, big enemyRooks = 0){
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

        if (enPassant != -1 && ((pawnAttackMask & (1ul << enPassant)) != 0) && ((captureMask & (1ul << (enPassant + (-1 * 8 * moveFactor)))) != 0)){
            //This means en-passant actually captures the pawn that is causing check

            big kingAsRook = pseudoLegalRookMoves(friendKingPos,allPieces ^ ((1ul << pawnPosition) | (1ul << (enPassant + (-1 * 8 * moveFactor)))));
            if((row(friendKingPos) != row(enPassant + (8 * moveFactor * -1))) | ((kingAsRook&enemyRooks) == 0)){
                finalMoveMask |= (1ul << enPassant);
            }
        }

        return finalMoveMask;
    }  
    
    big pseudoLegalKingMoves(int kingPosition,const big& allPieces, bool color, bool kingCastling, bool queenCastling, big& dangerSquares){
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

    int dealWithEnemyPawns(big enemyPawnPositions, int friendKingPos, int enemyColor ,big& allDangerSquares){
        int moveFactor = enemyColor ? -1 : 1;

        int checkerPos = -1;

        //Attacks to the left
        const big col1 = (1ul << 7) + (1ul << 15) + (1ul << 23) + (1ul << 31) + (1ul << 39) + (1ul << 47) + (1ul << 55) + (1ul << 63);
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

        if (attacksToTheLeft & (1ul << friendKingPos)){
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
        if (attacksToTheRight & (1ul << friendKingPos)){
            checkerPos = friendKingPos + (8 * moveFactor * -1) + 1;
        }
 
        return checkerPos;
    }

    int dealWithEnemyKnights(big enemyKnightPositions, int friendKingPos, big& allDangerSquares){
        int checkerPos = -1;
        ubyte positions[10];
        int nbPositions = places(enemyKnightPositions, positions);
        for (int i = 0; i < nbPositions; ++i){
            int currentKnightPos = positions[i];
            big dangerSquares = pseudoLegalKnightMoves(currentKnightPos);

            allDangerSquares |= dangerSquares;

            if (dangerSquares & (1ul << friendKingPos)){
                //Knight is giving check (only 1 knight can give check)
                checkerPos = currentKnightPos;
            }
        }
        return checkerPos;
    }

    int dealWithEnemyBishops(big enemyBishopPositions, big& allPieces, int friendKingPos, big& allDangerSquares){
        int checkerPos = -1;
        ubyte positions[11];
        int nbPositions = places(enemyBishopPositions, positions);
        for (int i = 0; i < nbPositions; ++i){
            int currentBishopPos = positions[i];
            big dangerSquares = pseudoLegalBishopMoves(currentBishopPos, allPieces ^ (1ul << friendKingPos));

            allDangerSquares |= dangerSquares;

            if (dangerSquares & (1ul << friendKingPos)){
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
                big pinnedPieceMask = (ray ^ (1ul << currentBishopPos)) & allPieces;
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

    int dealWithEnemyRooks(big enemyRookPositions, big& allPieces, int friendKingPos, big& allDangerSquares){
        int checkerPos = -1;
        ubyte positions[11];
        int nbPositions = places(enemyRookPositions, positions);
        for (int i = 0; i < nbPositions; ++i){
            int currentRookPos = positions[i];
            big dangerSquares = pseudoLegalRookMoves(currentRookPos, allPieces ^ (1ul << friendKingPos));

            allDangerSquares |= dangerSquares;

            if (dangerSquares & (1ul << friendKingPos)){
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
                big pinnedPieceMask = (ray ^ (1ul << currentRookPos)) & allPieces;
                //There is a pinned piece;
                if (countbit(pinnedPieceMask) == 1){
                    int pinnedPiecePos = __builtin_ctzll(pinnedPieceMask);
                    pinnedMasks[pinnedPiecePos] = ray;  
                }
            }
        }
        return checkerPos; 
    }

    void dealWithEnemyKing(int enemyKingPos, big& allDangerSquares){
        //The 4 0's are to prevent from generating castling
        big dangerSquares = pseudoLegalKingMoves(enemyKingPos,0,0,0,0,allDangerSquares);
        allDangerSquares |= dangerSquares;
    }

    void legalKingMoves(const GameState& state, Move* moves, int& nbMoves, big friendlyPieces, big allPieces, big dangerSquares, big captureMask = -1){
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

    void legalPawnMoves(big pawnMask, bool friendlyColor, int lastDoublePawnPush, big moveMask, big captureMask, Move* pawnMoves, int& nbMoves, big allPieces, big allEnemies, big enemyRooks){
        ubyte pos[8];
        int nbPos=places(pawnMask, pos);
        for (int p = 0;p<nbPos;++p){
            big pawnMoveMask = pseudoLegalPawnMoves(pos[p], friendlyColor, allPieces, friendlyKingPosition, moveMask, captureMask, allEnemies, lastDoublePawnPush, enemyRooks);
            // printf("Pawn at pos %d wants to move to :\n",pos[p]);
            // print_mask(pawnMoveMask);
            pawnMoveMask &= pinnedMasks[pos[p]];
            // if(pinnedMasks[pos[p]] != -1ul){
            //     printf("Pinn mask for piece:\n");
            //     print_mask(pinnedMasks[pos[p]]);
            //     printf("After pin and:\n");
            //     print_mask(pawnMoveMask);
            // }

            maskToMoves<true>(pos[p], pawnMoveMask, pawnMoves, nbMoves, PAWN);
        }
    }

    void legalKnightMoves(big knightMask, big moveMask, big captureMask, Move* knightMoves, int& nbMoves){
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

    void legalSlidingMoves(big moveMask, big captureMask, Move* slidingMoves, int& nbMoves, big allPieces){
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

    const big* friendlyPieces;
    const big* enemyPieces;
    big allFriends;
    big allEnemies;
    big allPieces;

    int friendlyKingPosition;
    int enemyyKingPosition;

    public : int generateLegalMoves(const GameState& state, bool& inCheck, Move* legalMoves, bool onlyCapture=false){
        //Set all pinned masks to -1 (= no pinning)
        memset(pinnedMasks, 0xFF, sizeof(pinnedMasks));

        //All allowed spots for a piece to move (not allowed if king is in check)
        big moveMask = -1; //Totaly true
        //All allowed spots for a piece to capture another one (not allowed if there is a checker)
        big captureMask = -1; //Totaly false

        int nbMoves = 0;
        int nbCheckers = 0;
        int checkerPos = -1;
        int checkerType = -1;

        friendlyPieces = state.friendlyPieces();
        enemyPieces = state.enemyPieces();

        friendlyKingPosition = __builtin_ctzll(friendlyPieces[KING]);
        enemyyKingPosition = __builtin_ctzll(enemyPieces[KING]);

        allFriends = 0;
        allEnemies = 0;
        for (int i = 0; i < 6; ++i){
            big boardFriend = friendlyPieces[i];
            allFriends |= boardFriend;
            big boardEnemy = enemyPieces[i];
            allEnemies |= boardEnemy;
        }

        allPieces = allFriends | allEnemies;

        moveMask = (~allFriends);
        captureMask = allEnemies;

        big allDangerSquares = 0;

        dealWithEnemyKing(enemyyKingPosition,allDangerSquares);

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
                checkerType = BISHOP;
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
                checkerType = ROOK;
            }
        }

        //From here we have the pinned pieces, the number of checkers, and the danger squares
        if(onlyCapture){
            moveMask = 0;
            legalKingMoves(state, legalMoves, nbMoves, allFriends, allPieces, allDangerSquares, captureMask);
        }
        else{
            legalKingMoves(state, legalMoves, nbMoves, allFriends, allPieces, allDangerSquares);
        }

        if (nbCheckers == 2){
            inCheck = true;
            //Because if there are two checkers than only king moves are interesting
            return nbMoves;
        }

        else if (nbCheckers == 1){
            inCheck = true;
            captureMask = (1ul << checkerPos);
            if (checkerType == BISHOP){
                moveMask &= pseudoLegalBishopMoves(checkerPos, allPieces) & pseudoLegalBishopMoves(friendlyKingPosition, allPieces);
            }
            else if (checkerType == ROOK){
                moveMask &= pseudoLegalRookMoves(checkerPos, allPieces) & pseudoLegalRookMoves(friendlyKingPosition, allPieces);
            }
            else{
                moveMask = 0;
            }
        }

        legalPawnMoves(friendlyPieces[PAWN], state.friendlyColor(), state.lastDoublePawnPush, moveMask, captureMask, legalMoves, nbMoves, allPieces, allEnemies, enemyPieces[ROOK]);
        legalKnightMoves(friendlyPieces[KNIGHT], moveMask, captureMask, legalMoves, nbMoves);
        legalSlidingMoves(moveMask, captureMask, legalMoves, nbMoves, allPieces);
        return nbMoves;
    }
};

#endif