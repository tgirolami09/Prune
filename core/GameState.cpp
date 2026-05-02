#include "GameState.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include <cassert>
#include <string>
using namespace std;

big zobrist[nbZobrist];

__attribute__((constructor)) void init_zobrs(){
    big state(42);
    for(int idz=0; idz<nbZobrist; idz++){
        big z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        zobrist[idz] = z ^ (z >> 31);
    }
}

GameState::GameState(){
}

void GameState::updateZobrists(int piece, bool color, int square){
    big zobr = zobrist[(color*6+piece)*64+square];
    zobristHash ^= zobr;
    if(piece == PAWN)
        pawnZobrist ^= zobr;
    if(piece == KNIGHT || piece == BISHOP || piece == KING)
        minorZobrist ^= zobr;
}

void GameState::fromFen(string fen){
    zobristHash=0;
    pawnZobrist = 0;
    minorZobrist = 0;
    board.reset();
    int id=0;
    int dec=63;
    for(; id<(int)fen.size(); id++){
        char c=fen[id];
        if(isalpha(c)){
            int piece=piece_to_id.at(tolower(c));
            int color_p;
            if(isupper(c))
                color_p = WHITE;
            else
                color_p = BLACK;
            board.addPiece(dec, piece, color_p);
            updateZobrists(piece, color_p, dec);
            dec--;
        }else if(isdigit(c)){
            dec -= c-'0';
        }else if(c == ' ')break;
    }
    id++;
    turnNumber = fen[id] == 'w';
    if(!turnNumber)
        zobristHash ^= zobrist[zobrTurn];
    else
        movesSinceBeginning[0] = nullMove;
    id += 2;
    for(int side=0; side<2; side++)
        for(int kingside=0; kingside<2; kingside++)
            castlingRights[side][kingside] = 0;
    if(fen[id] == '-')
        id++;
    else{
        for(; id<(int)fen.size(); id++){
            if(fen[id] == ' ')break;

            bool isBlack = true;
            if(isupper(fen[id])){
                isBlack = false;
            }

            bool isKing = true;
            if(tolower(fen[id]) == 'q'){
                isKing = false;
            }
            castlingRights[isBlack][isKing] = 1;
            zobristHash ^= zobrist[zobrCastle+isBlack*2+isKing];
        }
    }
    for(int c=0; c<2; c++){
        for(int side=0; side<2; side++){
            posRook[c][side] = castlingRights[c][side] ? (7-side*7)+c*(8*7) : -1;
        }
    }
    id++;
    if(fen[id] == '-')lastDoublePawnPush = -1;
    else{
        lastDoublePawnPush = 7-(fen[id]-'a'), id++;
        lastDoublePawnPush += 8 * (fen[id] - '1'), id++;
    }
    //printf("In fen to data -> en passant goes to %d\n",lastDoublePawnPush);
    if(lastDoublePawnPush != -1)
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
    id += 2;
    int move50 = 0;
    while(id < (int)fen.size() && fen[id] != ' '){
        move50 = move50*10+fen[id]-'0';
        id++;
    }
    repHist[turnNumber] = zobristHash;
    rule50[turnNumber] = move50;
}

string GameState::toFen() const{
    string fen="";
    for(int row=0; row<8; row++){
        int nbSpace = 0;
        for(int col=0; col<8; col++){
            int p = getfullPiece(63-(row << 3 | col));
            if(type(p) == SPACE){
                nbSpace++;
            }else{
                if(nbSpace){
                    fen += (char)nbSpace+'0';
                    nbSpace = 0;
                }
                char c = id_to_piece[type(p)];
                if(color(p) == WHITE)c = toupper(c);
                fen += c;
            }
        }
        if(nbSpace)fen += (char)nbSpace+'0';
        if(row != 7)
            fen += "/";
    }
    fen += " ";
    if(friendlyColor() == WHITE)fen += "w";
    else fen += "b";
    fen += " ";
    string castlingPart="";
    for(int c=0; c<2; c++){
        for(int side=0; side < 2; side++){
            char s=side?'k':'q';
            if(!c)s = toupper(s);
            if(castlingRights[c][side]){
                castlingPart += s;
            }
        }
    }
    if(castlingPart == "")
        fen += "-";
    else fen += castlingPart;
    fen += " ";
    if(lastDoublePawnPush != -1){
        // fen += (char)7-lastDoublePawnPush+'a';
        // fen += friendlyColor() == WHITE?'6':'3';
        fen += 'h' - (lastDoublePawnPush%8);
        fen += '0'+ (lastDoublePawnPush / 8 + 1);
    }else fen += "-";
    fen += " ";
    fen += to_string(rule50[turnNumber]);
    fen += " ";
    fen += to_string(turnNumber);
    return fen;
}

big GameState::castlingMask(){
    big res = 0;
    for(int c=0; c<2; c++){
        for(int side=0; side<2; side++){
            if(castlingRights[c][side])
                res |= 1ULL << (56*c+7*!side);
        }
    }
    return res;
}

void GameState::castlingFromMask(big mask){
    for(int c=0; c<2; c++){
        for(int side=0; side<2; side++){
            castlingRights[c][side] = bool(mask & (1ULL << (56*c+7*!side)));
        }
    }

}

int GameState::friendlyColor() const{
    //Turn 1 is white (so friend on odd is white)
    return (turnNumber%2)?WHITE:BLACK;
}

int GameState::enemyColor() const{
    //Turn 1 is white (so enemy on odd is black)
    return (turnNumber%2)?BLACK:WHITE;
}

template<bool back>
inline bool GameState::isEnPassantPossibility(const Move& move){
    big sidePawn=((1ULL << clipped_left(move.to()))|(1ULL << clipped_right(move.to())));
    if(back)
        sidePawn &= getFriendlyMask(PAWN);
    else
        sidePawn &= getEnemyMask(PAWN);
    return move.piece == PAWN && 
        abs(move.from()-move.to()) == 2*8 && 
        sidePawn;
}

int GameState::rule50_count() const{
    return rule50[turnNumber];
}
bool GameState::twofold() const{
    const int minposs = max(0, turnNumber-rule50_count());
    for(int i=turnNumber-4; i >= minposs; i--){
        if(repHist[i] == repHist[turnNumber])
            return true;
    }
    return false;
}

bool GameState::threefold() const{
    bool alreadyRep=false;
    const int minposs = max(0, turnNumber-rule50_count());
    for(int i=turnNumber-4; i >= minposs; i--){
        if(repHist[i] == repHist[turnNumber]){
            if(alreadyRep)
                return true;
            alreadyRep = true;
        }
    }
    return false;
}

int GameState::playMove(Move move){
    if(lastDoublePawnPush != -1)
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
    const bool curColor=friendlyColor();
    updateZobrists(move.piece, curColor, move.from());
    if(move.capture != -2){
        const bool enColor=enemyColor();
        if(move.capture == ROOK){
            if(move.to() == posRook[enColor][0] && castlingRights[enColor][0]){
                zobristHash ^= zobrist[zobrCastle + enColor*2];
                castlingRights[enColor][0] = false;
            }else if(move.to() == posRook[enColor][1] && castlingRights[enColor][1]){
                zobristHash ^= zobrist[zobrCastle + enColor*2 + 1];
                castlingRights[enColor][1] = false;
            }
        }
        int pieceCapture = move.capture>=0?move.capture:PAWN;
        int posCapture = move.to();
        if(move.capture == -1){
            if(enColor == BLACK)posCapture -= 8;
            else posCapture += 8;
        }
        updateZobrists(pieceCapture, enColor, posCapture);
        board.remPiece(posCapture, pieceCapture, enColor);
    }
    if(move.promotion() == -1){
        updateZobrists(move.piece, curColor, move.to());
        board.addPiece(move.to(), move.piece, curColor);
    }else{
        board.addPiece(move.to(), move.promotion(), curColor);
        updateZobrists(move.promotion(), curColor, move.to());
    }
    board.remPiece(move.from(), move.piece, curColor);
    movesSinceBeginning[turnNumber] = move;
    if(isEnPassantPossibility<false>(move)){
        lastDoublePawnPush = 8 * ((row(move.from()) + row(move.to())) / 2) + col(move.from());
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
    }else{
        lastDoublePawnPush = -1;
    }
    if(move.piece == KING){
        if(castlingRights[curColor][0]){
            zobristHash ^= zobrist[zobrCastle + curColor*2];
            castlingRights[curColor][0] = false;
        }
        if(castlingRights[curColor][1]){
            zobristHash ^= zobrist[zobrCastle + curColor*2 + 1];
            castlingRights[curColor][1] = false;
        }
        if(abs(move.to()-move.from()) == 2){//castling
            int startRook=move.from(), endRook=move.to();
            if(move.from() > move.to()){//queen side
                startRook &= ~7;
                endRook++;
            }else{ //king side
                startRook |= 7;
                endRook--;
            }
            updateZobrists(ROOK, curColor, startRook);
            updateZobrists(ROOK, curColor, endRook);
            board.remPiece(startRook, ROOK, curColor);
            board.addPiece(endRook  , ROOK, curColor);
        }
    }else if(move.piece == ROOK){
        if(move.from() == posRook[curColor][0] && castlingRights[curColor][0]){
            zobristHash ^= zobrist[zobrCastle + curColor*2];
            castlingRights[curColor][0] = false;
        }else if(move.from() == posRook[curColor][1] && castlingRights[curColor][1]){
            zobristHash ^= zobrist[zobrCastle + curColor*2 + 1];
            castlingRights[curColor][1] = false;
        }
    }
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    repHist[turnNumber] = zobristHash;
    if(move.isChanger())
        rule50[turnNumber] = 0;
    else
        rule50[turnNumber] = rule50[turnNumber-1]+1;
    return 0;
}

void GameState::playNullMove(){
    movesSinceBeginning[turnNumber] = nullMove;
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    if(lastDoublePawnPush != -1){
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
        lastDoublePawnPush = -1;
    }
    repHist[turnNumber] = zobristHash;
    rule50[turnNumber] = rule50[turnNumber-1]+1;
}
Move GameState::getLastMove() const{
    if(turnNumber > 0 && (movesSinceBeginning[0].moveInfo != nullMove.moveInfo || turnNumber > 1))
        return movesSinceBeginning[turnNumber-1];
    return nullMove;
}

Move GameState::getContMove() const{
    if(turnNumber > 1 && (movesSinceBeginning[0].moveInfo != nullMove.moveInfo || turnNumber > 2))
        return movesSinceBeginning[turnNumber-2];
    return nullMove;
}


Move GameState::playPartialMove(Move move){
    int piece=getPiece(move.to());
    if(piece != SPACE){
        move.capture = piece;
    }
    int mover = getPiece(move.from());
    if(mover == PAWN && col(move.from()) != col(move.to()) && move.capture == -2){
        move.capture = -1;
    }
    move.piece = mover;
    playMove(move);
    return move;
}

int GameState::getPiece(int square) const{
    return type(board.mailbox[square]);
}

int GameState::getfullPiece(int square) const{
    return board.mailbox[square];
}
big GameState::getFriendlyMask(int piece) const{
    return board.getMask(piece, friendlyColor());
}
big GameState::getEnemyMask(int piece) const{
    return board.getMask(piece, enemyColor());
}

void GameState::print() const{
    printf("/−");
    for(int i=1; i<7; i++){
        printf("−−");
    }
    printf("−−\\\n");
    for(int row=0; row<8; row++){
        printf("|");
        for(int col=0; col<8; col++){
            int pos = 63-(row << 3 | col);
            int piece = getfullPiece(pos);
            char c;
            if(piece == SPACE*2)
                c = ' ';
            else{
                c=id_to_piece[type(piece)];
                if(color(piece) == WHITE){
                    c = toupper(c);
                }
            }
            printf("%c|", c);
        }
        printf("\n");
        if(row != 7){
            printf("|");
            for(int i=0; i<7; i++){
                printf("−+");
            }
            printf("−|\n");
        }
    }
    printf("\\−");
    for(int i=1; i<7; i++){
        printf("−−");
    }
    printf("−−/\n");
    for(int c=0; c<2; c++){
        for(int side=0; side < 2; side++){
            char s=side?'k':'q';
            if(!c)s = toupper(s);
            if(castlingRights[c][side])
                printf("%c", s);
        }
    }
    if(lastDoublePawnPush != -1){
        printf(" %c", 'h' - (lastDoublePawnPush%8));
        printf("%c", '0'+ (lastDoublePawnPush / 8 + 1));
    }
    printf("\n%16" PRIx64 "\n", zobristHash);
    printf("%s", toFen().c_str());
    printf("\n");
}

void GameState::initMove(Move& move){
    int piece=getPiece(move.to());
    if(piece != SPACE){
        move.capture = piece;
    }
    int mover = getPiece(move.from());
    if(mover == PAWN && col(move.from()) != col(move.to()) && move.capture == -2){
        move.capture = -1;
    }
    move.piece = mover;
}

// Inline zobrist update for forward-only move application
static inline void updateZobr(big& zobristHash, big& pawnZobrist, big& minorZobrist,
                               int piece, bool color, int square){
    big zobr = zobrist[(color*6+piece)*64+square];
    zobristHash ^= zobr;
    if(piece == PAWN)
        pawnZobrist ^= zobr;
    if(piece == KNIGHT || piece == BISHOP || piece == KING)
        minorZobrist ^= zobr;
}

// Forward-only playMove: same board/zobrist updates as playMove<false>()
// but with simplified castling (direct bool set, no counters).
void GameState::playMoveForward(Move move){
    // Clear en passant from zobrist
    if(lastDoublePawnPush != -1)
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];

    const bool curColor = friendlyColor();

    // Handle captures
    if(move.capture != -2){
        const bool enColor = enemyColor();
        int pieceCapture = move.capture >= 0 ? move.capture : PAWN;
        int posCapture = move.to();
        if(move.capture == -1){ // en passant
            if(enColor == BLACK) posCapture -= 8;
            else posCapture += 8;
        }
        updateZobr(zobristHash, pawnZobrist, minorZobrist,pieceCapture, enColor, posCapture);
        board.remPiece(posCapture, pieceCapture, enColor);

        // Rook captured: revoke castling rights directly (no deathRook needed)
        if(move.capture == ROOK){
            if(move.to() == posRook[enColor][0] && castlingRights[enColor][0]){
                zobristHash ^= zobrist[zobrCastle + enColor*2];
                castlingRights[enColor][0] = false;
            }else if(move.to() == posRook[enColor][1] && castlingRights[enColor][1]){
                zobristHash ^= zobrist[zobrCastle + enColor*2 + 1];
                castlingRights[enColor][1] = false;
            }
        }
    }

    // Move piece on board + update zobrists
    board.remPiece(move.from(), move.piece, curColor);
    if(move.promotion() == -1){
        updateZobr(zobristHash, pawnZobrist, minorZobrist,move.piece, curColor, move.to());
        board.addPiece(move.to(), move.piece, curColor);
    }else{
        board.addPiece(move.to(), move.promotion(), curColor);
        updateZobr(zobristHash, pawnZobrist, minorZobrist,move.promotion(), curColor, move.to());
    }
    updateZobr(zobristHash, pawnZobrist, minorZobrist,move.piece, curColor, move.from());

    // Store move for getLastMove()/getContMove()
    movesSinceBeginning[turnNumber] = move;

    // En passant possibility
    if(move.piece == PAWN && abs(move.from()-move.to()) == 16){
        big sidePawn = (1ULL << clipped_left(move.to())) | (1ULL << clipped_right(move.to()));
        if(sidePawn & getEnemyMask(PAWN)){
            lastDoublePawnPush = 8 * ((row(move.from()) + row(move.to())) / 2) + col(move.from());
            zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
        }else{
            lastDoublePawnPush = -1;
        }
    }else{
        lastDoublePawnPush = -1;
    }

    // King move: revoke all castling rights for this color
    if(move.piece == KING){
        if(castlingRights[curColor][0]){
            zobristHash ^= zobrist[zobrCastle + curColor*2];
            castlingRights[curColor][0] = false;
        }
        if(castlingRights[curColor][1]){
            zobristHash ^= zobrist[zobrCastle + curColor*2 + 1];
            castlingRights[curColor][1] = false;
        }
        // Castling: move the rook
        if(abs(move.to()-move.from()) == 2){
            int startRook = move.from(), endRook = move.to();
            if(move.from() > move.to()){ // queen side
                startRook &= ~7;
                endRook++;
            }else{ // king side
                startRook |= 7;
                endRook--;
            }
            updateZobr(zobristHash, pawnZobrist, minorZobrist,ROOK, curColor, startRook);
            updateZobr(zobristHash, pawnZobrist, minorZobrist,ROOK, curColor, endRook);
            board.remPiece(startRook, ROOK, curColor);
            board.addPiece(endRook  , ROOK, curColor);
        }
    }else if(move.piece == ROOK){
        // Rook moves from starting square: revoke that castling right
        if(move.from() == posRook[curColor][0] && castlingRights[curColor][0]){
            zobristHash ^= zobrist[zobrCastle + curColor*2];
            castlingRights[curColor][0] = false;
        }else if(move.from() == posRook[curColor][1] && castlingRights[curColor][1]){
            zobristHash ^= zobrist[zobrCastle + curColor*2 + 1];
            castlingRights[curColor][1] = false;
        }
    }

    // Advance turn
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    repHist[turnNumber] = zobristHash;
    if(move.isChanger())
        rule50[turnNumber] = 0;
    else
        rule50[turnNumber] = rule50[turnNumber-1]+1;
}

// Forward-only partial move (fills in piece/capture, then calls playMoveForward)
Move GameState::playPartialMoveForward(Move move){
    int piece=getPiece(move.to());
    if(piece != SPACE){
        move.capture = piece;
    }
    int mover = getPiece(move.from());
    if(mover == PAWN && col(move.from()) != col(move.to()) && move.capture == -2){
        move.capture = -1;
    }
    move.piece = mover;
    playMoveForward(move);
    return move;
}

// Forward-only null move
void GameState::playNullMoveForward(){
    movesSinceBeginning[turnNumber] = nullMove;
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    if(lastDoublePawnPush != -1){
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
        lastDoublePawnPush = -1;
    }
    repHist[turnNumber] = zobristHash;
    rule50[turnNumber] = rule50[turnNumber-1]+1;
}

// Optimized repetition detection: step by 2 since zobrist includes turn bit,
// so matches can only occur at same-parity indices.
bool GameState::twofoldFast(){
    const int minposs = max(0, turnNumber-rule50_count());
    for(int i=turnNumber-4; i >= minposs; i -= 2){
        if(repHist[i] == repHist[turnNumber])
            return true;
    }
    return false;
}

bool GameState::threefoldFast(){
    bool alreadyRep=false;
    const int minposs = max(0, turnNumber-rule50_count());
    for(int i=turnNumber-4; i >= minposs; i -= 2){
        if(repHist[i] == repHist[turnNumber]){
            if(alreadyRep)
                return true;
            alreadyRep = true;
        }
    }
    return false;
}
int GameState::material(){
    return
        countbit(board.pieces[PAWN  ])*1 +
        countbit(board.pieces[KNIGHT])*3 +
        countbit(board.pieces[BISHOP])*3 +
        countbit(board.pieces[ROOK  ])*5 +
        countbit(board.pieces[QUEEN ])*9;
}