#include "GameState.hpp"
#include "Const.hpp"
#include "Functions.hpp"
#include <cassert>
#include <string>
using namespace std;

big zobrist[nbZobrist];
static inline int posCastlingRook(bool side, bool c){
    return ((7-side*7)+c*(8*7));
}
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
    board.reset();
    zobristHash = 0;
    minorZobrist = 0;
    pawnZobrist = 0;
    turnNumber = 0;
    lastDoublePawnPush = -1;
    castlingMask = 0;
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
        movesSinceBeginning[0] = EnullMove;
    id += 2;
    castlingMask = 0;
    if(fen[id] == '-')
        id++;
    else{
        for(; id<(int)fen.size(); id++){
            if(fen[id] == ' ')break;

            bool isBlack = true;
            if(isupper(fen[id])){
                isBlack = false;
            }
            char position = tolower(fen[id]);
            int pos;
            if(position == 'q' || position == 'k'){
                int upto = posCastlingRook(position == 'k', isBlack);
                pos = __builtin_ctzll(directions[__builtin_ctzll(board.getMask(KING*2+isBlack))][upto]&board.getMask(ROOK*2+isBlack));
            }else{
                pos = isBlack*56+7-(position-'a');
            }
            castlingMask |= 1ULL << pos;
            zobristHash ^= zobrist[zobrCastle+pos];
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
    big Cmask = castlingMask;
    while(Cmask){
        int position = __builtin_ctzll(Cmask);
        bool isBlack = false;
        if(position >= 56){
            isBlack = true;
            position -= 56;
        }
        char c = (7-position)+'a';
        if(!isBlack)c = toupper(c);
        castlingPart += c;
        Cmask &= Cmask-1;
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


int GameState::friendlyColor() const{
    //Turn 1 is white (so friend on odd is white)
    return (turnNumber%2)?WHITE:BLACK;
}

int GameState::enemyColor() const{
    //Turn 1 is white (so enemy on odd is black)
    return (turnNumber%2)?BLACK:WHITE;
}

inline bool GameState::isEnPassantPossibility(const int piece, const Move& move){
    big sidePawn=((1ULL << clipped_left(move.to()))|(1ULL << clipped_right(move.to())));
    sidePawn &= getEnemyMask(PAWN);
    return piece == PAWN && 
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

void GameState::playNullMove(){
    movesSinceBeginning[turnNumber] = EnullMove;
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    if(lastDoublePawnPush != -1){
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
        lastDoublePawnPush = -1;
    }
    repHist[turnNumber] = zobristHash;
    rule50[turnNumber] = rule50[turnNumber-1]+1;
}
ExpendedMove GameState::getLastMove() const{
    if(turnNumber > 0 && (movesSinceBeginning[0].move.moveInfo != nullMove.moveInfo || turnNumber > 1))
        return movesSinceBeginning[turnNumber-1];
    return EnullMove;
}

ExpendedMove GameState::getContMove() const{
    if(turnNumber > 1 && (movesSinceBeginning[0].move.moveInfo != nullMove.moveInfo || turnNumber > 2))
        return movesSinceBeginning[turnNumber-2];
    return EnullMove;
}


Move GameState::playPartialMove(Move move){
    initMove(move);
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
            if(castlingMask & (1ULL << posCastlingRook(side, c)))
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
    int mover = getPiece(move.from());
    if(mover == PAWN && col(move.from()) != col(move.to()) && piece == SPACE){
        move.setFlag(Move::fep);
    }
    int colorPiece = color(board.mailbox[move.from()]);
    if(mover == KING && (board.colors[colorPiece] & (1ULL << move.to()))){
        move.setFlag(Move::fcastle);
    }if(mover == KING && !isdfrc && abs(move.from()-move.to()) == 2){
        move.setFlag(Move::fcastle);
        move.resetTo(move.to()+(move.to() > move.from()?2:-1));
    }
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

ExpendedMove GameState::playMove(Move move){
    if(lastDoublePawnPush != -1)
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
    rule50[turnNumber+1] = (rule50[turnNumber]+1)*!board.isChanger(move);
    const bool curColor=friendlyColor();
    const int piece = getPiece(move.from());
    const int toSquare = move.toMover();
    const int capture = board.getCapture(move);
    movesSinceBeginning[turnNumber] = {move, piece, capture};
    updateZobrists(piece, curColor, move.from());
    if(capture != SPACE){
        const bool enColor=enemyColor();
        if((1ULL << move.to()) & castlingMask*(capture == ROOK)){
            castlingMask ^= 1ULL << move.to();
            zobristHash ^= zobrist[zobrCastle+move.to()];
        }
        int pieceCapture = capture;
        int posCapture = move.to();
        if(move.getFlag() == Move::fep){
            if(enColor == BLACK)posCapture -= 8;
            else posCapture += 8;
        }
        updateZobrists(pieceCapture, enColor, posCapture);
        board.remPiece(posCapture, pieceCapture, enColor);
    }
    board.remPiece(move.from(), piece, curColor);
    if(isEnPassantPossibility(piece, move)){
        lastDoublePawnPush = 8 * ((row(move.from()) + row(move.to())) / 2) + col(move.from());
        zobristHash ^= zobrist[zobrPassant+col(lastDoublePawnPush)];
    }else{
        lastDoublePawnPush = -1;
    }
    if(piece == KING){
        big cM = castlingMask & mask_row[row(move.from())];
        if(cM){
            {
                int idx = __builtin_ctzll(cM);
                zobristHash ^= zobrist[zobrCastle+idx];
                castlingMask ^= 1ULL << idx;
            }
            cM &= cM-1;
            if(cM){
                int idx = __builtin_ctzll(cM);
                zobristHash ^= zobrist[zobrCastle+idx];
                castlingMask ^= 1ULL << idx;
            }
        }
        if(move.getFlag() == Move::fcastle){//castling
            int startRook = move.to();
            int endRook = toSquare+2*(move.from() > move.to())-1;
            updateZobrists(ROOK, curColor, startRook);
            updateZobrists(ROOK, curColor, endRook);
            board.remPiece(startRook, ROOK, curColor);
            board.addPiece(endRook  , ROOK, curColor);
        }
    }else if((1ULL << move.from()) & castlingMask*(piece == ROOK)){
        castlingMask ^= 1ULL << move.from();
        zobristHash ^= zobrist[zobrCastle+move.from()];
    }
    updateZobrists(piece|move.promotion(), curColor, toSquare);
    board.addPiece(toSquare, piece|move.promotion(), curColor);
    turnNumber++;
    zobristHash ^= zobrist[zobrTurn];
    repHist[turnNumber] = zobristHash;
    return movesSinceBeginning[turnNumber-1];
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