#ifndef GAMESATE_HPP
#define GAMESATE_HPP
#include "Move.hpp"
#include <cctype>
#include <string>
#include <vector>
#include <cstdlib>
#include "Const.hpp"
#include "Functions.hpp"
#include <random>
#include <cassert>
using namespace std;
const int zobrCastle=64*2*6;
const int zobrPassant=zobrCastle+4;
const int zobrTurn=zobrPassant+8;
const int nbZobrist=zobrTurn+1;

//Represents a state in the game
class GameState{
    public : bool isFinished = false;

    // (not necessary if we create new states for exploration)
    vector<Move> movesSinceBeginning;

    //To determine whose turn it is to play AND rules that involve turn count
    int turnNumber;

    bool castlingRights[2][2];

    //End of last double pawn push, (-1) if last move was not a double pawn push
    int lastDoublePawnPush;

    //Contains a bitboard of the white pieces, then a bitboard of the black pieces
    big boardRepresentation[2][6];
    big zobrist[nbZobrist];
    big zobristHash;
    short nbMoves[2][3];
    short posRook[2][2];
    short deathRook[2][2];

    GameState(){
        mt19937_64 gen(42);
        uniform_int_distribution<big> dist(0, MAX_BIG);
        for(int idz=0; idz<nbZobrist; idz++){
            zobrist[idz] = dist(gen);
        }
    }

    //TODO : implement this
    void fromFen(string fen){
        zobristHash=0;
        for(int c=0; c<2; c++)
            for(int p=0; p<6; p++)
                boardRepresentation[c][p] = 0;
        int id=0;
        int dec=63;
        for(; id<fen.size(); id++){
            char c=fen[id];
            if(isalpha(c)){
                int piece=piece_to_id[tolower(c)];
                int color_p;
                if(isupper(c))
                    color_p = WHITE;
                else
                    color_p = BLACK;
                boardRepresentation[color_p][piece] |= 1ULL << dec;
                zobristHash ^= zobrist[(color_p*6+piece)*64+dec];
                dec--;
            }else if(isdigit(c)){
                dec -= c-'0';
            }else if(c == ' ')break;
        }
        id++;
        turnNumber = fen[id] == 'w';
        if(!turnNumber)
            zobristHash ^= zobrist[zobrTurn];
        id += 2;
        for(int side=0; side<2; side++)
            for(int kingside=0; kingside<2; kingside++)
                castlingRights[side][kingside] = 0;
        if(fen[id] == '-')
            id++;
        else{
            for(; id<fen.size(); id++){
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
            bool allOk=true;
            for(int side=0; side<2; side++){
                if(castlingRights[c][side]){
                    nbMoves[c][side] = 0;
                    deathRook[c][side] = -1;
                    posRook[c][side] = side*7+c*(8*7);
                }else{
                    allOk=false;
                    //TODO: complete for rook pos, death etc.
                }
            }
            if(allOk){
                nbMoves[c][2] = 0;
            }else{
                nbMoves[c][2] = 1;
            }
        }
        id++;
        if(fen[id] == '-')lastDoublePawnPush = -1;
        else lastDoublePawnPush = fen[id]-'a', id++;
        if(lastDoublePawnPush != -1)
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        id += 2;
    }

    //TODO : implement this
    string toFen(){
        return "";
    }

    int friendlyColor(){
        //Turn 1 is white (so friend on odd is white)
        return (turnNumber%2)?WHITE:BLACK;
    }

    int enemyColor(){
        //Turn 1 is white (so enemy on odd is black)
        return (turnNumber%2)?BLACK:WHITE;
    }

    //Returns the 6 bitboards of the FRIENDLY pieces on the board
    big* friendlyPieces(){
        int friendlyIndex = friendlyColor();
        return boardRepresentation[friendlyIndex];
    }

    //Returns the 6 bitboards of the ENEMY pieces on the board
    big* enemyPieces(){
        int enemyIndex = enemyColor();
        return boardRepresentation[enemyIndex];
    }

    void changeCastlingRights(int c, int side, bool enable=false){
        if(castlingRights[c][side] != enable)
            zobristHash ^= zobrist[zobrCastle+c*2+side];
        castlingRights[c][side] = enable;
    }

    void updateCastlingRights(int c, int side, bool back, int pos=-1, bool change=true){
        int add=1;
        if(back)add = -1;
        nbMoves[c][side] += add;
        if(pos != -1)
            posRook[c][side] = pos;
        if(change){
            if(nbMoves[c][side] == 0)
                changeCastlingRights(c, side, true);
            else
                changeCastlingRights(c, side);
        }
    }

    void moveKing(int c, bool back){
        int add=1;
        if(back)add=-1;
        nbMoves[c][2] += add;
        updateCastlingRights(c, 0, back);
        updateCastlingRights(c, 1, back);
    }

    void captureRook(int pos, int c){
        int side=-1;
        if(pos == posRook[c][0])
            side=0;
        else{
            side = 1;
            assert(pos == posRook[c][1]);
        }
        posRook[c][side] = -1;
        deathRook[c][side] = turnNumber;
        updateCastlingRights(c, side, false);
    }

    void uncaptureRook(int pos, int c){
        int side=-1;
        if(turnNumber == deathRook[c][0]){
            side = 0;
        }else{
            side = 1;
            assert(turnNumber == deathRook[c][1]);
        }
        deathRook[c][side] = -1;
        updateCastlingRights(c, side, true, pos);
    }

    //TODO : make it work for castle and test it for rest (I think en passant may work)
    void playMove(Move move, bool back=false, int _piece=-1){
        if(lastDoublePawnPush != -1)
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        int piece=_piece;
        if(piece == -1){
            if(back)
                piece = ROOK; //it's when castling
            else
                piece=getPiece(move.start_pos);
        }
        int curColor=friendlyColor();
        int add=(curColor*6+piece)*64;
        if(move.promoteTo == -1){
            zobristHash ^= zobrist[add+move.end_pos];
            boardRepresentation[curColor][piece] ^= (1ULL << move.end_pos);
        }else{
            boardRepresentation[curColor][move.promoteTo] ^= (1ULL << move.end_pos);
            zobristHash ^= zobrist[(curColor*6+move.promoteTo)*64+move.end_pos];
            piece = PAWN;
        }
        boardRepresentation[curColor][piece] ^= (1ULL<<move.start_pos);
        zobristHash ^= zobrist[(curColor*6+piece)*64+move.start_pos];
        if(move.capture != -2){
            int enColor=enemyColor();
            if(move.capture == ROOK){
                if(back)
                    uncaptureRook(move.end_pos, enColor);
                else
                    captureRook(move.end_pos, enColor);
            }
            int pieceCapture = move.capture>=0?move.capture:PAWN;
            int add = (enColor*6+pieceCapture)*64;
            int correction=0;
            if(move.capture == -1){
                correction = (enColor == BLACK)?-8:8;
            }
            int posCapture = move.end_pos+correction;
            zobristHash ^= zobrist[add+posCapture];//correction for en passant (not currently exact)
            boardRepresentation[enColor][pieceCapture] ^= 1ULL << posCapture;
        }
        if(!back)
            movesSinceBeginning.push_back(move);
        if(!back && piece == PAWN && abs(move.start_pos-move.end_pos) == 2*8){//bouge de 2 colonnes
            lastDoublePawnPush = col(move.start_pos);
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        }else lastDoublePawnPush = -1;
        if(piece == KING){
            moveKing(curColor, back);
            zobristHash ^= zobrist[zobrCastle+2*curColor] ^ zobrist[zobrCastle+2*curColor+1];
            if(abs(move.end_pos-move.start_pos) == 2){//castling
                int pieceCastle = -1;
                if(back)pieceCastle=ROOK;
                if(move.start_pos > move.end_pos)//queen side ?
                    playMove({move.start_pos&~7, move.end_pos+1}, true, pieceCastle);
                else //king side ?
                    playMove({move.start_pos|7, move.end_pos-1}, true, pieceCastle);
                //return;
            }
        }if(piece == ROOK){//avoid to do the castling rights a second time when it castling
            //printf("%d %d\n", move.start_pos, move.end_pos);
            if(back && _piece != -1)swap(move.start_pos, move.end_pos);
            //printf("%d %d %d %d\n", move.start_pos, move.end_pos, posRook[curColor][0], posRook[curColor][1]);
            if(move.start_pos == posRook[curColor][0])
                updateCastlingRights(curColor, 0, back && _piece != -1, move.end_pos);//, (_piece != -1 || !back));
            else{
                assert(move.start_pos == posRook[curColor][1]);
                updateCastlingRights(curColor, 1, back && _piece != -1, move.end_pos);//, (_piece != -1 || !back));
            }
        }
        if(!back){
            turnNumber++;
            zobristHash ^= zobrist[zobrTurn];
        }
    }

    //TODO : (not necessary if we create new states for exploration)
    void undoLastMove(){
        turnNumber--;
        zobristHash ^= zobrist[zobrTurn];
        Move move=movesSinceBeginning.back();
        movesSinceBeginning.pop_back();
        playMove(move, true, getPiece(move.end_pos)); // playMove should be a lot similar to undoLastMove, so like this we just have to correct the little changements between undo and do
        Move nextMove=movesSinceBeginning.back();
        if(getPiece(nextMove.end_pos) == PAWN && abs(nextMove.end_pos-nextMove.start_pos) == 2*8){
            lastDoublePawnPush = col(nextMove.start_pos);
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        }
    }

    void playPartialMove(Move move){
        int piece=getPiece(move.end_pos);
        if(piece != SPACE){
            move.capture = piece;
        }
        int mover = getPiece(move.start_pos);
        if(mover == PAWN && col(move.start_pos) != col(move.end_pos) && move.capture == -2){
            move.capture = -1;
        }
        printf("%s\n", move.to_str().c_str());
        playMove(move);
    }

    int getPiece(int square){
        big mask=1ULL << square;
        for(int c=0; c<2; c++){
            for(int p=0; p<nbPieces; p++){
                if(mask&boardRepresentation[c][p])return p;
            }
        }
        return SPACE;
    }

    void print(){
        printf("/−");
        for(int i=1; i<7; i++){
            printf("−−");
        }
        printf("−−\\\n");
        for(int row=0; row<8; row++){
            printf("|");
            for(int col=0; col<8; col++){
                big mask = 1ULL << (63-(row << 3 | col));
                int piece = SPACE*2;
                for(int i=0; i< 12; i++){
                    if(boardRepresentation[i%2][i/2] & mask){
                        piece = i;
                        break;
                    }
                }
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
                char s=side?'q':'k';
                if(!c)s = toupper(s);
                if(castlingRights[c][side])
                    printf("%c", s);
            }
        }
        if(lastDoublePawnPush != -1){
            printf(" %c", 7-lastDoublePawnPush+'a');
        }
        printf("\n%16llx", zobristHash);
        printf("\n");
    }
};
#endif