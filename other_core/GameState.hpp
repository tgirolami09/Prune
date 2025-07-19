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
using namespace std;
const int zobrCastle=64*2*6;
const int zobrPassant=zobrCastle+4;
const int zobrTurn=zobrPassant+8;
const int nbZobrist=zobrTurn+1;

//Represents a state in the game
class GameState{
    bool isFinished = false;

    // (not necessary if we create new states for exploration)
    vector<Move> movesSinceBeginning;

    //To determine whose turn it is to play AND rules that involve turn count
    int turnNumber;

    bool castlingRights[2][2];

    //End of last double pawn push, (-1) if last move was not a double pawn push
    int lastDoublePawnPush;

    //Contains a bitboard of the white pieces, then a bitboard of the black pieces
    big zobrist[nbZobrist];
    short nbMoves[2][3];
    short posRook[2][2];
    short deathRook[2][2];
public : 
    big zobristHash;
    big boardRepresentation[2][6];
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

    const int friendlyColor() const{
        //Turn 1 is white (so friend on odd is white)
        return (turnNumber%2)?WHITE:BLACK;
    }

    const int enemyColor() const{
        //Turn 1 is white (so enemy on odd is black)
        return (turnNumber%2)?BLACK:WHITE;
    }

    //Returns the 6 bitboards of the FRIENDLY pieces on the board
    const big* friendlyPieces() const{
        int friendlyIndex = friendlyColor();
        return boardRepresentation[friendlyIndex];
    }

    //Returns the 6 bitboards of the ENEMY pieces on the board
    const big* enemyPieces() const{
        int enemyIndex = enemyColor();
        return boardRepresentation[enemyIndex];
    }
    template<bool enable, int side>
    void changeCastlingRights(int c){
        if(castlingRights[c][side] != enable)
            zobristHash ^= zobrist[zobrCastle+c*2+side];
        castlingRights[c][side] = enable;
    }
    template<bool back, int side>
    void updateCastlingRights(int c, int pos=-1){
        if(back)nbMoves[c][side]--;
        else nbMoves[c][side] ++;
        if(pos != -1)
            posRook[c][side] = pos;
        if(nbMoves[c][side] == 0){
            if(nbMoves[c][2] == 0)changeCastlingRights<true, side>(c);
        }else
            changeCastlingRights<false, side>(c);
    }

    template<bool back>
    void moveKing(int c){
        if(back)nbMoves[c][2]--;
        else nbMoves[c][2]++;
        if(nbMoves[c][2] == 0){
            if(nbMoves[c][0] == 0)changeCastlingRights<true, 0>(c);
            if(nbMoves[c][1] == 0)changeCastlingRights<true, 1>(c);
        }else{
            changeCastlingRights<false, 0>(c);
            changeCastlingRights<false, 1>(c);
        }
    }

    void captureRook(int pos, int c){
        int side=-1;
        if(pos == posRook[c][0]){
            side=0;
            updateCastlingRights<false, 0>(c);
        }else{
            side = 1;
            updateCastlingRights<false, 1>(c);
#ifdef ASSERT
            assert(pos == posRook[c][1]);
#endif
        }
        posRook[c][side] = -1;
        deathRook[c][side] = turnNumber;
    }

    void uncaptureRook(int pos, int c){
        int side=-1;
        if(turnNumber == deathRook[c][0]){
            side = 0;
            updateCastlingRights<true, 0>(c, pos);
        }else{
            side = 1;
            updateCastlingRights<true, 1>(c, pos);
#ifdef ASSERT
            assert(turnNumber == deathRook[c][1]);
#endif
        }
        deathRook[c][side] = -1;
    }

    //TODO : make it work for castle and test it for rest (I think en passant may work)
    template<bool back>
    void playMove(Move move){
        if(lastDoublePawnPush != -1)
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        const int curColor=friendlyColor();
        const int add=(curColor*6+move.piece)*64;
        if(move.promoteTo == -1){
            zobristHash ^= zobrist[add+move.end_pos];
            boardRepresentation[curColor][move.piece] ^= (1ULL << move.end_pos);
        }else{
            boardRepresentation[curColor][move.promoteTo] ^= (1ULL << move.end_pos);
            zobristHash ^= zobrist[(curColor*6+move.promoteTo)*64|move.end_pos];
        }
        boardRepresentation[curColor][move.piece] ^= (1ULL<<move.start_pos);
        zobristHash ^= zobrist[add|move.start_pos];
        if(move.capture != -2){
            const int enColor=enemyColor();
            if(move.capture == ROOK){
                if(back)
                    uncaptureRook(move.end_pos, enColor);
                else
                    captureRook(move.end_pos, enColor);
            }
            int pieceCapture = move.capture>=0?move.capture:PAWN;
            int indexCapture = (enColor*6+pieceCapture)*64;
            int posCapture = move.end_pos;
            if(move.capture == -1){
                if(enColor == BLACK)posCapture -= 8;
                else posCapture += 8;
            }
            zobristHash ^= zobrist[indexCapture|posCapture];//correction for en passant (not currently exact)
            boardRepresentation[enColor][pieceCapture] ^= 1ULL << posCapture;
        }
        if(!back)
            movesSinceBeginning.push_back(move);
        if(!back && move.piece == PAWN && abs(move.start_pos-move.end_pos) == 2*8){//move of 2 row = possibility of en passant
            lastDoublePawnPush = col(move.start_pos);
            zobristHash ^= zobrist[zobrPassant|lastDoublePawnPush];
        }else lastDoublePawnPush = -1;
        if(move.piece == KING){
            moveKing<back>(curColor);
            //zobristHash ^= zobrist[zobrCastle+2*curColor] ^ zobrist[zobrCastle+2*curColor+1];
            if(abs(move.end_pos-move.start_pos) == 2){//castling
                Move moveRook = {move.start_pos, move.end_pos, ROOK};
                if(move.start_pos > move.end_pos){//queen side ?
                    moveRook.start_pos &= ~7;
                    moveRook.end_pos++;
                }else{ //king side ?
                    moveRook.start_pos |= 7;
                    moveRook.end_pos--;
                }
                if(back)swap(moveRook.start_pos, moveRook.end_pos);
                if(moveRook.start_pos == posRook[curColor][0]){
                    updateCastlingRights<back, 0>(curColor, moveRook.end_pos);
                }else{
#ifdef ASSERT
                    assert(moveRook.start_pos == posRook[curColor][1]);
#endif
                    updateCastlingRights<back, 1>(curColor, moveRook.end_pos);//, (_piece != -1 || !back));
                }
                int indexZobr=(curColor*6+ROOK)*64;
                zobristHash ^= zobrist[indexZobr|moveRook.start_pos]^zobrist[indexZobr|moveRook.end_pos];
                boardRepresentation[curColor][ROOK] ^= (1ULL << moveRook.start_pos)|(1ULL << moveRook.end_pos);
                //playMove<true, !back>(moveRook);
            }
        }else if(move.piece == ROOK){
            if(back)swap(move.start_pos, move.end_pos);
            if(move.start_pos == posRook[curColor][0])
                updateCastlingRights<back, 0>(curColor, move.end_pos);//, (_piece != -1 || !back));
            else{
#ifdef ASSERT
                assert(move.start_pos == posRook[curColor][1]);
#endif
                updateCastlingRights<back, 1>(curColor, move.end_pos);//, (_piece != -1 || !back));
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
        playMove<true>(move); // playMove should be a lot similar to undoLastMove, so like this we just have to correct the little changements between undo and do
        if(movesSinceBeginning.size() > 0){
            Move nextMove=movesSinceBeginning.back();
            if(nextMove.piece == PAWN && abs(nextMove.end_pos-nextMove.start_pos) == 2*8){
                lastDoublePawnPush = col(nextMove.start_pos);
                zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
            }
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
        move.piece = mover;
        playMove<false>(move);
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