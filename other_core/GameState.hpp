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
const int nbZobrist=zobrPassant+1;

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
        turnNumber = fen[id] == 'b';
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

    //TODO : make it work for castle and test it for rest (I think en passant may work)
    void playMove(Move move){
        if(lastDoublePawnPush != -1)
            zobristHash ^= zobrist[zobrPassant+lastDoublePawnPush];
        int piece=getPiece(move.start_pos);
        int curColor=friendlyColor();
        int add=(curColor*6+piece)*64;
        zobristHash ^= zobrist[add+move.start_pos]^zobrist[add+move.end_pos];
        boardRepresentation[curColor][piece] ^= (1ULL<<move.start_pos)|(1ULL << move.end_pos);
        if(move.capture != -2){
            int enColor=enemyColor();
            int pieceCapture = move.capture>=0?move.capture:PAWN;
            int add = (enColor*6+pieceCapture)*64;
            int correction=0;
            if(move.capture == -1){
                correction = enColor == BLACK?-8:8;
            }
            int posCapture = move.end_pos+correction;
            zobristHash ^= zobrist[add+posCapture];//correction for en passant (not currently exact)
            boardRepresentation[enColor][pieceCapture] ^= 1ULL << posCapture;
        }

    }

    //TODO : (not necessary if we create new states for exploration)
    void undoLastMove(){

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
                int piece = SPACE;
                for(int i=0; i< 12; i++){
                    if(boardRepresentation[i >= nbPieces][i%nbPieces] & mask){
                        piece = i;
                        break;
                    }
                }
                char c;
                if(piece == SPACE)
                    c = ' ';
                else{
                    c=id_to_piece[type(piece)];
                    if(color(piece) == WHITE){
                        c = toupper(c);
                    }
                }
                printf("|");
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
    }
};
#endif