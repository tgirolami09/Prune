#ifndef GAMESATE_HPP
#define GAMESATE_HPP
#include "Move.hpp"
#include <cctype>
#include <string>
#include <vector>
#include <cstdlib>
#include "Const.hpp"
#include "Functions.hpp"
using namespace std;

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

    //TODO : implement this
    void fromFen(string fen){
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
                dec--;
            }else if(isdigit(c)){
                dec -= c-'0';
            }else if(c == ' ')break;
        }
        id++;
        turnNumber = fen[id] == 'b';
        id += 2;
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
            }
        }
        id++;
        if(fen[id] == '-')lastDoublePawnPush = -1;
        else lastDoublePawnPush = fen[id]-'a', id++;
        id += 2;
    }

    //TODO : implement this
    string toFen(){
        return "";
    }

    int friendlyColor(){
        //Turn 1 is white (so friend on odd is white)
        return (BLACK - ((turnNumber%2) * WHITE));
    }

    int enemyColor(){
        //Turn 1 is white (so enemy on odd is black)
        return ((turnNumber%2) * WHITE) + WHITE;
    }

    //Returns the 6 bitboards of the FRIENDLY pieces on the board
    big* friendlyPieces(){
        int friendlyIndex = (friendlyColor()/8)-1;

        return boardRepresentation[friendlyIndex];
    }

    //Returns the 6 bitboards of the ENEMY pieces on the board
    big* enemyPieces(){
        int enemyIndex = (enemyColor()/8)-1;

        return boardRepresentation[enemyIndex];
    }

    //TODO : (Make this update the representation)
    void playMove(Move move){

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