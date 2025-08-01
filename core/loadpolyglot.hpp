#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include "Move.hpp"
#include "GameState.hpp"
#include "polyglotHash.hpp"

using namespace std;

//Might be needed
uint16_t swap16(uint16_t x) {
    return __builtin_bswap16(x);
}

uint64_t swap64(uint64_t x) {
    return __builtin_bswap64(x);
}
uint32_t swap32(uint32_t x) {
    return __builtin_bswap32(x);
}

struct PolyglotEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;

    int to_square;
    int from_square; 
    int promotion; 

    void parse(){
        key = swap64(key);
        move = swap16(move);
        weight = swap16(weight);
        learn = swap32(learn);

        to_square = move & 0x3f;
        from_square = (move >> 6) & 0x3f;
        promotion = (move >> 12) & 0x7;
    }

    void printMove(){
        printf("Key = %llu : ",key);

        printf("going from %d to %d. Weight = %hd\n",from_square,to_square,weight);
        if (promotion != 0){
            printf("And promoting to %d\n",promotion);
        }
        if (learn != 0){
            printf("Learn is %d\n",learn);
        }

    }

    Move toMove(int movingPiece, int capturedPiece){
        Move newMove;
        newMove.start_pos = from_square;
        newMove.end_pos = to_square;
        if (promotion == 0){
            newMove.promoteTo = -1;
        }
        else{
            newMove.promoteTo = promotion;
        }
        newMove.piece = movingPiece;
        newMove.capture = capturedPiece;

        return newMove;
    }

    bool operator< (const PolyglotEntry& other)const{
        if (key == other.key){
            //Want moves with higher weights first
            return weight>other.weight;
        }
        else{
            return key<other.key;
        }
    }
};

unordered_map<uint64_t,PolyglotEntry> load_book(const string& filename) {
    vector<PolyglotEntry> InputBook;
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open book file");

    printf("Loading file '%s' for opening book\n",filename.c_str());

    while (file) {
        PolyglotEntry entry;
        file.read(reinterpret_cast<char*>(&entry.key), sizeof(entry.key));
        file.read(reinterpret_cast<char*>(&entry.move), sizeof(entry.move));
        file.read(reinterpret_cast<char*>(&entry.weight), sizeof(entry.weight));
        file.read(reinterpret_cast<char*>(&entry.learn), sizeof(entry.learn));
        entry.parse();
        InputBook.push_back(entry);
    }

    //Just in case
    sort(InputBook.begin(),InputBook.end());

    unordered_map<uint64_t,PolyglotEntry> book;

    uint64_t lastKey = 0;
    for (PolyglotEntry entry : InputBook){
        if (entry.key != lastKey){
            book[entry.key] = entry;
            lastKey = entry.key;
        }
    }

    printf("From %d entries to %d entries in the opening book\n",InputBook.size(),book.size());

    return book;
}

Move findPolyglot(GameState state, bool& inTable, unordered_map<uint64_t,PolyglotEntry> book){
    uint64_t gameHash = polyglotHash(state);

    Move bestMove;

    if (book.find(gameHash) != book.end()){
        PolyglotEntry bestPolyglotEntry = book[gameHash];
        //Determine moving piece
        int movingPiece = -1;
        for (int id = 0; id < 6; ++ id){
            if ((state.friendlyPieces()[id] & (1ul << bestPolyglotEntry.from_square)) != 0){
                movingPiece = id;
                break;
            }
        }

        int capturedPiece = -2;
        for (int id = 0; id < 6; ++ id){
            if ((state.enemyPieces()[id] & (1ul << bestPolyglotEntry.to_square)) != 0){
                capturedPiece = id;
                break;
            }
        }

        bestMove = book[gameHash].toMove(movingPiece,capturedPiece);
        inTable = true;
    }
    else{
        inTable = false;
    }

    return bestMove;
}