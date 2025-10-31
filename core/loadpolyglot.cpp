#include "loadpolyglot.hpp"
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

bool PolyglotEntry::parse(){
    key = swap64(key);
    move = swap16(move);
    weight = swap16(weight);
    learn = swap32(learn);

    to_square = move & 0x3f;
    to_square = row(to_square) * 8 + (7 - col(to_square));
    from_square = (move >> 6) & 0x3f;
    from_square = row(from_square) * 8 + (7 - col(from_square));
    promotion = (move >> 12) & 0x7;

    // The move was parsed as a null move
    if (from_square == to_square){
        return false;
    }

    else{
        return true;
    }
}

void PolyglotEntry::printMove(){
    printf("Key = %" PRIu64 " : ",key);

    printf("going from %d to %d. Weight = %hd\n",from_square,to_square,weight);
    if (promotion != 0){
        printf("And promoting to %d\n",promotion);
    }
    if (learn != 0){
        printf("Learn is %d\n",learn);
    }

}

Move PolyglotEntry::toMove(int movingPiece, int capturedPiece){
    //Deal with chess960 notation for castling
    if (movingPiece == KING){
        if (to_square == 7 && from_square == 3){
            to_square = 5;
        }
        else if (to_square == 0 && from_square == 3){
            to_square = 1;
        }
        else if (to_square == 63 && from_square == 59){
            to_square = 61;
        }
        else if (to_square == 56 && from_square == 59){
            to_square = 57;
        }
        
    }
    Move newMove;
    // newMove.start_pos = from_square;
    // newMove.end_pos = to_square;
    newMove.updateFrom(from_square);
    newMove.updateTo(to_square);
    // if (promotion == 0){
        // newMove.promoteTo = -1;
    // }
    if (promotion != 0){
        // newMove.promoteTo = promotion;
        newMove.updatePromotion(promotion);
    }
    newMove.piece = movingPiece;
    newMove.capture = capturedPiece;

    return newMove;
}

bool PolyglotEntry::operator< (const PolyglotEntry& other)const{
    if (key == other.key){
        //Want moves with higher weights first
        return weight>other.weight;
    }
    else{
        return key<other.key;
    }
}

unordered_map<uint64_t,PolyglotEntry> load_book(const string& filename, bool mute) {
    vector<PolyglotEntry> InputBook;
    unordered_map<uint64_t,PolyglotEntry> book;
    //Stores the amount of moves that where parsed as null
    int nullMoveAmount = 0;
    ifstream file(filename, ios::binary);
    if (!file.is_open()){
        //throw runtime_error("Cannot open book file");
        if(!mute)
            printf("info string Book file could not be openened, returning empty book\n");
        return book;
    }
    if(!mute)
        printf("info string Loading file '%s' for opening book\n",filename.c_str());

    while (file) {
        PolyglotEntry entry;
        file.read(reinterpret_cast<char*>(&entry.key), sizeof(entry.key));
        file.read(reinterpret_cast<char*>(&entry.move), sizeof(entry.move));
        file.read(reinterpret_cast<char*>(&entry.weight), sizeof(entry.weight));
        file.read(reinterpret_cast<char*>(&entry.learn), sizeof(entry.learn));
        bool succesfulParse = entry.parse();
        if (succesfulParse){
            InputBook.push_back(entry);
        }
        else{
            ++nullMoveAmount;
        }
    }

    //Just in case
    sort(InputBook.begin(),InputBook.end());

    uint64_t lastKey = 0;
    for (PolyglotEntry entry : InputBook){
        if (entry.key != lastKey){
            book[entry.key] = entry;
            lastKey = entry.key;
        }
    }
    if(!mute){
        printf("info string From %" PRId64 " entries to %" PRId64 " entries in the opening book\n",InputBook.size(),book.size());
        printf("info string Parsed %d moves as null moves (they where consequently ignored)\n",nullMoveAmount);
    }

    return book;
}

Move findPolyglot(const GameState& state, bool& inTable, unordered_map<uint64_t,PolyglotEntry>& book){
    uint64_t gameHash = polyglotHash(state);

    Move bestMove;

    auto moveIt = book.find(gameHash);
    if (moveIt != book.end()){
        PolyglotEntry bestPolyglotEntry = moveIt->second;
        //Determine moving piece
        int movingPiece = -1;
        for (int id = 0; id < 6; ++ id){
            if ((state.friendlyPieces()[id] & (1ull << bestPolyglotEntry.from_square)) != 0){
                movingPiece = id;
                break;
            }
        }

        int capturedPiece = -2;
        for (int id = 0; id < 6; ++ id){
            if ((state.enemyPieces()[id] & (1ull << bestPolyglotEntry.to_square)) != 0){
                capturedPiece = id;
                break;
            }
        }

        bestMove = bestPolyglotEntry.toMove(movingPiece,capturedPiece);
        inTable = true;
    }
    else{
        inTable = false;
    }

    return bestMove;
}