#include <cstdint>
#include <unordered_map>
#include "Move.hpp"
#include "GameState.hpp"

using namespace std;

//Might be needed
uint16_t swap16(uint16_t x);

uint64_t swap64(uint64_t x);
uint32_t swap32(uint32_t x);

struct PolyglotEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;

    int to_square;
    int from_square; 
    int promotion; 

    bool parse();
    void printMove();
    Move toMove(int movingPiece, int capturedPiece);
    bool operator< (const PolyglotEntry& other)const;
};

unordered_map<uint64_t,PolyglotEntry> load_book(const string& filename, bool mute);
Move findPolyglot(const GameState& state, bool& inTable, unordered_map<uint64_t,PolyglotEntry>& book);