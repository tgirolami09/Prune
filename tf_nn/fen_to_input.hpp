#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <cctype>

#ifndef TF_INPUT_SIZE_VAR
#define TF_INPUT_SIZE_VAR
// 768 board + 1 turn + 1 static_eval
const int TF_INPUT_SIZE = 770; 
#endif

std::vector<float> fenToInput(const std::string& fen, float static_eval) {
    std::vector<float> input(TF_INPUT_SIZE, 0.0f);

    std::unordered_map<char, int> PIECE_TO_PLANE = {
        {'P', 0}, {'N', 1}, {'B', 2}, {'R', 3}, {'Q', 4}, {'K', 5},
        {'p', 6}, {'n', 7}, {'b', 8}, {'r', 9}, {'q', 10}, {'k', 11}
    };

    std::istringstream ss(fen);
    std::string boardStr, turn, rest;
    ss >> boardStr >> turn;
    // Ignore rest of FEN

    int sq = 0;
    int rankIdx = 0;
    int fileIdx = 0;
    for (char c : boardStr) {
        if (c == '/'){
            fileIdx = 0;
            rankIdx++;
            continue;
        }
        if (std::isdigit(c)) {
            sq += (c - '0');
            fileIdx += (c - '0');
        } else {
            auto it = PIECE_TO_PLANE.find(c);
            if (it != PIECE_TO_PLANE.end()) {
                // int idx = it->second * 64 + sq;
                int idx = rankIdx*8*12 + fileIdx*12 + it->second;
                input[idx] = 1.0f;
            }
            sq++;
            fileIdx++;
        }
    }

    // turn to move
    input[768] = (turn == "w" ? 1.0f : 0.0f);
    // static_eval
    input[769] = static_eval;

    // printf("Data as a 1D vector of floats\n");
    // for (float f : input){
    //     printf("%:.1f ",f);
    // }
    // printf("\n");

    // printf("This %f should be == to %f\n",input[769],static_eval);

    return input;
}
