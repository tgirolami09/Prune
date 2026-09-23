#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "Evaluator.hpp"
#include "GameState.hpp"
#include "NNUE.hpp"
using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

int main(_unused int argc, char** argv) {
    ifstream file(argv[1]);
    IncrementalEvaluator eval;
    GameState state;
    string line;
    big sum_eval = 0;
    int count = 0;
    float loss = 0.0;
    while (getline(file, line)) {
        string fen;
        stringstream ss(line);
        string token;
        for (int i = 0; i < 6; i++) {
            ss >> token;
            fen += token + ' ';
        }
        string label;
        float result;
        ss >> label;
        if (label == "[0.0]")
            result = 0.0;
        else if (label == "[0.5]")
            result = 0.5;
        else if (label == "[1.0]")
            result = 1.0;
        else {
            printf("%s\n", label.c_str());
            assert(false);
        }
        state.fromFen(fen);
        eval.init(state, globnnue);
        int staticeval = eval.getRaw(state.friendlyColor(), globnnue);
        if (state.friendlyColor() == BLACK)
            staticeval = -staticeval;
        sum_eval += abs(staticeval);
        loss += pow(sigmoid(staticeval / 400.0) - result, 2);
        count++;
    }
    printf("%lf\n", ((double)sum_eval) / count);
    printf("%lf\n", loss / count);
}
