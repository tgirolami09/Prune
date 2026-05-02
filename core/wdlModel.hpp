#ifndef WDL_MODEL
#define WDL_MODEL

#include <utility>
using namespace std;


namespace WDLmodel{
    extern bool enabled;
    static constexpr double as[] = {-167.91466546, 512.73835575, -470.06377011, 391.48951340};
    static constexpr double bs[] = {-9.46772054, 89.88269475, -48.64658159, 56.69635829};
    pair<double, double> wdlParams(int material); 
    pair<int, int> wdl(int score, int material);
    int normalize(int score, int material);
};

#endif