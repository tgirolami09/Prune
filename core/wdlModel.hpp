#ifndef WDL_MODEL
#define WDL_MODEL

#include <utility>
using namespace std;


namespace WDLmodel{
    extern bool enabled;
    constexpr double as[] = {-78.45598559, 311.88123874, -372.23086593, 450.44860362};
    constexpr double bs[] = {27.82745092, -14.91027822, 25.13548380, 65.34779274};
    pair<double, double> wdlParams(int material); 
    pair<int, int> wdl(int score, int material);
    int normalize(int score, int material);
};

#endif