#include "wdlModel.hpp"
#include <cmath>
#include "TablebaseProbe.hpp"

namespace WDLmodel{
    bool enabled = true;
    pair<double, double> wdlParams(int material){
        return {
            ((as[0] * material / 58 + as[1]) * material / 58 + as[2]) * material / 58 + as[3],
            ((bs[0] * material / 58 + bs[1]) * material / 58 + bs[2]) * material / 58 + bs[3]       
        };

    }

    pair<int, int> wdl(int score, int material){
        const auto [a, b] = wdlParams(material);
        const double x = score;
        return{
            std::round(1000.0 / (1.0 + std::exp((a - x) / b))),
            std::round(1000.0 / (1.0 + std::exp((a + x) / b)))
        };
    }

    int normalize(int score, int material){
        if(score < -TB_WIN_SCORE+100 || score > TB_WIN_SCORE-100)return score;
        return score*100/wdlParams(material).first;
}
}