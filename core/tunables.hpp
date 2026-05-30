#ifndef TUNABLE_HPP
#define TUNABLE_HPP
#include <cstdlib>
#include <vector>
#include <cassert>
using namespace std;

template<typename T>
constexpr T cabs(T x){
    return x < 0?-x:x;
}
template<typename T>
constexpr T cmin(T a, T b){
    return a < b?a:b;
}

struct TunableInt{
    int value;
    int minimum, maximum;
    float c_end;
    float r_end;
    explicit constexpr TunableInt(int v):value(v),minimum(v/2), maximum(v*2), c_end(cabs(maximum-minimum)/20.0), r_end(0.002 / (cmin(0.5f, c_end) / 0.5)){
    }
    explicit constexpr TunableInt(int v, int mi, int ma):value(v),minimum(mi), maximum(ma), c_end(cabs(maximum-minimum)/20.0), r_end(0.002 / (cmin(0.5f, c_end) / 0.5)){
    }
    operator int() const{
        return value;
    }
    operator float() const{
        return value;
    }
    int operator-() const{
        return -value;
    }
    void operator=(int a){
        value = a;
    }
    template<typename T>
    T operator*(T a) const{
        return a*value;
    }
    template<typename T>
    int operator*=(T a){
        return value *= a;
    }
    template<typename T>
    T operator+(T a) const{
        return a+value;
    }
    template<typename T>
    int operator+=(T a){
        return value += a;
    }
    template<typename T>
    T operator-(T a) const{
        return a+value;
    }
    template<typename T>
    int operator-=(T a){
        return value += a;
    }
    bool operator<(int a) const{
        return value < a;
    }
    bool operator<=(int a) const{
        return value <= a;
    }
    bool operator>(int a) const{
        return value > a;
    }
    bool operator>=(int a) const{
        return value >= a;
    }
    
};

template<typename T>
T operator*(T a, TunableInt b){
    return b*a;
}
template<typename T>
T operator+(T a, TunableInt b){
    return b+a;
}
template<typename T>
T operator-(T a, TunableInt b){
    return a+-b;
}
template<typename T>
bool operator<(T a, TunableInt b){
    return b>a;
}
template<typename T>
bool operator<=(T a, TunableInt b){
    return b>=a;
}
template<typename T>
bool operator>(T a, TunableInt b){
    return b<a;
}
template<typename T>
bool operator>=(T a, TunableInt b){
    return b<=a;
}


struct TunableFloat{
    float value;
    float minimum, maximum;
    float c_end, r_end;
    constexpr TunableFloat(float v):value(v),minimum(v/2), maximum(v*2), c_end(cabs(maximum-minimum)/20), r_end(0.002){
        assert(v > 0);
    }
    constexpr TunableFloat(float v, float mi, float ma, float ce, float re):value(v), minimum(mi), maximum(ma), c_end(ce), r_end(re){}
    operator float() const{
        return value;
    }
    void operator=(float a){
        value = a;
    }
};

class tunables{
public:
    constexpr tunables():
        iir_min_depth(3),
        iir_validity_depth(4),
        rfp_improving(94),
        rfp_nimproving(163),
        nmp_red_depth_div(263),
        nmp_red_base(2793),
        se_validity_depth(3),
        se_min_depth(6),
        se_dext_margin(17),
        lmp_base(4),
        lmp_mul(4),
        mhp_mul(1618),
        fp_base(329),
        fp_mul(148),
        fp_max_depth(5),
        lmr_history(531),
        mo_mul_malus(256),
        aw_base(20),
        see_born(1),
        pvalue(104),
        nvalue(280),
        bvalue(271),
        rvalue(487),
        qvalue(990),
        lmr_base(851),
        lmr_div(335),
        mchp_mul(1600),
        mats_pawn(0, 0, 400),
        mats_knight(1024),
        mats_bishop(1024),
        mats_rook(2048),
        mats_queen(4096),
        mats_offset(36864),
        aw_mul(1.9495),
        nodetm_base(2.13688),
        nodetm_mul(1.37487){}

    TunableInt
        iir_min_depth,
        iir_validity_depth,
        rfp_improving,
        rfp_nimproving,
        nmp_red_depth_div,
        nmp_red_base,
        se_validity_depth,
        se_min_depth,
        se_dext_margin,
        lmp_base,
        lmp_mul,
        mhp_mul,
        fp_base,
        fp_mul,
        fp_max_depth,
        lmr_history,
        mo_mul_malus,
        aw_base,
        see_born,
        pvalue,
        nvalue,
        bvalue,
        rvalue,
        qvalue,
        lmr_base,
        lmr_div,
        mchp_mul,
        mats_pawn,
        mats_knight,
        mats_bishop,
        mats_rook,
        mats_queen,
        mats_offset;
    TunableFloat
        aw_mul,
        nodetm_base,
        nodetm_mul;
    vector<TunableInt*> to_tune_int();
    vector<TunableFloat*> to_tune_float();
};
#endif