#ifndef TUNABLE_HPP
#define TUNABLE_HPP
#include <cstdlib>
#include <vector>
#include <cassert>
using namespace std;
#define TUNE
#ifdef TUNE
struct TunableInt{
    int value;
    int minimum, maximum;
    float c_end;
    float r_end;
    explicit TunableInt(int v):value(v),minimum(v/2), maximum(v*2){
        c_end = abs(maximum-minimum)/20.0;
        r_end = 0.002 / (min(0.5f, c_end) / 0.5);
    }
    explicit TunableInt(int v, int mi, int ma):value(v),minimum(mi), maximum(ma){
        c_end = abs(maximum-minimum)/20.0;
        r_end = 0.002 / (min(0.5f, c_end) / 0.5);
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
    TunableFloat(float v):value(v),minimum(v/2), maximum(v*2){
        assert(v > 0);
        c_end = abs(maximum-minimum)/20;
        r_end = 0.002;
    }
    TunableFloat(float v, float mi, float ma, float ce, float re):value(v), minimum(mi), maximum(ma), c_end(ce), r_end(re){}
    operator float(){
        return value;
    }
    void operator=(float a){
        value = a;
    }
};
#else
using TunableFloat=float;
using TunableInt=int;
#endif
class tunables{
public:
    tunables();
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