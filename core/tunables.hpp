#ifndef TUNABLE_HPP
#define TUNABLE_HPP
#include "Const.hpp"
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
        return value -= a;
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

struct TunableHist{
    TunableInt
        order,
        lmr,
        mhp,
        fp;
    TunableInt
        bonus,
        malus;
    constexpr TunableHist(int _order, int _lmr, int _mhp, int _fp, int _bonus, int _malus):
        order(_order), lmr(_lmr), mhp(_mhp), fp(_fp), bonus(_bonus), malus(_malus){}
    enum{ORDER, LMR, MHP, FP};
    template<int id>
    const TunableInt& getParam() const{
        if constexpr(id == ORDER)return order;
        else if constexpr(id == LMR)return lmr;
        else if constexpr(id == MHP)return mhp;
        else if constexpr(id == FP)return fp;
    }
};

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
        iir_min_depth(335),
        iir_validity_depth(430),
        rfp_improving(93),
        rfp_nimproving(150),
        nmp_red_depth_div(265),
        nmp_red_base(305817),
        se_validity_depth(350),
        se_min_depth(759),
        se_dext_margin(18),
        se_triplext_margin(120),
        se_cutnode_negext(128),
        se_negext(128),
        se_dmul(1024),
        lmp_base(4),
        lmp_mul(4),
        mhp_mul(1782),
        mchp_mul(1539),
        mhcp_max_depth(512),
        fp_base(37530),
        fp_mul(135),
        fp_max_depth(693),
        fp_hmul(8210),
        lmr_history(676),
        lmr_base(683),
        lmr_div(259),
        lmr_min_depth(256),
        lmr_min_reduction_pv(0, 0, 1024),
        lmr_min_reduction_npv(0, 0, 1024),
        capthist_mul_malus(256),
        capthist_mul_bonus(500),
        aw_base(20),
        pvalue(107),
        nvalue(259),
        bvalue(278),
        rvalue(545),
        qvalue(904),
        mats_pawn(20),
        mats_knight(1084),
        mats_bishop(987),
        mats_rook(1924),
        mats_queen(3406),
        mats_offset(41705),
        see_mul_quiet(61),
        see_mul_tact(84),
        se_pv_offset(94),
        mainHist(1042, 1044, 1091, 1018, 539, 271),
        prevHist(994, 949, 1109, 1072, 492, 283),
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
        se_triplext_margin,
        se_cutnode_negext,
        se_negext,
        se_dmul,
        lmp_base,
        lmp_mul,
        mhp_mul,
        mchp_mul,
        mhcp_max_depth,
        fp_base,
        fp_mul,
        fp_max_depth,
        fp_hmul,
        lmr_history,
        lmr_base,
        lmr_div,
        lmr_min_depth,
        lmr_min_reduction_pv,
        lmr_min_reduction_npv,
        capthist_mul_malus,
        capthist_mul_bonus,
        aw_base,
        pvalue,
        nvalue,
        bvalue,
        rvalue,
        qvalue,
        mats_pawn,
        mats_knight,
        mats_bishop,
        mats_rook,
        mats_queen,
        mats_offset,
        see_mul_quiet,
        see_mul_tact,
        se_pv_offset;
    TunableHist
        mainHist,
        prevHist;
    TunableFloat
        aw_mul,
        nodetm_base,
        nodetm_mul;
    vector<TunableInt*> to_tune_int();
    vector<TunableFloat*> to_tune_float();
};
#endif