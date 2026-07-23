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
        iir_min_depth(330),
        iir_validity_depth(384),
        rfp_improving(93),
        rfp_nimproving(155),
        nmp_red_depth_div(289),
        nmp_red_base(318475),
        se_validity_depth(325),
        se_min_depth(811),
        se_dext_margin(18),
        se_cutnode_negext(126),
        se_negext(130),
        se_dmul(1024),
        lmp_base(4),
        lmp_mul(4),
        mhp_mul(1863),
        mchp_mul(1447),
        mhcp_max_depth(496),
        fp_base(34389),
        fp_mul(133),
        fp_max_depth(729),
        fp_hmul(8118),
        lmr_history_tact(1024),
        lmr_history_quiet(640),
        lmr_base(723),
        lmr_div(274),
        lmr_min_depth(277),
        capthist_mul_malus(236),
        capthist_mul_bonus(509),
        aw_base(19),
        pvalue(106),
        nvalue(263),
        bvalue(288),
        rvalue(567),
        qvalue(893),
        mats_pawn(20),
        mats_knight(1112),
        mats_bishop(1009),
        mats_rook(1895),
        mats_queen(3018),
        mats_offset(39862),
        see_mul_quiet(64),
        see_mul_tact(80),
        se_pv_offset(100),
        mainHist(1022, 1049, 1100, 1032, 510, 272),
        prevHist(996, 924, 1131, 1042, 481, 267),
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
        lmr_history_tact,
        lmr_history_quiet,
        lmr_base,
        lmr_div,
        lmr_min_depth,
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