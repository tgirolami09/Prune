#include "tunables.hpp"
tunables::tunables():
    iir_min_depth(3),
    iir_validity_depth(4),
    rfp_improving(103),
    rfp_nimproving(147),
    nmp_red_depth_div(298),
    nmp_red_base(2860),
    se_validity_depth(4),
    se_min_depth(6),
    se_dext_margin(15),
    lmp_base(4),
    lmp_mul(4),
    mhp_mul(113),
    fp_base(291),
    fp_mul(156),
    fp_max_depth(5),
    lmr_history(584),
    mo_mul_malus(2),
    aw_base(19),
    see_born(1),
    pvalue(87),
    nvalue(286),
    bvalue(230),
    rvalue(476),
    qvalue(995),
    lmr_base(731),
    lmr_div(317),
    mchp_mul(90),
    aw_mul(2.15959),
    nodetm_base(1.90289),
    nodetm_mul(1.16744)
{}

vector<int*> tunables::to_tune_int(){
    return {
        &iir_min_depth,
        &iir_validity_depth,
        &rfp_improving,
        &rfp_nimproving,
        &nmp_red_depth_div,
        &nmp_red_base,
        &se_validity_depth,
        &se_min_depth,
        &se_dext_margin,
        &lmp_base,
        &lmp_mul,
        &mhp_mul,
        &fp_base,
        &fp_mul,
        &fp_max_depth,
        &lmr_history,
        &mo_mul_malus,
        &aw_base,
        &see_born,
        &pvalue,
        &nvalue,
        &bvalue,
        &rvalue,
        &qvalue,
        &lmr_base,
        &lmr_div,
        &mchp_mul
    };
}

vector<float*> tunables::to_tune_float(){
    return {
        &aw_mul,
        &nodetm_base,
        &nodetm_mul
    };
}