#include "tunables.hpp"
tunables::tunables():
    iir_min_depth(3),
    iir_validity_depth(3),
    rfp_improving(115),
    rfp_nimproving(159),
    nmp_red_depth_div(272),
    nmp_red_base(3180),
    se_validity_depth(3),
    se_min_depth(6),
    se_dext_margin(20),
    lmp_base(4),
    lmp_mul(4),
    mhp_mul(96),
    fp_base(320),
    fp_mul(151),
    fp_max_depth(5),
    lmr_history(513),
    mo_mul_malus(3),
    aw_base(19),
    see_born(1),
    pvalue(106),
    nvalue(287),
    bvalue(289),
    rvalue(478),
    qvalue(1011),
    lmr_base(879),
    lmr_div(312),
    aw_mul(1.99421),
    nodetm_base(2.08567),
    nodetm_mul(1.56134)
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
        &lmr_div
    };
}

vector<float*> tunables::to_tune_float(){
    return {
        &aw_mul,
        &nodetm_base,
        &nodetm_mul
    };
}