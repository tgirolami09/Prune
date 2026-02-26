#include "tunables.hpp"
tunables::tunables():
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
    mo_mul_malus(32),
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
    aw_mul(1.9495),
    nodetm_base(2.13688),
    nodetm_mul(1.37487)
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