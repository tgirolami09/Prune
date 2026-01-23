#include "tunables.hpp"
tunables::tunables():
    iir_min_depth(3),
    iir_validity_depth(3),
    rfp_improving(120),
    rfp_nimproving(150),
    nmp_red_depth_div(4),
    nmp_red_base(3),
    se_validity_depth(3),
    se_min_depth(6),
    se_dext_margin(20),
    lmp_base(4),
    lmp_mul(4),
    mhp_mul(100),
    fp_base(300),
    fp_mul(150),
    fp_max_depth(5),
    lmr_history(512),
    mo_mul_malus(3),
    aw_base(20),
    see_born(0),
    pvalue(100),
    nvalue(300),
    bvalue(300),
    rvalue(500),
    qvalue(900),
    aw_mul(2.0),
    lmr_base(0.9),
    lmr_div(3.0),
    nodetm_base(2.0),
    nodetm_mul(1.6)
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
        &qvalue
    };
}

vector<float*> tunables::to_tune_float(){
    return {
        &aw_mul,
        &lmr_base,
        &lmr_div,
        &nodetm_base,
        &nodetm_mul
    };
}