#include "tunables.hpp"
tunables::tunables():
    iir_min_depth(396),
    iir_validity_depth(580),
    rfp_improving(122),
    rfp_nimproving(166),
    nmp_red_depth_div(271),
    nmp_red_base(398279),
    se_validity_depth(338),
    se_min_depth(792),
    se_dext_margin(16),
    lmp_base(66819),
    lmp_mul(531),
    mhp_mul(1691),
    fp_base(39083),
    fp_mul(181),
    fp_max_depth(658),
    lmr_history(75847),
    mo_mul_malus(197),
    aw_base(20),
    see_born(1),
    pvalue(157),
    nvalue(387),
    bvalue(303),
    rvalue(500),
    qvalue(850),
    lmr_base(131559),
    lmr_div(35835),
    mchp_mul(1852),
    mats_pawn(281),
    mats_knight(916),
    mats_bishop(1091),
    mats_rook(2202),
    mats_queen(4150),
    mats_offset(49275),
    fp_mh(87),
    aw_mul(1.58319),
    nodetm_base(2.30152),
    nodetm_mul(1.52559)
{}


vector<int*> tunables::to_tune_int(){
    return {
        //&iir_min_depth,
        //&iir_validity_depth,
        &rfp_improving,
        &rfp_nimproving,
        &nmp_red_depth_div,
        &nmp_red_base,
        //&se_validity_depth,
        //&se_min_depth,
        //&se_dext_margin,
        &lmp_base,
        &lmp_mul,
        &mhp_mul,
        &fp_base,
        &fp_mul,
        &fp_max_depth,
        &lmr_history,
        &mo_mul_malus,
        //&aw_base,
        //&see_born,
        &pvalue,
        &nvalue,
        &bvalue,
        &rvalue,
        &qvalue,
        &lmr_base,
        &lmr_div,
        &mchp_mul,
        &mats_pawn,
        &mats_knight,
        &mats_bishop,
        &mats_rook,
        &mats_queen,
        &mats_offset,
        &fp_mh
    };
}

vector<float*> tunables::to_tune_float(){
    return {
        //&aw_mul,
        //&nodetm_base,
        //&nodetm_mul
    };
}