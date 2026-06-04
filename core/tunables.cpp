#include "tunables.hpp"

vector<TunableInt*> tunables::to_tune_int(){
    return {
        //&iir_min_depth,
        //&iir_validity_depth,
        &rfp_improving,
        &rfp_nimproving,
        &nmp_red_depth_div,
        &nmp_red_base,
        //&se_validity_depth,
        //&se_min_depth,
        &se_dext_margin,
        //&lmp_base,
        //&lmp_mul,
        &mhp_mul,
        &fp_base,
        &fp_mul,
        //&fp_max_depth,
        &lmr_history,
        &mo_mul_malus,
        &aw_base,
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
        &see_mul_quiet,
        &see_mul_tact
    };
}

vector<TunableFloat*> tunables::to_tune_float(){
    return {
        //&aw_mul,
        //&nodetm_base,
        //&nodetm_mul
    };
}