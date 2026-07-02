#include "tunables.hpp"

vector<TunableInt*> tunables::to_tune_int(){
    vector<TunableInt*> tuns{
        &iir_min_depth,
        &iir_validity_depth,
        &rfp_improving,
        &rfp_nimproving,
        &nmp_red_depth_div,
        &nmp_red_base,
        &se_validity_depth,
        &se_min_depth,
        &se_dext_margin,
        //&se_dmul,
        //&lmp_base,
        //&lmp_mul,
        &mhp_mul,
        &mchp_mul,
        &fp_base,
        &fp_mul,
        &fp_max_depth,
        &fp_hmul,
        &lmr_history,
        &lmr_base,
        &lmr_div,
        &capthist_mul_malus,
        &capthist_mul_bonus,
        &aw_base,
        &pvalue,
        &nvalue,
        &bvalue,
        &rvalue,
        &qvalue,
        &mats_pawn,
        &mats_knight,
        &mats_bishop,
        &mats_rook,
        &mats_queen,
        &mats_offset,
        &see_mul_quiet,
        &see_mul_tact,
        &se_pv_offset,
    };
    for(TunableHist* tunshist:{&mainHist, &prevHist}){
        tuns.push_back(&tunshist->order);
        tuns.push_back(&tunshist->lmr);
        tuns.push_back(&tunshist->mhp);
        tuns.push_back(&tunshist->fp);
        tuns.push_back(&tunshist->bonus);
        tuns.push_back(&tunshist->malus);
    }
    return tuns;
}

vector<TunableFloat*> tunables::to_tune_float(){
    return {
        //&aw_mul,
        //&nodetm_base,
        //&nodetm_mul
    };
}