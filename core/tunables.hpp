#ifndef TUNABLE_HPP
#define TUNABLE_HPP
#include <vector>
using namespace std;

class tunables{
public:
    tunables();
    int iir_min_depth,
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
        mchp_mul;
    float
        aw_mul,
        nodetm_base,
        nodetm_mul;
    vector<int*> to_tune_int();
    vector<float*> to_tune_float();
};
#endif