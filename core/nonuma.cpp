#if !defined(NUMA) && !defined(NUMA_NOLIB)
#include "numa.hpp"

namespace prune_numa{
    vector<NNUE> nnues;
    bool init() {
        return true;
    }

    void bindThread(_unused uint32_t numaId) {
    }

    int nodeCount() {
        return 1;
    }

    int getNode(_unused unsigned int numaId) {
        return 0;
    }
    const NNUE& getnnue(_unused uint32_t numaId){
        return globnnue;
    }
}
#endif