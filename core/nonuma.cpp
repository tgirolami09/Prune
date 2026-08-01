#if !defined(NUMA) && !defined(NUMA_NOLIB)
#pragma message("not using numa")
#include "numa.hpp"

namespace prune_numa{
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