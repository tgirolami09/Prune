//numa code from https://github.com/Ciekce/Stormphrax
#ifndef NUMA_HPP
#define NUMA_HPP

#include <cstdint>
#include <vector>
#include "NNUE.hpp"

using namespace std;

namespace prune_numa{
    extern vector<NNUE> nnues;
    bool init();
    void bindThread(uint32_t numaId);
    const NNUE& getnnue(uint32_t numaId);
    int nodeCount();
    vector<cpu_set_t> threadMapping();

    int32_t getNode(uint32_t numaId);
};
#endif