//numa code from https://github.com/Ciekce/Stormphrax
#ifndef NUMA_HPP
#define NUMA_HPP

#include <numa.h>
#include <cassert>
#include <cstdint>
#include <sched.h>
#include <vector>
#include "NNUE.hpp"

using namespace std;

namespace prune_numa{
    extern vector<NNUE> nnues;
    bool init();
    void bindThread(uint32_t numaId);
    int nodeCount();
    vector<cpu_set_t> threadMapping();

    int32_t getNode(uint32_t numaId);
};
#endif