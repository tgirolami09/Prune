// numa code from https://github.com/Ciekce/Stormphrax
#ifdef NUMA
#include "numa.hpp"
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <cstdio>

namespace prune_numa {
vector<NNUE> nnues;
bool init() {
    if (numa_available() < 0) {
        return false;
    }

    threadMapping();

    const int numNodes = nodeCount();
    printf("info string %d NUMA nodes\n", numNodes);
    nnues.resize(numNodes);
    for (int i = 0; i < numNodes; i++) {
        memcpy(&nnues[i], baseModel, sizeof(NNUE));
    }

    return true;
}

void bindThread(uint32_t numaId) {
    const auto node = getNode(numaId);
    const auto handle = pthread_self();
    const auto cpuSet = threadMapping()[node];
    pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuSet);
}

int nodeCount() {
    return static_cast<int>(threadMapping().size());
}

std::span<const cpu_set_t> threadMapping() {
    static const auto s_mapping = [] {
        const auto maxNode = numa_max_node();

        vector<cpu_set_t> masks{};
        masks.reserve(maxNode + 1);

        for (int node = 0; node <= maxNode; ++node) {
            auto* cpumask = numa_allocate_cpumask();

            if (numa_node_to_cpus(node, cpumask) != 0) {
                fprintf(stderr, "failed to get CPU mask for NUMA node %d\n", node);
                exit(1);
            }

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);

            for (unsigned int cpu = 0; cpu < cpumask->size; ++cpu) {
                if (numa_bitmask_isbitset(cpumask, cpu)) {
                    CPU_SET(cpu, &cpuset);
                }
            }

            numa_free_cpumask(cpumask);
            masks.push_back(cpuset);
        }

        return masks;
    }();

    return s_mapping;
}

int getNode(unsigned int numaId) {
    return numaId % nodeCount();
}

const NNUE& getnnue(uint32_t numaId) {
    return nnues[getNode(numaId)];
}
}  // namespace prune_numa
#endif
