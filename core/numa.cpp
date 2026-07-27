#include "numa.hpp"
#include <pthread.h>
#include <cstdio>

namespace prune_numa{
    bool init() {
        if (numa_available() < 0) {
            return false;
        }

        threadMapping();

        printf("%d NUMA nodes\n", nodeCount());

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

    vector<cpu_set_t> threadMapping() {
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
}