//alternative for no libnuma and linux system
#define NUMA_NOLIB
#ifdef NUMA_NOLIB
#include "numa.hpp"
#include <pthread.h>
#include <cstdio>
#include <sched.h>
#include <string>
#include <filesystem>
#include <sys/sysinfo.h>
namespace fs = std::filesystem;

namespace prune_numa{
    vector<NNUE> nnues;
    bool init() {

        threadMapping();

        const int numNodes = nodeCount();
        printf("%d NUMA nodes\n", numNodes);
        nnues.reserve(numNodes);
        for(int i=0; i<numNodes; i++){
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
        
        std::string path = "/sys/devices/system/node/";
        unsigned int nodeCount = 0;
        for (const auto & entry : fs::directory_iterator(path))
            if(entry.is_directory()){
                string filename = entry.path().filename().string();
                if(filename.substr(0, 4) == "node"){
                    nodeCount++;
                }
            }
        return nodeCount;
    }

    vector<cpu_set_t> threadMapping() {
        static const auto s_mapping = [] {
            const auto maxNode = nodeCount()-1;

            vector<cpu_set_t> masks{};
            masks.reserve(maxNode + 1);
            int cpucount = get_nprocs();
            for (int node = 0; node <= maxNode; ++node) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                string filename = "/sys/devices/system/node/node"+to_string(node)+"/cpumap";
                FILE* nodefile = fopen(filename.c_str(), "r");
                unsigned int number;
                int cpu = 0;
                while(fscanf(nodefile, "%x", &number)){

                }
                for (unsigned int cpu = 0; cpu < cpucount; ++cpu) {
                    if(){
                        CPU_SET(cpu, &cpuset);
                    }
                }
                masks.push_back(cpuset);
            }

            return masks;
        }();

        return s_mapping;
    }

    int getNode(unsigned int numaId) {
        return numaId % nodeCount();
    }

    const NNUE& getnnue(uint32_t numaId){
        return nnues[numaId];
    }
}
#endif