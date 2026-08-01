//alternative for no libnuma and linux system
#ifdef NUMA_NOLIB
#pragma message("using numa")
#include "numa.hpp"
#include <pthread.h>
#include <cstdio>
#include <sched.h>
#include <string>
#include <filesystem>
#include <sys/sysinfo.h>
namespace fs = std::filesystem;

namespace prune_numa{
    unsigned int nbNodes=0;
    vector<NNUE> nnues;
    bool __attribute__((constructor(100))) init() {

        std::string path = "/sys/devices/system/node/";
        unsigned int _nbNodes = 0;
        for (const auto & entry : fs::directory_iterator(path))
            if(entry.is_directory()){
                string filename = entry.path().filename().string();
                if(filename.substr(0, 4) == "node"){
                    _nbNodes++;
                }
            }
        nbNodes = _nbNodes;
        nnues.resize(nbNodes);

        const int numNodes = nodeCount();
        printf("%d NUMA nodes\n", numNodes);
        printf("nnues.size() = %lu/%d (line %d)\n", nnues.size(), nbNodes, __LINE__);
        for(int i=0; i<numNodes; i++){
            memcpy(&nnues[i], baseModel, sizeof(NNUE));
        }

        threadMapping();

        return true;
    }

    void bindThread(uint32_t numaId) {
        printf("nnues.size() = %lu/%d (line %d)\n", nnues.size(), nbNodes, __LINE__);
        const auto node = getNode(numaId);
        const auto handle = pthread_self();
        const auto cpuSet = threadMapping()[node];
        pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuSet);
    }

    int nodeCount() {
        printf("nnues.size() = %lu/%d (line %d)\n", nnues.size(), nbNodes, __LINE__);
        return nbNodes;
    }

    vector<cpu_set_t> threadMapping() {
        printf("nnues.size() = %lu/%d (line %d)\n", nnues.size(), nbNodes, __LINE__);
        static const auto s_mapping = [] {
            const auto maxNode = nodeCount()-1;

            vector<cpu_set_t> masks{};
            masks.reserve(maxNode + 1);
            const unsigned int cpucount = get_nprocs();
            for (int node = 0; node <= maxNode; ++node) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                string filename = "/sys/devices/system/node/node"+to_string(node)+"/cpumap";
                FILE* nodefile = fopen(filename.c_str(), "r");
                unsigned int number;
                vector<unsigned int> curmasks;
                curmasks.reserve(cpucount/32);
                while(fscanf(nodefile, "%x", &number) != EOF){
                    curmasks.push_back(number);
                    if(fgetc(nodefile) == EOF)break;
                }
                const int nbMasks = curmasks.size();
                for (unsigned int cpu = 0; cpu < cpucount; ++cpu) {
                    if(curmasks[nbMasks-1-cpu/32] & (1U << (cpu%32))){
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
        printf("nnues.size() = %lu/%d (line %d)\n", nnues.size(), nbNodes, __LINE__);
        return nnues[getNode(numaId)];
    }
}
#endif