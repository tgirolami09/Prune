#include "helper_numa.hpp"

numa::numa(){
    if(numa_available() < 0){
        printf("info string numa is not available\n");
        isavailable = false;
    }else{
        isavailable = true;
        numnode = numa_max_node()+1;
        cpumasks.reserve(numnode);
        for(int node=0; node<numnode; node++){
            auto* cpumask = numa_allocate_cpumask();
            numa_node_to_cpus(node, cpumask);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for(int cpu=0; cpu<(int)cpumask->size; cpu++){
                if(numa_bitmask_isbitset(cpumask, cpu)){
                    CPU_SET(cpu, &cpuset);
                }
            }
            numa_free_cpumask(cpumask);
            cpumasks.push_back(cpuset);
        }
        printf("info string numa is available : %d nodes for a total of %d cpus\n", numnode, nbcpu);
    }
}

void numa::pinThread(int idThread) const{
    if(isavailable)
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpumasks[getNode(idThread)]);
}

int numa::getNode(int idThread) const{
    if(!isavailable)return 0;
    return idThread*numnode/nbcpu;
}