#include <numa.h>
#include <cstdio>
#include <pthread.h>
#include <vector>
#include <thread>
using namespace std;

class numa{
public:
    int numnode;
    bool isavailable;
    vector<cpu_set_t> cpumasks;
    const int nbcpu = thread::hardware_concurrency();;
    numa();
    void pinThread(int idThread) const;
    int getNode(int idThread) const;
};