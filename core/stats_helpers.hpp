#ifndef STATS_HELPER
#define STATS_HELPER
#include <cmath>
#include <cstdint>
#include <string>
#include <cassert>
#include <cstring>
using namespace std;
template<typename T, int maxi=-1, int mini=0>
class StatVar{
public:
    T sum;
    T sumSquare;
    uint64_t number;
    T obsmin;
    T obsmax;
    int hist[maxi-mini+1];
    StatVar(){reset();}
    void reset(){
        sum = 0;
        sumSquare = 0;
        number = 0;
        obsmin = maxi;
        obsmax = mini;
        memset(hist, 0, sizeof(hist));
    }
    void update(const T& v){
        if constexpr(maxi >= mini){
            assert(v <= maxi && v >= mini);
            hist[v-mini]++;
        }
        if(v > obsmax)obsmax = v;
        if(v < obsmin)obsmin = v;
        sum += v;
        sumSquare += v*v;
        number++;
    }
    void print(string start) const{
        double mean = ((double)sum)/number;
        printf("%s = %.2f ± %.2f (n=%ld)\n", start.c_str(), mean, sqrt(sumSquare/number-mean*mean), number);
        printf("  %ld <= x <= %ld\n", obsmin, obsmax);
        int cum[maxi-mini+2];
        cum[0] = 0;
        for(int i=0; i<maxi-mini+1; i++){
            cum[i+1] = cum[i]+hist[i];
        }
        printf("  percentiles:\n");
        for(double percent:{0.1, 1., 5., 10., 20., 25., 30., 40., 50., 60., 70., 75., 80., 90., 95., 99., 99.9}){
            int search = cum[maxi-mini+1]*percent/100;
            int left = 0;
            int right = maxi-mini+2;
            while(left < right){
                int mid = (left+right)/2;
                if(cum[mid] < search){
                    left = mid+1;
                }else if(cum[mid] > search){
                    right = mid;
                }else{
                    left = mid;
                    break;
                }
            }
            printf("    %.2f : %.2f (%.2f %.2f)\n", percent,
                ((double)(cum[left]-search)*(left-1+mini)+(search-cum[left-1])*(left+mini))/(cum[left]-cum[left-1]),
                (double)cum[left]*100/cum[maxi-mini+1], (double)cum[left-1]*100/cum[maxi-mini+1]);
        }
    }
};
#endif