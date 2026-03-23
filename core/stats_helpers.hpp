#ifndef STATS_HELPER
#define STATS_HELPER
#include <cmath>
#include <cstdint>
#include <string>
#include <cassert>
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
        printf("%s = %.2f ± %.2f\n", start.c_str(), mean, sqrt(sumSquare/number-mean*mean));
        printf("  %ld <= x <= %ld", obsmin, obsmax);
        int cum[maxi-mini+2];
        cum[0] = 0;
        for(int i=0; i<maxi-mini+1; i++){
            cum[i+1] = cum[i]+hist[i];
        }
        printf("  percentiles:\n");
        for(double percent:{0.1, 1., 5., 25., 50., 75., 95., 99., 99.9}){
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
            printf("    %.2f : %d\n", percent, left+mini);
        }
    }
};
#endif