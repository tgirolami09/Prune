#include <cstdio>
#include <random>
#include "util_magic.hpp"
using namespace std;

std::random_device rd;
std::mt19937_64 e2(rd());
std::uniform_int_distribution<big> dist(0, MAX_BIG);
/*
0   1   2   3   4   5   6   7
8                           15
16                          23
24                          31
32                          39
40                          47
48                          55
56  57  58  59  60  61  62  63
*/
big generate(){
    return dist(e2);
}

class node{
public:
    int right, left;
    int last;
    node(){
        right = -1;
        left = -1;
        last = -1;
    }
};
class creux{
public:
    vector<node> Ns;
    vector<int> nbNodesPerDepth;
    vector<big> mapping;
    int valid;
    int depthMax;
    creux(){}
    creux(int depth){
        Ns = {node()};
        nbNodesPerDepth = vector<int>(depth);
        valid = 0;
        depthMax = depth;
    }
    void push(big x, big res){
        mapping.push_back(res);
        born curL=0, curR=((born)1)<<depthMax;
        int T=0;
        int depth=0;
        int id=mapping.size()-1;
        while(1){
            if(Ns[T].last != -1 && mapping[Ns[T].last] != mapping[id])
                valid = max(valid, depth+1);
            Ns[T].last = id;
            if(curL+1 == curR)return;
            born mid = (curL+curR)/2;
            int nT;
            if(x >= mid){
                curL = mid;
                nT = Ns[T].right;
            }else{
                curR = mid;
                nT = Ns[T].left;
            }if(nT == -1){
                nT = Ns.size();
                nbNodesPerDepth[depth]++;
                Ns.push_back(node());
                if(x >= mid)Ns[T].right = nT;
                else Ns[T].left = nT;
            }
            depth++;
            T = nT;
        }
    }
};

info test_magic(big magic, int square, bool is_rook){
    int nbBits = __builtin_popcountll(get_mask(is_rook, MAX_BIG, square));
    creux tree[63];
    for(int i=0; i<63; i++)tree[i] = creux(64-i);
    for(big id=0; id<(1<<nbBits); id++){
        big mask = get_mask(is_rook, id, square);
        big res = mask*magic;
        big res_mask = get_usefull(is_rook, mask, square);
        for(int i=0; i<63; i++){
            tree[i].push(res&(MAX_BIG >> i), res_mask);
        }
    }
    info res = {64, 0, magic};
    for(int i=0; i<63; i++){
        //printf("%d ", tree[i].valid);
        if(tree[i].valid <= 64-i && tree[i].valid < res.minimum){
            res.minimum = tree[i].valid;
            res.decR = i;
        }
    }
    //printf("\n");
    return res;
}

int main(int argc, char* argv[]){
    init_lines();
    vector<vector<info>> best;
    if(argc > 2)
        best = load_info(argv[2]);
    int ok=0;
    if(best.size() == 0){
        best = vector<vector<info>>(2, vector<info>(64, {20, 0, 1}));
    }else{
        print_table(best);
        ok=best[0][0].minimum != 20?128:64;
    }
    int totLength=0;
    for(vector<info> board:best)
        for(info i:board)
            totLength += 1 << i.minimum;
    while(1){
        big magic = generate();
        bool change=false;
        for(int is_rook=0; is_rook < 2; is_rook++){
            bool is_rook_printed=false;
            for(int square=0; square<64; square++){
                info r=test_magic(magic, square, is_rook);
                if(r.minimum < best[is_rook][square].minimum){
                    totLength += (1<<r.minimum)-(1<<best[is_rook][square].minimum);
                    if(best[is_rook][square].minimum == 20)
                        ok++;
                    string display_rook = is_rook?"  rook":"bishop";
                    if(change){
                        if(is_rook_printed)
                            printf("%*ccase = %2d needed bits = %2d decRight = %2d\n", 32, ' ', square, r.minimum, r.decR);
                        else
                            printf("%*c%s case = %2d needed bits = %2d decRight = %2d\n", 25, ' ', display_rook.c_str(), square, r.minimum, r.decR);
                    }else
                        printf("magic = %16lx %s case = %2d needed bits = %2d decRight = %2d\n", magic, display_rook.c_str(), square, r.minimum, r.decR);
                    best[is_rook][square] = r;
                    change=true;
                    is_rook_printed = true;
                }
            }
        }
        if(change && ok==128){
            int place_lost=dump_entire(best, argv[1]);
            printf("%d/%d->%.2f\n", place_lost, totLength, place_lost*100.0/totLength);
            for(int is_rook=0; is_rook < 2; is_rook++){
                int maxi=0;
                int mini=20;
                for(info i:best[is_rook]){
                    maxi = max(maxi, i.minimum);
                    mini = min(mini, i.minimum);
                }
                printf("\tmin: %2d max: %2d => ", mini, maxi);
                vector<int> occ(maxi-mini+1, 0);
                for(info i:best[is_rook])
                    occ[i.minimum-mini]++;
                for(int t=0; t<maxi-mini+1; t++)
                    printf("%2d ", occ[t]);
                printf("\n");
            }
        }
    }
}
