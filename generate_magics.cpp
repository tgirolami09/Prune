#include <cstdio>
#include <random>
#include <cstdint>
#include <cassert>
#include <fstream>
using namespace std;
#define big uint64_t
#define born __uint128_t
#define forsquare for(int square=0; square<64; square++)
const big MAX_BIG = ~((big)0);
void print_mask(big mask){
    int col=0;
    for(int row=0; row<=64; row+=8){
        for(; col<row; col++){
            printf("%d", (int)mask&1);
            mask >>= 1;
        }
        printf("\n");
    }
}
std::random_device rd;
std::mt19937_64 e2(rd());
std::uniform_int_distribution<big> dist(0, MAX_BIG);
/*
0   1   2   3   4   5   6   7
8
16
24
32
40
48
56
*/
big clipped_row[8];
big clipped_col[8];
void init(){
    big row = MAX_BIG >> (8*7+2) << 1;
    big col = 0x0001010101010100ULL;
    for(int i=0; i<8; i++){
        clipped_row[i] = row;
        //print_mask(row);
        //print_mask(col);
        clipped_col[i] = col;
        row <<= 8;
        col <<= 1;
    }
}
big generate(){
    return dist(e2);
}
big generate_mask(big id, big square){
    big mask = (clipped_row[square>>3]|clipped_col[square&7])&(~(1ULL<<square));
    //print_mask(mask);
    big square_mask = 1;
    for(int i=0; i<64; i++){
        if(mask&square_mask){
            if((id&1) == 0)
                mask ^= square_mask;
            id >>= 1;
        }
        square_mask <<= 1;
    }
    return mask;
}
big usefull(big mask, int square){
    big cur_mask = 1ULL << square;
    int col = square&7;
    int row = square >> 3;
    big mask_square = cur_mask;
    for(int i=1; i < col; i++){
        cur_mask |= mask_square >> i;
        if(mask&cur_mask)break;
    }
    big bef=mask&cur_mask;
    for(int i=col+1; i<7; i++){
        cur_mask |= mask_square << (i-col);
        if((mask & cur_mask) != bef)break;
    }
    bef = mask&cur_mask;
    for(int i=1; i<row; i++){
        cur_mask |= mask_square << 8*i;
        if((mask&cur_mask) != bef)break;
    }
    bef = mask&cur_mask;
    for(int i=row+1; i<7; i++){
        cur_mask |= cur_mask << (i-col)*8;
        if((mask&cur_mask) != bef)break;
    }
    return cur_mask;
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
    void _push(big x, int id, born curL, born curR, int T, int depth){
        if(Ns[T].last != -1){
            if(mapping[Ns[T].last] != mapping[id]){
                valid = max(valid, depth+1);
            }
        }
        Ns[T].last = id;
        if(curL+1 == curR)return;
        assert(depth < depthMax);
        born mid = curL+(curR-curL)/2;
        born nL=curL, nR=curR;
        int nT;
        if(x >= mid){
            nL = mid;
            nT = Ns[T].right;
        }else{
            nR = mid;
            nT = Ns[T].left;
        }if(nT == -1){
            nT = Ns.size();
            nbNodesPerDepth[depth]++;
            Ns.push_back(node());
            if(x >= mid)Ns[T].right = nT;
            else Ns[T].left = nT;
        }
        _push(x, id, nL, nR, nT, depth+1);
    }
    void push(big x, big res){
        mapping.push_back(res);
        _push(x, mapping.size()-1, 0, ((born)1ULL)<<depthMax, 0, 0);
    }
};
class info{
public:
    int minimum;
    int decR;
    big magic;
};
info test_magic(big magic, int square){
    int col = square & 7;
    int row = square >> 3;
    int nbBits = 10+(col%7 == 0)+(row%7 == 0);
    creux tree[63];
    for(int i=0; i<63; i++)tree[i] = creux(64-i);
    for(big id=0; id<(1<<nbBits); id++){
        big mask = generate_mask(id, square);
        big res = mask*magic;
        big res_mask = usefull(mask, square);
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
void dump_table(ofstream& file, info magic, int square){
    vector<big> table(1<<magic.minimum);
    int col = square & 7;
    int row = square >> 3;
    int nbBits = 10+(col%7 == 0)+(row%7 == 0);
    for(big id=0; id<(1<<nbBits); id++){
        big mask = generate_mask(id, square);
        big res = mask*magic.magic;
        big res_mask = usefull(mask, square);
        int key = (res&(MAX_BIG>>magic.decR)) >> (64-magic.decR+magic.minimum);
        assert(table[key] == res_mask || table[key] == 0);
        table[key] = res_mask;
    }
    file << magic.magic << " ";
    file << table.size() << " ";
    for(big i:table)
        file << i << " ";
}
int main(int argc, char* argv[]){
    init();
    vector<info> best(64, {20, 0});
    int totLength=(1<<20)*64;
    try{
        while(1){
            big magic = generate();
            for(int square=0; square<64; square++){
                info r=test_magic(magic, square);
                if(r.minimum < best[square].minimum){
                    totLength += (1<<r.minimum)-(1<<best[square].minimum);
                    printf("magic = %16lx\tcase = %d\tneeded bits = %d\tdecRight = %d\n", magic, square, r.minimum, r.decR);
                    best[square] = r;
                }
            }
        }
    }catch(int error){
        printf("error code : %d", error);
        ofstream file(argv[1]);
        for(int square=0; square < 64; square++){
            dump_table(file, best[square], square);
        }
    }
}
