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
    for(int row=8; row<=64; row+=8){
        for(; col<row; col++){
            printf("%d", (int)mask&1);
            mask >>= 1;
        }
        printf("\n");
    }
}
std::random_device rd;
std::mt19937_64 e2(42);
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
big clipped_row[8];
big clipped_col[8];
big clipped_diag[15];
big clipped_idiag[15];
big clipped_mask = (MAX_BIG >> 16 << 8) & (~0x8181818181818181);
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
    big diag = 0;
    big idiag = 0;
    for(int i=0; i<15; i++){
        diag <<= 8;
        if(i < 8)diag |= 1 << i;
        idiag <<= 8;
        if(i < 8)idiag |= 1 << (7-i);
        clipped_diag[i] = diag&clipped_mask;
        clipped_idiag[i] = idiag&clipped_mask;
        /*print_mask(clipped_diag[i]);
        printf("\n");
        print_mask(clipped_idiag[i]);
        printf("\n");*/
    }
}
big generate(){
    return dist(e2);
}

big apply_id(big id, big mask){
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

big rook_mask(big id, big square){
    big mask = (clipped_row[square>>3]|clipped_col[square&7])&(~(1ULL<<square));
    return apply_id(id, mask);
}

big bishop_mask(big id, big square){
    int col = square&7;
    int row = square>>3;
    big mask = (clipped_diag[col+row]|clipped_idiag[row-col+7])&(~(1ULL<<square));
    return apply_id(id, mask);
}

big get_mask(bool is_rook, big id, big square){
    return (is_rook?rook_mask:bishop_mask)(id, square);
}

big usefull_rook(big mask, int square){
    big mask_square = 1ULL << square;
    int col = square&7;
    int row = square >> 3;
    big cur_mask = 0;
    for(int i=1; i <= col; i++){
        cur_mask |= mask_square >> i;
        if(mask&cur_mask)break;
    }
    big bef=mask&cur_mask;
    for(int i=col+1; i<8; i++){
        cur_mask |= mask_square << (i-col);
        if((mask & cur_mask) != bef)break;
    }
    bef = mask&cur_mask;
    for(int i=1; i<=row; i++){
        cur_mask |= mask_square >> 8*i;
        if((mask&cur_mask) != bef)break;
    }
    bef = mask&cur_mask;
    for(int i=row+1; i<8; i++){
        cur_mask |= mask_square << (i-row)*8;
        if((mask&cur_mask) != bef)break;
    }
    return cur_mask;
}

big usefull_bishop(big mask, int square){
    big cur_mask=0;
    vector<int> poss;
    if((square&7) != 0)
        poss.push_back(+7), poss.push_back(-9);
    if((square&7) != 7)
        poss.push_back(+9), poss.push_back(-7);
    for(int direction:poss){
        int cur_square = square;
        big p;
        do{
            cur_square += direction;
            if(cur_square < 0 || cur_square >= 64)break;
            p = 1ULL << cur_square;
            cur_mask |= p;
        }while(clipped_mask&(1ULL<<cur_square) && (p&mask) == 0);
        //cur_mask |= (1ULL << cur_square);
    }
    return cur_mask&(~(1ULL << square));
}

big get_usefull(bool is_rook, big mask, int square){
    return (is_rook?usefull_rook:usefull_bishop)(mask, square);
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
class info{
public:
    int minimum;
    int decR;
    big magic;
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
int dump_table(ofstream& file, info magic, int square, bool is_rook){
    vector<big> table(1<<magic.minimum);
    vector<big> last_mask(1<<magic.minimum);
    int col = square & 7;
    int row = square >> 3;
    int nbBits = 10+(col%7 == 0)+(row%7 == 0);
    for(big id=0; id<(1<<nbBits); id++){
        big mask = get_mask(is_rook, id, square);
        big res = mask*magic.magic;
        big res_mask = get_usefull(is_rook, mask, square);
        big key;
        if(magic.minimum == 0)
            key = 0;
        else
            key = (res&(MAX_BIG>>magic.decR)) >> (64-magic.decR-magic.minimum);
        if(key >= table.size()){
            print_mask(mask);
            printf("%lu\n", res_mask);
            printf("%lu\n", (res&(MAX_BIG>>magic.decR)));
            printf("%d\n", (64-magic.decR-magic.minimum));
            printf("%lu\n", key);
            assert(false);
        }
        if(table[key] != res_mask && table[key] != 0){
            printf("magic:%lu\n", magic.magic);
            printf("id:%ld\n", id);
            printf("res:%ld\nmask:\n", res);
            print_mask(mask);
            printf("\nlast_mask:\n");
            print_mask(last_mask[key]);
            printf("key:%ld\nlast usefull:\n", key);
            print_mask(table[key]);
            printf("\nusefull\n");
            print_mask(res_mask);
            printf("\n");
            printf("square:%d\n", square);
            printf("decr:%d minimum:%d\n", magic.decR, magic.minimum);
            assert(false);
        }
        table[key] = res_mask;
        last_mask[key] = mask;
    }
    file << magic.magic << " ";
    file << magic.decR << " ";
    file << magic.minimum << " ";
    file << table.size() << " ";
    int res=0;
    for(big i:table){
        file << i << " ";
        if(i == 0)res++;
    }
    return res;
}
int main(int argc, char* argv[]){
    init();
    vector<vector<info>> best(2, vector<info>(64, {20, 0, 1}));
    int totLength=(1<<20)*64*2;
    int ok=0;
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
            ofstream file(argv[1]);
            int place_lost=0;
            for(int is_rook=0; is_rook<2; is_rook++){
                for(int square=0; square < 64; square++){
                    place_lost += dump_table(file, best[is_rook][square], square, is_rook);
                }
            }
            file.close();
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
