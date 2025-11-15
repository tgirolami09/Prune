#include <cstdint>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>
using namespace std;
#define big uint64_t
#define born __uint128_t
const big MAX_BIG = ~((big)0);

template<typename T>
void write_bytes(T a, ofstream& file){
    file.write(reinterpret_cast<char*>(&a), sizeof(a));
}

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

big clipped_row[8];
big clipped_col[8];
big clipped_diag[15];
big clipped_idiag[15];
const big clipped_brow = (MAX_BIG >> 16 << 8);
const big clipped_bcol = (~0x8181818181818181);
const big clipped_mask = clipped_brow & clipped_bcol;
void init_lines(){
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

class info{
public:
    int minimum;
    int decR;
    big magic;
};

big apply_id(big id, big mask){
    big square_mask = 1;
    big new_mask=0;
    while(mask){
        int bit=__builtin_ctzll(mask);
        big m=1ULL << bit;
        if((id&1) == 1)
            new_mask |= m;
        id >>= 1;
        mask ^= m;
    }
    return new_mask;
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

inline big go_dir(big mask, int square, int dir, big clipped){
    int cur_square = square;
    big cur_mask=0;
    big p;
    do{
        cur_square += dir;
        if(cur_square < 0 || cur_square >= 64)break;
        p = 1ULL << cur_square;
        cur_mask |= p;
    }while((clipped&p) && (p&mask) == 0);
    return cur_mask;
}

big usefull_rook(big mask, int square){
    big mask_square = 1ULL << square;
    int col = square&7;
    int row = square >> 3;
    big cur_mask = 0;
    if(col != 7)
        cur_mask |= go_dir(mask, square, 1, clipped_bcol);
    if(col != 0)
        cur_mask |= go_dir(mask, square, -1, clipped_bcol);
    if(row != 7)
        cur_mask |= go_dir(mask, square, 8, clipped_brow);
    if(row != 0)
        cur_mask |= go_dir(mask, square, -8, clipped_brow);
    return cur_mask;
}

big usefull_bishop(big mask, int square){
    big cur_mask=0;
    int col=square&7;
    int row=square >> 3;
    if(col != 0){
        if(row != 7)
            cur_mask |= go_dir(mask, square, +7, clipped_mask);
        if(row != 0)
            cur_mask |= go_dir(mask, square, -9, clipped_mask);
    }if(col != 7){
        if(row != 7)
            cur_mask |= go_dir(mask, square, +9, clipped_mask);
        if(row != 0)
            cur_mask |= go_dir(mask, square, -7, clipped_mask);
    }
    return cur_mask&(~(1ULL << square));
}

big get_usefull(bool is_rook, big mask, int square){
    return (is_rook?usefull_rook:usefull_bishop)(mask, square);
}
int dump_table(ofstream& file, info magic, int square, bool is_rook){
    vector<big> table(1<<magic.minimum);
    vector<big> last_mask(1<<magic.minimum);
    int col = square & 7;
    int row = square >> 3;
    int nbBits = __builtin_popcountll(get_mask(is_rook, MAX_BIG, square));
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
            printf("%llu\n", res_mask);
            printf("%llu\n", (res&(MAX_BIG>>magic.decR)));
            printf("%d\n", (64-magic.decR-magic.minimum));
            printf("%llu\n", key);
            assert(false);
        }
        if(table[key] != res_mask && table[key] != 0){
            printf("magic:%llu\n", magic.magic);
            printf("id:%llu\n", id);
            printf("res:%llu\nmask:\n", res);
            print_mask(mask);
            printf("\nlast_mask:\n");
            print_mask(last_mask[key]);
            printf("key:%llu\nlast usefull:\n", key);
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
    write_bytes(magic.magic, file);
    write_bytes(magic.decR, file);
    write_bytes(magic.minimum, file);
    write_bytes<int>(table.size(), file);
    int res=0;
    for(big i:table){
        write_bytes(i, file);
        if(i == 0)res++;
    }
    return res;
}

int dump_entire(vector<vector<info>> table, char* name){
    ofstream file(name);
    int place_lost=0;
    for(int is_rook=0; is_rook<2; is_rook++){
        for(int square=0; square < 64; square++){
            place_lost += dump_table(file, table[is_rook][square], square, is_rook);
        }
    }
    file.close();
    return place_lost;
}

vector<vector<info>> load_info(const char* name){
    ifstream file(name);
    if(file.good()){ // exist
        vector<info> bests;
        big magic;
        int decR, minimum;
        while(file.read(reinterpret_cast<char*>(&magic), sizeof(magic))){
            int size;
            file.read(reinterpret_cast<char*>(&decR), sizeof(decR));
            file.read(reinterpret_cast<char*>(&minimum), sizeof(minimum));
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            bests.push_back({minimum, decR, magic});
            for(int i=0; i<size; i++){
                big mask;
                file.read(reinterpret_cast<char*>(&mask), sizeof(mask));
            }
        }
        if(bests.size() == 64)
            return {vector<info>(64, {20, 0, 1}), bests};
        else{
            if(bests.size() != 128){
                printf("this file is not incomplet or have too much elements : %ld/128", bests.size());
                exit(1);
            }
            vector<vector<info>> res(2);
            res[0] = vector<info>(bests.begin(), bests.begin()+64);
            res[1] = vector<info>(bests.begin()+64, bests.end());
            return res;
        }
    }else return {};
}

void load_whole(info* constants, big** table, char* name){
    ifstream file(name);
    big magic;
    int decR, minimum, size;
    big mask;
    int current = 0;
    while(file.read(reinterpret_cast<char*>(&magic), sizeof(magic))){
        file.read(reinterpret_cast<char*>(&decR), sizeof(decR));
        file.read(reinterpret_cast<char*>(&minimum), sizeof(minimum));
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        constants[current] = {minimum, decR, magic};
        table[current] = (big*)calloc(size, sizeof(big));
        for(int i=0; i<size; i++){
            file.read(reinterpret_cast<char*>(&mask), sizeof(mask));
            table[current][i] = mask;
        }
        current++;
    }
}

void print_table(vector<vector<info>> table){
    for(int is_rook=0; is_rook<2; is_rook++){
        printf(is_rook?"rook\n":"bishop\n");
        for(int row=0; row<8; row++){
            for(int col=0; col<8; col++)
                printf("%16llx ", table[is_rook][row << 3 | col].magic);
            printf("\n");
        }
        for(int row=0; row<8; row++){
            for(int col=0; col<8; col++)
                printf("%2d ", table[is_rook][row << 3 | col].minimum);
            printf("\n");
        }
        //for(int square=0; square < 64; square++)
        //    printf("\tcase: %2d magic: %16llx bits needed: %2d dec right: %2d\n", square, table[is_rook][square].magic, table[is_rook][square].minimum, table[is_rook][square].decR);
    }
}