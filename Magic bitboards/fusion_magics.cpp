#include "util_magic.hpp"

int main(int argc, char** argv){
    assert(argc > 3);
    init_lines();
    vector<vector<info>> table1=load_info(argv[2]);
    vector<vector<info>> table2=load_info(argv[3]);
    vector<vector<info>> table_fusion(2, vector<info>(64));
    for(int is_rook=0; is_rook < 2; is_rook++){
        for(int square=0; square<64; square++){
            if(table1[is_rook][square].minimum <= table2[is_rook][square].minimum)
                table_fusion[is_rook][square] = table1[is_rook][square];
            else
                table_fusion[is_rook][square] = table2[is_rook][square];
        }
    }
    dump_entire(table_fusion, argv[1]);
    print_table(table_fusion);
}