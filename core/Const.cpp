#include "Const.hpp"

big clipped_row[8];
big clipped_col[8];
big clipped_diag[15];
big clipped_idiag[15];
big mask_row[8];
big mask_col[8];
big mask_diag[15];
big mask_idiag[15];
big bishop_empty[64];
big rook_empty[64];
big bishop_full[64];
big rook_full[64];


big directions[64][64];
big fullDir[64][64];

__attribute__((constructor(101))) void init_lines(){
    {
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
    row = 0xff;
    col = 0x0101010101010101LL;
    for(int i=0; i<8; i++){
        mask_row[i] = row;
        mask_col[i] = col;
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
        mask_diag[i] = diag;
        mask_idiag[i] = idiag;
    }
    for(int i=0; i<64; i++){
        int colP=i&7, rowP=i>>3;
        big maskPos = ~(1ULL << i);
        bishop_empty[i] = clipped_diag[colP+rowP] ^ clipped_idiag[rowP-colP+7];
        bishop_full[i] = mask_diag[colP+rowP] ^ mask_idiag[rowP-colP+7];
        rook_empty[i] = (clipped_col[colP]|clipped_row[rowP])&maskPos;
        rook_full[i] = (mask_col[colP]|mask_row[rowP])&maskPos;
    }
    }
    //Set everything to 0 first just to be sure
    for (int i = 0; i < 64; ++i){
        for (int j = 0; j < 64; ++j){
            directions[i][j] = 0;
            fullDir[i][j] = 0;
        }    
    }
    for(int row=0; row<8; row++){
        for(int col=0; col<8; col++){
            int square = row*8+col;
            for(int idDir=0; idDir<8; idDir++){
                int r=row+dirs[idDir][0];
                int c=col+dirs[idDir][1];
                big mask = 0;
                while(r >= 0 && r < 8 && c >= 0 && c < 8){
                    int sq = (r*8+c);
                    mask |= 1ULL << sq;
                    directions[square][sq] = mask; // line of 1 between square and sq
                    r += dirs[idDir][0];
                    c += dirs[idDir][1];
                }
                r=row+dirs[idDir][0];
                c=col+dirs[idDir][1];
                while(r >= 0 && r < 8 && c >= 0 && c < 8){
                    int sq = (r*8+c);
                    fullDir[square][sq] = mask; // line of 1 from square in the direction of sq
                    r += dirs[idDir][0];
                    c += dirs[idDir][1];
                }
            }
        }
    }
}