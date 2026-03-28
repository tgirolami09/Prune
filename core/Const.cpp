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

inline big mask_empty_rook(int square){
    return (clipped_col[square&7]|clipped_row[square >> 3])&(~(1ULL << square));
}
inline big mask_empty_bishop(int square){
    int col=square&7, row=square >> 3;
    return clipped_diag[col+row] ^ clipped_idiag[row-col+7];
}
inline big mask_full_rook(int square){
    return (mask_col[square&7]|mask_row[square >> 3])&(~(1ULL << square));
}
inline big mask_full_bishop(int square){
    int col=square&7, row=square >> 3;
    return mask_diag[col+row] ^ mask_idiag[row-col+7];
}

__attribute__((constructor(101))) void init_lines(){
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
        bishop_empty[i] = mask_empty_bishop(i);
        bishop_full[i] = mask_full_bishop(i);
        rook_empty[i] = mask_empty_rook(i);
        rook_full[i] = mask_full_rook(i);
    }
}

bool pawnStruct::operator==(pawnStruct s){
    return blackPawn == s.blackPawn && whitePawn == s.whitePawn;
}