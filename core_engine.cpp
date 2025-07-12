#include "util_magic.cpp"
#include <strings.h>
#include <map>
//#define ALTERNATE_COLOR
big* table[128]; // [bishop, rook]
info constants[128];
const int WHITE=0;
const int BLACK=1; // odd = black
const int PAWN=0;
const int KNIGHT=2;
const int BISHOP=4;
const int ROOK=6;
const int QUEEN=8;
const int KING=10;
const int SPACE=12;
map<char, int> piece_to_id = {{'r', ROOK}, {'n', KNIGHT}, {'b', BISHOP}, {'q', QUEEN}, {'k', KING}, {'p', PAWN}};
map<int, char> id_to_piece = {{ROOK, 'r'}, {KNIGHT, 'n'}, {BISHOP, 'b'}, {QUEEN, 'q'}, {KING, 'k'}, {PAWN, 'p'}, {SPACE, ' '}};

class coup{
public:
    int start_position;
    int end_position;
};

const int nbBitboard=12;
class chess{
public:
    big bitboards[nbBitboard];
    int passant;
    big castle_rights;
    bool turn; // WHITE|BLACK
    chess(string fen){
        int id=0;
        int dec=63;
        for(int i=0; i<nbBitboard; i++)
            bitboards[i] = 0;
        for(; id<fen.size(); id++){
            char c=fen[id];
            if(isalpha(c)){
                int piece=piece_to_id[tolower(c)];
                if(isupper(c))
                    piece += WHITE;
                else
                    piece += BLACK;
                bitboards[piece] |= 1ULL << dec;
                dec--;
            }else if(isdigit(c)){
                dec -= c-'0';
            }else if(c == ' ')break;
        }
        id++;
        turn = fen[id] == 'b';
        id += 2;
        if(fen[id] == '-')
            castle_rights = 0, id++;
        else{
            for(; id<fen.size(); id++){
                if(fen[id] == ' ')break;
                int place=0;
                if(isupper(fen[id])){
                    place += 8*7;
                }
                if(tolower(fen[id]) == 'k')
                    place += 7;
                castle_rights |= 1ULL << place;
            }
        }
        id++;
        if(fen[id] == '-')passant = -1;
        else passant = fen[id]-'a', id++;
        id += 2;

    }
    chess(){
        *this = chess("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    chess(chess& board){
        for(int i=0; i<nbBitboard; i++){
            bitboards[i] = board.bitboards[i];
        }
        passant = board.passant;
        castle_rights = board.castle_rights;
        turn = board.turn;
    }
    private: void change_to_black(){
        printf("\x1b[37;40m");
    }
    void reset(){
        printf("\e[0m");
    }
    void change_to_white(){
        printf("\e[30;47m");
    }
    public: void print(){
        printf("/−");
        for(int i=1; i<7; i++){
            printf("−−");
        }
        printf("−−\\\n");
        for(int row=0; row<8; row++){
            printf("|");
            for(int col=0; col<8; col++){
                big mask = 1ULL << (63-(row << 3 | col));
                int piece = SPACE;
                for(int i=0; i<nbBitboard; i++){
                    if(bitboards[i] & mask){
                        piece = i;
                        break;
                    }
                }
                char c;
                if(piece == SPACE)
                    c = ' ';
                else{
                    c=id_to_piece[piece&(~1)];
                    if((piece&1) == WHITE){
                        c = toupper(c);
                    }
                }
#ifdef ALTERNATE_COLOR
                if(row+col&1)
                    change_to_black();
                else change_to_white();
#endif
                printf("%c", c);
#ifdef ALTERNATE_COLOR
                reset();
#endif
                printf("|");
            }
            printf("\n");
            if(row != 7){
                printf("|");
                for(int i=0; i<7; i++){
                    printf("−+");
                }
                printf("−|\n");
            }
        }
        printf("\\−");
        for(int i=1; i<7; i++){
            printf("−−");
        }
        printf("−−/\n");
    }
    big moves_table(int index, big mask_pieces){
        return table[index][(mask_pieces*constants[index].magic & (MAX_BIG >> constants[index].decR)) >> (64-constants[index].decR-constants[index].minimum)];
    }
    inline big _moves_rook(int square, big mask_pieces){
        return moves_table(square+64, mask_pieces);
    }
    inline big _moves_bishop(int square, big mask_pieces){
        return moves_table(square, mask_pieces);
    }
    inline big mask_empty_rook(int square){
        return (clipped_col[square&7]|clipped_row[square >> 3])&(~(1ULL << square));
    }
    inline big mask_empty_bishop(int square){
        int col=square&7, row=square >> 3;
        return clipped_diag[col+row] ^ clipped_idiag[row-col+7];
    }
    inline big mask_moves(int square, big black_pieces, big white_pieces, const char type, bool color){
        big mask=0;
        big ens=black_pieces|white_pieces;
        if(type&1){
            big _mask = mask_empty_bishop(square);
            mask = _moves_bishop(square, _mask&ens);
        }if(type & 2){
            mask |= _moves_rook(square, mask_empty_rook(square)&ens);
        }
        if(color)
            mask &= ~black_pieces;
        else mask &= ~white_pieces;
        return mask;
    }
    vector<int> mask_to_moves(big mask){
        vector<int> res;
        while(mask){
            int bit = ffsll(mask);
            mask ^= 1 << bit;
            res.push_back(bit);
        }
        return res;
    }
    vector<coup> get_pseudo_legal_moves();
    vector<coup> get_legal_moves();
};

void load_table(char* name){
    load_whole(constants, table, name);
}

int main(){
    chess board;
    board.print();
}