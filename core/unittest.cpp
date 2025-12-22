#include "GameState.hpp"
#include "Evaluator.hpp"
#include "BestMoveFinder.hpp"
#include "polyglotHash.hpp"
#include <string>
#include <vector>
#include "viriformatUtil.hpp"
using namespace std;

string suitFens[71] = {
    "6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - -",
    "5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - -",
    "4R3/2r3p1/5bk1/1p1r3p/p2PR1P1/P1BK1P2/1P6/8 b - -",
    "4R3/2r3p1/5bk1/1p1r1p1p/p2PR1P1/P1BK1P2/1P6/8 b - -",
    "4r1k1/5pp1/nbp4p/1p2p2q/1P2P1b1/1BP2N1P/1B2QPPK/3R4 b - -",
    "2r1r1k1/pp1bppbp/3p1np1/q3P3/2P2P2/1P2B3/P1N1B1PP/2RQ1RK1 b - -",
    "7r/5qpk/p1Qp1b1p/3r3n/BB3p2/5p2/P1P2P2/4RK1R w - -",
    "6rr/6pk/p1Qp1b1p/2n5/1B3p2/5p2/P1P2P2/4RK1R w - -",
    "7r/5qpk/2Qp1b1p/1N1r3n/BB3p2/5p2/P1P2P2/4RK1R w - -",
    "6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - -",
    "6RR/4bP2/8/8/5r2/3K4/5p2/4k3 w - -",
    "7R/5P2/8/8/6r1/3K4/5p2/4k3 w - -",
    "7R/5P2/8/8/6r1/3K4/5p2/4k3 w - -",
    "7R/4bP2/8/8/1q6/3K4/5p2/4k3 w - -",
    "8/4kp2/2npp3/1Nn5/1p2PQP1/7q/1PP1B3/4KR1r b - -",
    "8/4kp2/2npp3/1Nn5/1p2P1P1/7q/1PP1B3/4KR1r b - -",
    "2r2r1k/6bp/p7/2q2p1Q/3PpP2/1B6/P5PP/2RR3K b - -",
    "r2qk1nr/pp2ppbp/2b3p1/2p1p3/8/2N2N2/PPPP1PPP/R1BQR1K1 w kq -",
    "6r1/4kq2/b2p1p2/p1pPb3/p1P2B1Q/2P4P/2B1R1P1/6K1 w - -",
    "3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R4B/PQ3P1P/3R2K1 w - h6",
    "3q2nk/pb1r1p2/np6/3P2Pp/2p1P3/2R1B2B/PQ3P1P/3R2K1 w - h6",
    "2r4r/1P4pk/p2p1b1p/7n/BB3p2/2R2p2/P1P2P2/4RK2 w - -",
    "2r5/1P4pk/p2p1b1p/5b1n/BB3p2/2R2p2/P1P2P2/4RK2 w - -",
    "2r4k/2r4p/p7/2b2p1b/4pP2/1BR5/P1R3PP/2Q4K w - -",
    "8/pp6/2pkp3/4bp2/2R3b1/2P5/PP4B1/1K6 w - -",
    "4q3/1p1pr1k1/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - -",
    "4q3/1p1pr1kb/1B2rp2/6p1/p3PP2/P3R1P1/1P2R1K1/4Q3 b - -",
    "3r3k/3r4/2n1n3/8/3p4/2PR4/1B1Q4/3R3K w - -",
    "1k1r4/1ppn3p/p4b2/4n3/8/P2N2P1/1PP1R1BP/2K1Q3 w - -",
    "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - -",
    "rnb2b1r/ppp2kpp/5n2/4P3/q2P3B/5R2/PPP2PPP/RN1QKB2 w Q -",
    "r2q1rk1/2p1bppp/p2p1n2/1p2P3/4P1b1/1nP1BN2/PP3PPP/RN1QR1K1 b - -",
    "r1bqkb1r/2pp1ppp/p1n5/1p2p3/3Pn3/1B3N2/PPP2PPP/RNBQ1RK1 b kq -",
    "r1bq1r2/pp1ppkbp/4N1p1/n3P1B1/8/2N5/PPP2PPP/R2QK2R w KQ -",
    "r1bq1r2/pp1ppkbp/4N1pB/n3P3/8/2N5/PPP2PPP/R2QK2R w KQ -",
    "rnq1k2r/1b3ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq -",
    "rn2k2r/1bq2ppp/p2bpn2/1p1p4/3N4/1BN1P3/PPP2PPP/R1BQR1K1 b kq -",
    "r2qkbn1/ppp1pp1p/3p1rp1/3Pn3/4P1b1/2N2N2/PPP2PPP/R1BQKB1R b KQq -",
    "rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQ -",
    "r4rk1/3nppbp/bq1p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - -",
    "r4rk1/1q1nppbp/b2p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - -",
    "1r3r2/5p2/4p2p/2k1n1P1/2PN1nP1/1P3P2/8/2KR1B1R b - -",
    "1r3r2/5p2/4p2p/4n1P1/kPPN1nP1/5P2/8/2KR1B1R b - -",
    "2r2rk1/5pp1/pp5p/q2p4/P3n3/1Q3NP1/1P2PP1P/2RR2K1 b - -",
    "5rk1/5pp1/2r4p/5b2/2R5/6Q1/R1P1qPP1/5NK1 b - -",
    "1r3r1k/p4pp1/2p1p2p/qpQP3P/2P5/3R4/PP3PP1/1K1R4 b - -",
    "1r5k/p4pp1/2p1p2p/qpQP3P/2P2P2/1P1R4/P4rP1/1K1R4 b - -",
    "r2q1rk1/1b2bppp/p2p1n2/1ppNp3/3nP3/P2P1N1P/BPP2PP1/R1BQR1K1 w - -",
    "rnbqrbn1/pp3ppp/3p4/2p2k2/4p3/3B1K2/PPP2PPP/RNB1Q1NR w - -",
    "rnb1k2r/p3p1pp/1p3p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq -",
    "r1b1k2r/p4npp/1pp2p1b/7n/1N2N3/3P1PB1/PPP1P1PP/R2QKB1R w KQkq -",
    "2r1k2r/pb4pp/5p1b/2KB3n/4N3/2NP1PB1/PPP1P1PP/R2Q3R w k -",
    "2r1k2r/pb4pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w k -",
    "2r1k3/pbr3pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w - -",
    "5k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ -",
    "r4k2/p2P2pp/8/1pb5/1Nn1P1n1/6Q1/PPP4P/R3K1NR w KQ -",
    "5k2/p2P2pp/1b6/1p6/1Nn1P1n1/8/PPP4P/R2QK1NR w KQ -",
    "4kbnr/p1P1pppp/b7/4q3/7n/8/PP1PPPPP/RNBQKBNR w KQk -",
    "4kbnr/p1P1pppp/b7/4q3/7n/8/PPQPPPPP/RNB1KBNR w KQk -",
    "4kbnr/p1P1pppp/b7/4q3/7n/8/PPQPPPPP/RNB1KBNR w KQk -",
    "4kbnr/p1P4p/b1q5/5pP1/4n3/5Q2/PP1PPP1P/RNB1KBNR w KQk f6",
    "4kbnr/p1P4p/b1q5/5pP1/4n3/5Q2/PP1PPP1P/RNB1KBNR w KQk f6",
    "4kbnr/p1P4p/b1q5/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk f6",
    "1n2kb1r/p1P4p/2qb4/5pP1/4n2Q/8/PP1PPP1P/RNB1KBNR w KQk -",
    "rnbqk2r/pp3ppp/2p1pn2/3p4/3P4/N1P1BN2/PPB1PPPb/R2Q1RK1 w kq -",
    "3N4/2K5/2n5/1k6/8/8/8/8 b - -",
    "3n3r/2P5/8/1k6/8/8/3Q4/4K3 w - -",
    "r2n3r/2P1P3/4N3/1k6/8/8/8/4K3 w - -",
    "8/8/8/1k6/6b1/4N3/2p3K1/3n4 w - -",
    "8/8/1k6/8/8/2N1N3/4p1K1/3n4 w - -",
    "r1bqk1nr/pppp1ppp/2n5/1B2p3/1b2P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
};

string suitMoves[71] = {
    "f1f4","d6f4","h5g4","h5g4","g4f3","d6e5","e1e8","e1e8","e1e8","f7f8q",
    "f7f8n","f7f8q","f7f8b","f7f8r","h1f1","h1f1","c5c1","f3e5","f4e5","g5h6",
    "g5h6","c3c8","c3c8","c3c5","g2c6","e6e4","h7e4","d3d4","d3e5","d3e5",
    "h4f6","g4f3","c6d4","e6g7","e6g7","d6h2","d6h2","g4f3","b4c3","b6b2",
    "f6d5","b8b3","b8b4","c8c1","f5c2","a5a2","a5a2","d5e7","d3e4","e4d6",
    "e4d6","d5c6","d5c6","d5c6","d7d8q","d7d8q","d7d8q","c7c8q","c7c8q","c7c8q",
    "g5f6","g5f6","g5f6","c7b8q","g1h2","c6d8","c7d8q","e6d8","e3d1","c3d1",
    "e1g1"
};
class Init{
public:
    Init(){
        PrecomputeKnightMoveData();
        init_lines();
        load_table();
        precomputePawnsAttack();
        precomputeCastlingMasks();
        precomputeNormlaKingMoves();
        precomputeDirections();
        init_zobrs();
    }
};
Init _init_everything;

void testSEE(){
    for(int i=0; i<71; i++){
        GameState state;
        state.fromFen(suitFens[i]);
        Move move;
        move.from_uci(suitMoves[i]);
        state.initMove(move);
        SEE_BB bb(state);
        int threshold = -fastSEE(move, state);
        if(move.capture != -2)
            threshold += value_pieces[max<int8_t>(0, move.capture)];
        if(!see_ge(bb, threshold, move, state))
            printf("%s & %s & %d\n", suitFens[i].c_str(), suitMoves[i].c_str(), threshold);
    }
}
const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const string position2FromCPW = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
const string position3FromCPW = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const string position4FromCPW = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const string position5FromCPW = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
struct PTest{
    string fen;
    vector<int> expResults;
};
const PTest TestsPerft[] = {
    {StartFen, {
        20,
        400,
        8902,
        197281
    }},
    {position2FromCPW, {
        48,
        2039,
        97862
    }},
    {position3FromCPW, {
        14,
        191,
        2812,
        43238,
        674624
    }},
    {position4FromCPW, {
        6,
        264,
        9467,
    }},
    {position5FromCPW, {
        44,
        1486,
        62379
    }},
    {"7k/8/8/1Pp5/1K6/8/8/7B w - c6 0 2", {
        8,
        29,
        369
    }},
    {"rnbqk1nr/pppp1pbp/6p1/4P3/8/2K5/PPP1PPPP/RNBQ1BNR b kq - 0 4", {
        28,
        912
    }},
    {"8/2p5/3p4/KP5r/4Ppk1/8/6P1/7R b - e3 0 3", {20}},
    {"8/p2n1p2/1p1Pp2p/4P1k1/r4p1P/5K2/6P1/4R3 b - - 0 1", {
        4,
        59,
        1219,
        16899,
        347600,
        5076659
    }}
};

void testPerft(){
    Perft perft;
    for(PTest test: TestsPerft){
        GameState game;
        game.fromFen(test.fen);
        for(int depth=0; depth<(int)test.expResults.size(); depth++){
            int res = perft.perft(game, depth+1, false);
            if(test.expResults[depth] != res){
                printf("ERROR : fen %s expected %d != result %d\n", test.fen.c_str(), test.expResults[depth], res);
            }
        }
    }
}

const pair<string, big> TestsPolyHash[] = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0x463b96181691fc9c},
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0x823c9b50fd114196},
    {"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 0x22a48b5a8e47ff78},
    {"rnbqkbnr/p1pppppp/8/8/PpP4P/8/1P1PPPP1/RNBQKBNR b KQkq c3 0 3", 0x3c8123ea7b067637},
    {"rnbqkbnr/p1pppppp/8/8/P6P/R1p5/1P1PPPP1/1NBQKBNR b Kkq - 0 4", 0x5c3f9b829b279560},
    {"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR b kq - 0 3", 0x652a607ca3f242c1},
    {"r1bqkbnr/ppp1pppp/2n5/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R b KQkq - 1 3", 15834916877423137634ul}
};

void testPolyHash(){
    for(auto test:TestsPolyHash){
        GameState game;
        game.fromFen(test.first);
        big resHash = polyglotHash(game);
        if(resHash != test.second){
            printf("ERROR : fen %s\nexpected %016" PRIx64 "\nresult   %016" PRIx64 "\n", test.first.c_str(), test.second, resHash);
        }
    }
}

vector<string> Games[] = {
    {"e2e4", "e7e5", "d1h5", "e8e7", "h5e5"},
    {"e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5", "e1g1", "e8g8"},
    {"e2e4", "d7d5", "g1f3", "c8g4", "f1b5", "b8c6", "e1g1", "d8d6", "f1e1", "e8c8"}
};

void testViri(){
    FILE* fptr;
    fptr = fopen("test.out", "wb");
    for(vector<string> moves:Games){
        GamePlayed game;
        game.startPos.fromFen(startpos);
        GameState state;
        state.fromFen(startpos);
        for(string move:moves){
            MoveInfo mv;
            mv.move.from_uci(move);
            mv.move = state.playPartialMove(mv.move);
            game.game.push_back(mv);
        }
        game.result = 2;
        game.dump(fptr);
    }
    fclose(fptr);
}

int main(){
    testSEE();
    testPerft();
    testPolyHash();
    testViri();
}