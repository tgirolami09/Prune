int nbThreads = 1;
#include <gtest/gtest.h>
#include "../core/BestMoveFinder.cpp"
#include "../core/GameState.cpp"
#include "../core/LegalMoveGenerator.cpp"
#include "../core/TranspositionTable.cpp"
#include "../core/MoveOrdering.cpp"
#include "../core/Evaluator.cpp"
#include "../core/Move.cpp"
#include "../core/Functions.cpp"
#include "../core/Const.cpp"
#include "../core/TimeManagement.cpp"
#include "../core/loadpolyglot.cpp"
#include "../core/polyglotHash.cpp"
#include "../core/NNUE.cpp"
#include "../core/corrhist.cpp"
const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const string position2FromCPW = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
const string position3FromCPW = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const string position4FromCPW = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const string position5FromCPW = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const int alloted_space=64*1000;

TEST(PerftTests, perft1) { 
    
    Perft perft;
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,1), 20);
}

TEST(PerftTests, perft2) { 
    Perft perft;
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,2), 400);
}

TEST(PerftTests, perft3) { 
    Perft perft;
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,3), 8902);
}

TEST(PerftTests, perft4) { 
    Perft perft;
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,4), 197281);
}

TEST(Position2FromCPW, perft1) { 
    Perft perft;
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,1), 48);
}

TEST(Position2FromCPW, perft2) { 
    Perft perft;
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,2), 2039);
}

TEST(Position2FromCPW, perft3) { 
    Perft perft;
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,3), 97862);
}

TEST(Position3FromCPW, perft1) { 
    Perft perft;
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,1), 14);
}

TEST(Position3FromCPW, perft2) { 
    Perft perft;
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,2), 191);
}

TEST(Position3FromCPW, perft3) { 
    Perft perft;
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,3), 2812);
}

TEST(Position3FromCPW, perft4) { 
    Perft perft;
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,4), 43238);
}

TEST(Position3FromCPW, perft5) { 
    Perft perft;
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,5), 674624);
}

TEST(Position4FromCPW, perft1) { 
    Perft perft;
    GameState game;
    game.fromFen(position4FromCPW);
    EXPECT_EQ(perft.perft(game,1), 6);
}

TEST(Position4FromCPW, perft2) { 
    Perft perft;
    GameState game;
    game.fromFen(position4FromCPW);
    EXPECT_EQ(perft.perft(game,2), 264);
}

TEST(Position4FromCPW, perft3) { 
    Perft perft;
    GameState game;
    game.fromFen(position4FromCPW);
    EXPECT_EQ(perft.perft(game,3), 9467);
}

TEST(Position5FromCPW, perft1) { 
    Perft perft;
    GameState game;
    game.fromFen(position5FromCPW);
    EXPECT_EQ(perft.perft(game,1), 44);
}

TEST(Position5FromCPW, perft2) { 
    Perft perft;
    GameState game;
    game.fromFen(position5FromCPW);
    EXPECT_EQ(perft.perft(game,2), 1486);
}

TEST(Position5FromCPW, perft3) { 
    Perft perft;
    GameState game;
    game.fromFen(position5FromCPW);
    EXPECT_EQ(perft.perft(game,3), 62379);
}

TEST(PreventBishopFromEnPassant, perft1) {
    Perft perft;
    GameState game;
    game.fromFen("7k/8/8/1Pp5/1K6/8/8/7B w - c6 0 2");
    EXPECT_EQ(perft.perft(game,1), 8);
}

TEST(PreventBishopFromEnPassant, perft2) {
    Perft perft;
    GameState game;
    game.fromFen("7k/8/8/1Pp5/1K6/8/8/7B w - c6 0 2");
    EXPECT_EQ(perft.perft(game,2), 29);
}

TEST(PreventBishopFromEnPassant, perft3) {
    Perft perft;
    GameState game;
    game.fromFen("7k/8/8/1Pp5/1K6/8/8/7B w - c6 0 2");
    EXPECT_EQ(perft.perft(game,3), 369);
}
TEST(EnPassantInRay, perft2){
    Perft perft;
    GameState game;
    game.fromFen("rnbqk1nr/pppp1pbp/6p1/4P3/8/2K5/PPP1PPPP/RNBQ1BNR b kq - 0 4");
    EXPECT_EQ(perft.perft(game, 2), 912);
}
TEST(EnPassantNothing, perft1){
    Perft perft;
    GameState game;
    game.fromFen("8/2p5/3p4/KP5r/4Ppk1/8/6P1/7R b - e3 0 3");
    EXPECT_EQ(perft.perft(game, 1), 20);
}
TEST(EnPassantLogicTest, perft6){
    Perft perft;
    GameState game;
    game.fromFen("8/p2n1p2/1p1Pp2p/4P1k1/r4p1P/5K2/6P1/4R3 b - - 0 1");
    EXPECT_EQ(perft.perft(game, 6), 5076659);
}