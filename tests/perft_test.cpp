#include <gtest/gtest.h>
#include "../core/BestMoveFinder.hpp"
#include "../core/GameState.hpp"

const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const string position2FromCPW = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
const string position3FromCPW = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const int alloted_space=64*1000;

TEST(PerftTests, perft1) { 
    
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,1), 20);
}

TEST(PerftTests, perft2) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,2), 400);
}

TEST(PerftTests, perft3) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,3), 8902);
}

TEST(PerftTests, perft4) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(StartFen);
    EXPECT_EQ(perft.perft(game,4), 197281);
}

TEST(Position2FromCPW, perft1) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,1), 48);
}

TEST(Position2FromCPW, perft2) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,2), 2039);
}

TEST(Position2FromCPW, perft3) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position2FromCPW);
    EXPECT_EQ(perft.perft(game,3), 97862);
}

TEST(Position3FromCPW, perft1) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,1), 14);
}

TEST(Position3FromCPW, perft2) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,2), 191);
}

TEST(Position3FromCPW, perft3) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,3), 2812);
}

TEST(Position3FromCPW, perft4) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,4), 43238);
}

TEST(Position3FromCPW, perft5) { 
    Perft perft(alloted_space);
    GameState game;
    game.fromFen(position3FromCPW);
    EXPECT_EQ(perft.perft(game,5), 674624);
}