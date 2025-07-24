#include <gtest/gtest.h>
#include "../other_core/BestMoveFinder.hpp"
#include "../other_core/GameState.hpp"

const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
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