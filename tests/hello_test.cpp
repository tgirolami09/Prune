//AN EXAMPLE

//Must be included
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(HelloTestNameChange, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
  EXPECT_NE(10 * 1, 0 * 10);
}