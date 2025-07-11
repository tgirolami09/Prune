//An EXAMPLE

#include <gtest/gtest.h>

TEST(ComplexTest, InitialTest) {
  // Expect equality.
  int limit = 10;
  int sum = limit*(limit+1)/2;
  for (int i = 0;i<limit;++i){
    sum-=(i+1);
  }
  EXPECT_EQ(sum, 0);

}

//You can put multiple tests in a single file
TEST(ComplexTest, SecondTest) {
  // Expect equality.
  int limit = 1000;
  int sum = limit*(limit+1)/2;
  for (int i = 0;i<limit;++i){
    sum-=(i+1);
  }
  EXPECT_EQ(sum, 0);
}