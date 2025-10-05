cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-march=native")

include(FetchContent)
FetchContent_Declare(
  googletest
  #Latest version
  URL https://github.com/google/googletest/archive/3983f67e32fb3e9294487b9d4f9586efa6e5d088.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


#Tests
enable_testing()

#First line is name you want to use
#Second line is location from project folder
add_executable(
  perft_test
  tests/perft_test.cpp 
)

add_executable(
  polyglotHash_test
  tests/polyglotHash_test.cpp 
)

#First line is the name you chose
#Second line is always GTest::gtest_main
target_link_libraries(
  perft_test
  GTest::gtest_main
)

target_link_libraries(
  polyglotHash_test
  GTest::gtest_main
)

include(GoogleTest)

#Put in the name you chose
gtest_discover_tests(perft_test)
gtest_discover_tests(polyglotHash_test)

include_directories(/usr/local/include/tensorflow)
link_directories(/usr/local/lib)

add_executable(
  tf_test 
  tests/tf_test.cpp
)

#Also need tensorflow for this one
target_link_libraries(
  tf_test
  tensorflow
  GTest::gtest_main
)

gtest_discover_tests(tf_test)