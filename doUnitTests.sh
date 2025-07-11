#!/bin/sh

cmake -S . -B build || break

echo "---------------------------------------------------"

cmake --build build || break

echo "------------------Finished building----------------"

#Run the tests
cd build && ctest || break

#Go back in the project folder
cd ../ 