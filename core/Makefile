CC=g++
CFLAGS=-Wall -Wextra -march=native -O3 -std=gnu++17

all: main

coreTest: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^

main: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -D CalculatePV

coreGen: collectData.cpp
	$(CC) $(CFLAGS) -fopenmp -o $@ $^

clean:
	rm *.o