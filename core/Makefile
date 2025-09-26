CC=g++
CFLAGS=-Wall -Wextra -march=native -O3 -std=gnu++17
COM=$(shell git rev-parse HEAD)

all: main

coreTest: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^

main: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -D CalculatePV

coreGen: collectData.cpp
	$(CC) $(CFLAGS) -fopenmp -o $@ $^

coreNV: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -DCalculatePV -DCOMMIT=\"$(COM)\"

coreTV: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -DCOMMIT=\"$(COM)\"

clean:
	rm *.o