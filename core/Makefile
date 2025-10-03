CC=g++
CFLAGS=-Wall -Wextra -march=native -O3 -std=gnu++17
COM=$(shell git rev-parse HEAD)

all: prune

prune: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -DCOMMIT=\"$(COM)\"

coreGen: collectData.cpp
	$(CC) $(CFLAGS) -fopenmp -o $@ $^

core%: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^ -DCOMMIT=\"$(COM)\"

clean:
	rm *.o