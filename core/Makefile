CC=g++
CFLAGS=-Wall -Wextra -march=native -O3

all: main

main: engine.cpp
	$(CC) $(CFLAGS) -o $@ $^

coreGen: collectData.cpp
	$(CC) $(CFLAGS) -fopenmp -o $@ $^

clean:
	rm *.o