CC=g++
CFLAGS=-Wall -Wextra -march=native -O3

all: main

main: engine.cpp magics.o model.o
	$(CC) $(CFLAGS) -o $@ $^

magics.o:
	ld -r -b binary magics.out -o magics.o

model.o:
	ld -r -b binary ../nnue/model.bin -o model.o

coreGen: collectData.cpp magics.o model.o
	$(CC) $(CFLAGS) -fopenmp -o $@ $^

clean:
	rm *.o