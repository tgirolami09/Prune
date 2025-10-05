CC=g++
CFLAGS=-Wall -Wextra -O3 -std=gnu++17
COM=$(shell git rev-parse HEAD)
ARCH?=native
ifeq ($(OS), Windows_NT)
	PLUSFILES = model.o magics.o
else
	PLUSFILES =
endif
ifeq ($(ARCH), native)
	ARCHCOM=-march=$(ARCH)
else 
	ARCHCOM=-m$(ARCH)
endif
CROSS=clang++
CPUSET=avx512f avx2

all: prune

prune: engine.cpp $(PLUSFILES)
	$(CC) $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -fopenmp -o $@ $^

core%: engine.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\"

clean:
	rm *.o