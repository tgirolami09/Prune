CC=g++
CFLAGS=-Wall -Wextra -std=gnu++17
OPT=-O3
COM=$(shell git rev-parse HEAD)
ARCH?=native
ifeq ($(OS), Windows_NT)
	CC=clang++
endif
ifeq ($(ARCH), native)
	ARCHCOM=-march=$(ARCH)
else 
	ARCHCOM=-m$(ARCH)
endif
CROSS=clang++
CPUSET=avx512f avx2

all: prune

prune: engine.cpp
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS

model.o: model.bin
	objcopy --input-target=binary --output-target=elf64-x86-64 $^ $@

magics.o: magics.out
	objcopy --input-target=binary --output-target=elf64-x86-64 $^ $@

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(OPT) $(ARCHCOM) $(OPT) -fopenmp -o $@ $^

pruneDebug: engine.cpp
	$(CC) -m64 $(ARCHCOM) $(CFLAGS) -g -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS

core%: engine.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) $(OPT) -o $@ $^ -DCOMMIT=\"$(COM)\"

clean:
	rm *.o