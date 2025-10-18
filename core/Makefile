CC=g++
CFLAGS=-Wall -Wextra -O3 -std=gnu++17
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
	$(CC) -m64 $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS -I/usr/local/include -L/usr/local/lib -ltensorflow

model.o: model.bin
	objcopy --input-target=binary --output-target=elf64-x86-64 $^ $@

magics.o: magics.out
	objcopy --input-target=binary --output-target=elf64-x86-64 $^ $@

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -fopenmp -o $@ $^

pruneDebug: engine.cpp
	$(CC) -m64 $(ARCHCOM) -g -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS

core%: engine.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\"

clean:
	rm *.o