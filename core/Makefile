CC=g++
CFLAGS=-Wall -Wextra -O3 -std=gnu++17
COM=$(shell git rev-parse HEAD)
ARCH?=native
ifeq ($(ARCH), native)
	ARCHCOM=-march=$(ARCH)
else 
	ARCHCOM=-m$(ARCH)
endif
CROSS=clang++
OS=unknow-linux pc-linux apple-darwin
ARCHS=x86_64 aarch64 mips64 mips64el
CPUSET=avx512f avx2

all: prune

prune: engine.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\" -DSTOP_POSS

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -fopenmp -o $@ $^

core%: engine.cpp
	$(CC) $(CFLAGS) $(ARCHCOM) -o $@ $^ -DCOMMIT=\"$(COM)\"

release:
	$(foreach os,$(OS),$(foreach arch,$(ARCHS),$(foreach cpuset,$(CPUSET),echo "${$(tput bold)}generating prune-$(os)-$(arch)-$(cpuset)${$(tput normal)}";$(CROSS) $(CFLAGS) -m$(cpuset) -o prune-$(os)-$(arch)-$(cpuset) engine.cpp -DCOMMIT=\"$(COM)\" -DSTOP_POSS -target $(arch)-$(os);)))

clean:
	rm *.o