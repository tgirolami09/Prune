CC=g++
CFLAGS=-Wall -Wextra -std=gnu++17
OPT=-O3
COM=$(shell git rev-parse HEAD)
ARCH?=native
CPUSET?=unknow
ifeq ($(OS), Windows_NT)
	CC=clang++
else
	OS=$(shell uname -s)
	PROCESSOR_ARCHITECTURE=$(shell uname -m)
endif
ARCHCOM=-march=$(ARCH)
ifneq ($(CPUSET), unknow)
	ARCHCOM=$(ARCHCOM) -m$(ARCH)
endif
TAG=$(shell git tag --contains=HEAD)
ifeq ($(TAG), )
	REC=$(strip $(shell git rev-parse --short HEAD) )
else
	REC=$(TAG)
endif
ifneq ($(shell git status --porcelain),)
	REC=test
endif
ifeq ($(ARCH), native)
	ARCHNAME=
else
	ARCHNAME=-$(ARCH)
endif
ifeq ($(CPUSET), unknow)
	CPUSETNAME=
else
	CPUSETNAME=-$(CPUSET)
endif
NAME=prune-$(OS)-$(REC)
SHORTNAME=prune$(REC)
VERSIONNING=-DCOMMIT=\"$(COM)\" -DVERSION=\"$(REC)\"
ifeq ($(OS), Windows_NT)
	NAME := $(NAME).exe
	SHORTNAME := $(SHORTNAME).exe
endif

all:
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -o $(NAME) engine.cpp $(VERSIONNING)

prune: engine.cpp
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING)

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(OPT) $(ARCHCOM) $(OPT) -fopenmp -o $@ $^

pruneDebug: engine.cpp
	$(CC) -m64 $(ARCHCOM) $(CFLAGS) -g -o $@ $^ $(VERSIONNING)

clean:
	rm *.o