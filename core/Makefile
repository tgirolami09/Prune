CC=g++
ADDOPTION?=
CFLAGS=-Wall -Wextra -std=gnu++17 -static $(ADDOPTION)
OPT=-O3 -flto
COM=$(shell git rev-parse HEAD)
ARCH?=native
CPUSET?=unknow
SOURCES := $(wildcard *.cpp)
SOURCES := $(filter-out collectData.cpp,$(SOURCES))
OBJECTS := $(SOURCES:%.cpp=%.o)
DEPENDS := $(SOURCES:%.cpp=%.d)
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

all: $(OBJECTS)
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -o $(NAME) $(VERSIONNING) $^

prune: $(OBJECTS)
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING)

coreGen: collectData.cpp
	$(CC) $(CFLAGS) $(OPT) $(ARCHCOM) $(OPT) -fopenmp -o $@ $^

$(OBJECTS) : %.o: %.cpp
	$(CC) -m64 $(CFLAGS) $(OPT) $(ARCHCOM) -c $(VERSIONNING) -o $@ $^

pruneDebug: $(OBJECTS)
	$(CC) -m64 $(ARCHCOM) $(CFLAGS) -g -o $@ $^ $(VERSIONNING)

clean:
	rm $(OBJECTS) $(DEPENDS)

re:
	$(MAKE) clean
	$(MAKE) all