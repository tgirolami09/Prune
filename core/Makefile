CC=g++
ADDOPTION?=
DEBUGOPT?=
CFLAGS=-Werror -Wshadow -Wall -Wextra -std=gnu++17 $(ADDOPTION)
OPT?=-O3 -flto
COM=$(shell git rev-parse HEAD)
ARCH?=native
CPUSET?=unknow
EXE?=
SOURCES := $(wildcard *.cpp)
SOURCES := $(filter-out collectData.cpp,$(SOURCES))
SOURCES := $(filter-out engine.cpp,$(SOURCES))
SOURCES := $(filter-out unittest.cpp,$(SOURCES))
SOURCES := $(filter-out netScale.cpp,$(SOURCES))
SOURCES := $(filter-out spsa.cpp,$(SOURCES))
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTSHCE := $(SOURCES:%.cpp=%HCE.o)
OBJECTSHCE := $(filter-out NNUEHCE.o,$(OBJECTSHCE))
ifeq ($(OS), Windows_NT)
	OS = Windows
	CC=clang++
	CFLAGS := $(CFLAGS) -static
	OPT := $(OPT) -fuse-ld=lld
	LDFLAGS=
else
	OS=$(shell uname -s)
	PROCESSOR_ARCHITECTURE=$(shell uname -m)
	LDFLAGS=-lpthread
endif
CFLAGSALL=$(CFLAGS) $(DEBUGOPT)
ifneq ($(ARCH), unknow)
	ARCHCOM=-march=$(ARCH)
else
	ARCHCOM=
endif
ifneq ($(CPUSET), unknow)
	ARCHCOM := $(ARCHCOM) -m$(CPUSET)
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
EXTENSION=
ifeq ($(EXE), )
	NAME=prune-$(OS)-$(REC)
	SHORTNAME=prune$(REC)
	VERSIONNING=-DCOMMIT=\"$(COM)\" -DVERSION=\"$(REC)\"
	ifeq ($(OS), Windows)
		NAME := $(NAME).exe
		SHORTNAME := $(SHORTNAME).exe
		EXTENSION = .exe
	endif
else
	NAME=$(EXE)
	SHORTNAME=$(EXE)
endif

all: engine.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(NAME) $(VERSIONNING) $^ $(LDFLAGS)

unittest: unittest.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $(VERSIONNING) $^ $(LDFLAGS)

prune: engine.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING) $(LDFLAGS)

netScale: netScale.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o netScale $^ $(VERSIONNING) $(LDFLAGS)

pruneHCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE$(REC) $^ $(VERSIONNING) -DHCE $(LDFLAGS)

HCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE-$(OS)-$(REC)$(EXTENSION) $^ $(VERSIONNING) -DHCE $(LDFLAGS)

coreGen: collectData.cpp $(OBJECTS)
	$(CC) $(CFLAGSALL) $(OPT) $(ARCHCOM) -fopenmp -o $@ $^ $(LDFLAGS)

spsa: spsa.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o spsa $^ $(VERSIONNING) $(LDFLAGS)

spsaHCE: spsa.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o spsaHCE $^ $(VERSIONNING) -DHCE $(LDFLAGS)

%.o: %.cpp
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -c $(VERSIONNING) -o $@ $^

%HCE.o: %.cpp
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -c $(VERSIONNING) -o $@ $^ -DHCE

embeder.o: embeder.cpp
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ $^

embederHCE.o: embeder.cpp
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ $^

clean:
	rm $(OBJECTS) $(DEPENDS)

re:
	$(MAKE) clean
	$(MAKE) all
