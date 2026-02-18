CC=g++
ADDOPTION?=
DEBUGOPT?=
NUMABUILD=off
CFLAGS=-Werror -Wshadow -Wall -Wextra -std=gnu++17 $(ADDOPTION)
INCLUDES=
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
ifeq ($(NUMABUILD), on)
	INCLUDES := $(INCLUDES) -lnuma -DNUMA_BUILD
else
	SOURCES := $(filter-out helper_numa.cpp,$(SOURCES))
endif
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTSHCE := $(SOURCES:%.cpp=%HCE.o)
OBJECTSHCE := $(filter-out NNUEHCE.o,$(OBJECTSHCE))
ifeq ($(OS), Windows_NT)
	OS = Windows
	CC=clang++
	CFLAGS := $(CFLAGS) -static
	OPT := $(OPT) -fuse-ld=lld
else
	OS=$(shell uname -s)
	PROCESSOR_ARCHITECTURE=$(shell uname -m)
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
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(NAME) $(VERSIONNING) $^ $(INCLUDES)

unittest: unittest.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $(VERSIONNING) $^ $(INCLUDES)

prune: engine.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING) $(INCLUDES)

netScale: netScale.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o netScale $^ $(VERSIONNING) $(INCLUDES)

pruneHCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE$(REC) $^ $(VERSIONNING) -DHCE $(INCLUDES)

HCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE-$(OS)-$(REC)$(EXTENSION) $^ $(VERSIONNING) -DHCE $(INCLUDES)

coreGen: collectData.cpp $(OBJECTS)
	$(CC) $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $^ $(INCLUDES) -fopenmp

spsa: spsa.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o spsa $^ $(VERSIONNING) $(INCLUDES)

%.o: %.cpp
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -c $(VERSIONNING) -o $@ $^ $(INCLUDES)

%HCE.o: %.cpp
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -c $(VERSIONNING) -o $@ $^ -DHCE $(INCLUDES)

embeder.o: embeder.cpp
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ $^ $(INCLUDES)

clean:
	rm $(OBJECTS) $(DEPENDS)

re:
	$(MAKE) clean
	$(MAKE) all
