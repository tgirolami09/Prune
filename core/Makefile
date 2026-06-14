CC=g++
ADDOPTION?=
DEBUGOPT?=
NETNAME?=
CFLAGS=-Werror -Wshadow -Wall -Wextra -MMD -std=gnu++17 -I../Fathom/src $(ADDOPTION)
OPT?=-O3 -flto -funroll-loops
SHELL := /bin/bash
# Fathom tablebase library
FATHOM_DIR = ../Fathom/src
FATHOM_OBJ = tbprobe.o
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
SOURCES := $(filter-out datasetStats.cpp,$(SOURCES))
SOURCES := $(filter-out testTI.cpp,$(SOURCES))
SOURCES := $(filter-out quantise.cpp,$(SOURCES))
SOURCES := $(filter-out permute.cpp,$(SOURCES))
OBJECTS := $(SOURCES:%.cpp=%.o) $(FATHOM_OBJ)
DEPENDS := $(OBJECTS:%.o=%.d)
OBJECTSHCE := $(SOURCES:%.cpp=%HCE.o) $(FATHOM_OBJ)
OBJECTSTUNE := $(SOURCES:%.cpp=%TUNE.o) $(FATHOM_OBJ)
OBJECTSHCE := $(filter-out NNUEHCE.o,$(OBJECTSHCE))
ifeq ($(OS), Windows_NT)
	OS = Windows
	CC=clang++
	CFLAGS := $(CFLAGS) -static
	OPT := $(OPT) -fuse-ld=lld
	myLDFLAGS=
else
	OS=$(shell uname -s)
	PROCESSOR_ARCHITECTURE=$(shell uname -m)
	myLDFLAGS=-lpthread
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
VERSIONNING=-DCOMMIT=\"$(COM)\" -DVERSION=\"$(REC)\"
ifeq ($(EXE), )
	NAME=prune-$(OS)-$(REC)
	SHORTNAME=prune$(REC)
	ifeq ($(OS), Windows)
		NAME := $(NAME).exe
		SHORTNAME := $(SHORTNAME).exe
		EXTENSION = .exe
	endif
else
	NAME=$(EXE)
	SHORTNAME=$(EXE)
endif
ifeq ($(OS),Windows)
    NULL_DEVICE = nul
else
    NULL_DEVICE = /dev/null
endif
ifeq ($(NETNAME), )
	NEEDINGFILE=download
	NETNAME=$(shell tail -1 nn-name.txt)
else
	NEEDINGFILE=
endif

override CPPFLAGS += $(CFLAGSALL) $(OPT) $(ARCHCOM) $(ADDOPTION)
override LDFLAGS += $(myLDFLAGS)

all: engine.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(NAME) $(VERSIONNING) $^ $(LDFLAGS)

download:
	@if [ ! -f model.bin ]; then \
		echo "File missing, downloading..."; \
		curl -OL "https://github.com/tgirolami09/Prune-nets/releases/download/$(shell head -1 nn-name.txt)/$(NETNAME).bin"; \
	elif [ "$$(sha256sum $(NETNAME).bin | cut -c1-7)" != "$$(head -1 nn-name.txt | grep -P ".{7}$$" -o)" ]; then \
		echo "SHA mismatch, re-downloading..."; \
		curl -OL "https://github.com/tgirolami09/Prune-nets/releases/download/$(shell head -1 nn-name.txt)/$(NETNAME).bin"; \
	fi

permute: permute.cpp
	$(CC) -m64 $(OPT) $(ARCHCOM) -o permute $(VERSIONNING) permute.cpp $(LDFLAGS)

$(NETNAME): $(NEEDINGFILE)

model.bin: $(NETNAME) permute
	./permute $(NETNAME).bin model.bin

unittest: unittest.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $(VERSIONNING) $^ $(LDFLAGS)

prune: engine.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING) $(LDFLAGS)

-include $(DEPENDS)

netScale: netScale.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o netScale $^ $(VERSIONNING) $(LDFLAGS)

datastat: datasetStats.cpp $(OBJECTS)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $^ $(VERSIONNING) $(LDFLAGS)

pruneHCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE$(REC) $^ $(VERSIONNING) -DHCE $(LDFLAGS)

HCE: engine.cpp $(OBJECTSHCE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o pruneHCE-$(OS)-$(REC)$(EXTENSION) $^ $(VERSIONNING) -DHCE $(LDFLAGS)

coreGen: collectData.cpp $(OBJECTS)
	$(CC) $(CFLAGSALL) $(OPT) $(ARCHCOM) -fopenmp -o $@ $^ $(LDFLAGS)

spsa: spsa.cpp $(OBJECTSTUNE)
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o spsa $^ $(VERSIONNING) $(LDFLAGS)

embeder.o: embeder.cpp model.bin
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ embeder.cpp

embederTUNE.o: embeder.cpp model.bin
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ embeder.cpp

embederHCE.o: embeder.cpp
	$(CC) -m64 $(CFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ $^

testTI: $(OBJECTS) testTI.cpp
	$(CC) -m64 $(CFLAGSALL) $(OPT) $(ARCHCOM) -o $@ $(VERSIONNING) $^ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(DEPENDS)

re:
	$(MAKE) clean
	$(MAKE) all

# Compile Fathom tablebase library (C code)
$(FATHOM_OBJ): $(FATHOM_DIR)/tbprobe.c
	$(CC) -c -O3 -std=c++17 $(ARCHCOM) -D_CRT_SECURE_NO_WARNINGS -I$(FATHOM_DIR) -o $@ $<
