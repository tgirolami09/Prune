ADDOPTION?=
DEBUGOPT?=
NETNAME?=
myCPPFLAGS=-Werror -Wshadow -Wall -Wextra -MMD -std=gnu++17 -I../Fathom/src $(ADDOPTION)
OPT?=-O3 -flto -funroll-loops

TUNE?=false
HCE?=false
DATAGEN?=false
UNITTEST?=false

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

MAIN?=engine.cpp
MAINFLAGS=
THENET=model.bin
BASENAME=prune
ADDSUFFIX=true
NOTMAIN=
ifneq ($(MAIN), engine.cpp)
	BASENAME=$(MAIN:%.cpp=%)
	ADDSUFFIX=false
endif
ifeq ($(MAIN), collectData.cpp)
	BASENAME=coreGen
	MAINFLAGS := -fopenmp
	ADDSUFFIX=false
endif
ifeq ($(HCE), true)
	OBJECTS := $(filter-out NNUE.o,$(OBJECTS))
	myCPPFLAGS := $(myCPPFLAGS) -DHCE
	THENET :=
	BASENAME := pruneHCE
endif
ifeq ($(TUNE), true)
	NOTMAIN := $(NOTMAIN) -DTUNE
	BASENAME := spsa
	MAIN=spsa.cpp
	ADDSUFFIX=false
endif

myLDFLAGS=-lm
ifeq ($(OS), Windows_NT)
	OS = Windows
	CFLAGS := $(CFLAGS) -static
	OPT := $(OPT) -fuse-ld=lld
else
	OS=$(shell uname -s)
	PROCESSOR_ARCHITECTURE=$(shell uname -m)
	myLDFLAGS := $(myLDFLAGS) -lpthread
endif
myCPPFLAGS := $(myCPPFLAGS) $(DEBUGOPT)
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
EXTENSION=
VERSIONNING=-DCOMMIT=\"$(COM)\" -DVERSION=\"$(REC)\"
ifeq ($(EXE), )
	ifeq ($(ADDSUFFIX), true)
		NAME=$(BASENAME)-$(OS)-$(REC)
		SHORTNAME=$(BASENAME)$(REC)
	else
		NAME=$(BASENAME)
		SHORTNAME=$(BASENAME)
	endif
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

override CPPFLAGS += $(myCPPFLAGS) $(NOTMAIN) $(OPT) $(ARCHCOM) $(ADDOPTION)
override LDFLAGS += $(myLDFLAGS)

all: $(MAIN) $(OBJECTS)
	$(CXX) -m64 $(myCPPFLAGS) $(OPT) $(ARCHCOM) -o $(NAME) $(VERSIONNING) $^ $(LDFLAGS) $(MAINFLAGS)

download:
	@if [ ! -f model.bin ]; then \
		echo "File missing, downloading..."; \
		curl -OL "https://github.com/tgirolami09/Prune-nets/releases/download/$(shell head -1 nn-name.txt)/$(NETNAME).bin"; \
	elif [ "$$(sha256sum $(NETNAME).bin | cut -c1-7)" != "$$(head -1 nn-name.txt | grep -P ".{7}$$" -o)" ]; then \
		echo "SHA mismatch, re-downloading..."; \
		curl -OL "https://github.com/tgirolami09/Prune-nets/releases/download/$(shell head -1 nn-name.txt)/$(NETNAME).bin"; \
	fi

permute: permute.cpp
	$(CXX) -m64 $(OPT) $(ARCHCOM) $(myCPPFLAGS) -o permute $(VERSIONNING) permute.cpp $(LDFLAGS)

$(NETNAME): $(NEEDINGFILE)

model.bin: $(NETNAME) permute
	./permute $(NETNAME).bin model.bin

prune: $(MAIN) $(OBJECTS)
	$(CXX) -m64 $(myCPPFLAGS) $(OPT) $(ARCHCOM) -o $(SHORTNAME) $^ $(VERSIONNING) $(LDFLAGS) $(MAINFLAGS)

-include $(DEPENDS)

embeder.o: embeder.cpp $(THENET)
	$(CXX) -m64 $(myCPPFLAGS) -O3 $(ARCHCOM) -c $(VERSIONNING) -o $@ embeder.cpp

clean:
	rm -f $(OBJECTS) $(DEPENDS)

re:
	$(MAKE) clean
	$(MAKE) all

# Compile Fathom tablebase library (C code)
$(FATHOM_OBJ): $(FATHOM_DIR)/tbprobe.c
	$(CC) -c -O3 $(ARCHCOM) -D_CRT_SECURE_NO_WARNINGS -I$(FATHOM_DIR) -o $@ $<
