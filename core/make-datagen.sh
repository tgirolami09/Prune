make -B -j DATAGEN=true EXE=datagen-pgoing NUMA=$1 ADDOPTION="-fprofile-instr-generate -DNOTHREAD"
export LLVM_PROFILE_FILE="datagen-%p.profraw"
rm data*.out
./datagen-pgoing /home/jojo/DFRC_4852_v1.epd model.bin 5000 100
llvm-profdata-21 merge -output=datagen.profdata datagen-*.profraw
rm data*.out
make -B -j DATAGEN=true EXE=datagen NUMA=$1 ADDOPTION=-fprofile-instr-use=datagen.profdata