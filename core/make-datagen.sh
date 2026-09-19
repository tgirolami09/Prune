make -B -j DATAGEN=true EXE=datagen-pgoing NUMA=$2 ADDOPTION="-fprofile-instr-generate -DNOTHREAD"
export LLVM_PROFILE_FILE="datagen-%p.profraw"
rm -f data*.out data*.outpositions
./datagen-pgoing $1 model.bin 5000 100
llvm-profdata-21 merge -output=datagen.profdata datagen-*.profraw
rm -f data*.out data*.outpositions
make -B -j DATAGEN=true EXE=datagen NUMA=$2 ADDOPTION="-fprofile-instr-use=datagen.profdata -Wno-profile-instr-out-of-date"
