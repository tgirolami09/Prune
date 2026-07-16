#include "embeder.hpp"
#ifndef HCE
const unsigned char baseModel[] = {
#embed "model.bin"
};
#endif
const unsigned char magicsData[] = {
#embed "magics.out"
};