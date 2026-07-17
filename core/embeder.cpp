#include "embeder.hpp"
#ifndef HCE
alignas(64) const unsigned char baseModel[] = {
#embed "model.bin"
};
#endif
alignas(64) const unsigned char magicsData[] = {
#embed "magics.out"
};
