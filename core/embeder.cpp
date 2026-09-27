#include "embeder.hpp"
#ifndef HCE
alignas(64) constexpr unsigned char baseModel[] = {
#embed "model.bin"
};
#endif
alignas(64) constexpr unsigned char magicsData[] = {
#embed "magics.out"
};
