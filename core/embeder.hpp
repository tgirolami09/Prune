#define BINARY_INCLUDE(buffername) \
extern "C"{\
    alignas(64) extern const unsigned char buffername[]; \
}

BINARY_INCLUDE(magicsData);
#ifndef HCE
BINARY_INCLUDE(baseModel);
#endif
