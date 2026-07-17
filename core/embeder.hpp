#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
}

BINARY_INCLUDE(magicsData);
#ifndef HCE
BINARY_INCLUDE(baseModel);
#endif