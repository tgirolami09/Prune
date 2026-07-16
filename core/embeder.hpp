#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}

BINARY_INCLUDE(magicsData);
#ifndef HCE
BINARY_INCLUDE(baseModel);
#endif