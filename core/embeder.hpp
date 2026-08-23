#define BINARY_INCLUDE(buffername)                                             \
    extern "C" {                                                               \
    alignas(64) extern const unsigned char buffername[];                       \
    }

BINARY_INCLUDE(magicsData);
BINARY_INCLUDE(baseModel);
