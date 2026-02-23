#ifdef __unix__
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section .rodata\n" \
    ".global " #buffername "\n" \
    ".type " #buffername ", @object\n" \
    ".align 4\n" \
    #buffername":\n" \
    ".incbin " #filename "\n" \
    #buffername"_end:\n" \
    ".global "#buffername"_size\n" \
    ".type "#buffername"_size, @object\n" \
    ".align 4\n" \
    #buffername"_size:\n" \
    ".int "#buffername"_end - "#buffername"\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#elif defined(__APPLE__)
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section __TEXT,__const\n" \
    ".globl _" #buffername "\n" \
    ".align 4\n" \
    "_" #buffername":\n" \
    ".incbin " #filename "\n" \
    "_" #buffername"_end:\n" \
    ".globl _" #buffername"_size\n" \
    ".align 4\n" \
    "_" #buffername"_size:\n" \
    ".long _" #buffername"_end - _" #buffername "\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#else
#define BINARY_ASM_INCLUDE(filename, buffername) \
    __asm__(".section .rdata\n" \
    ".global " #buffername "\n" \
    ".align 4\n" \
    "" #buffername":\n" \
    ".incbin " #filename "\n" \
    "" #buffername"_end:\n" \
    ".globl " #buffername"_size\n" \
    ".align 4\n" \
    "" #buffername"_size:\n" \
    ".long " #buffername"_end - " #buffername "\n"\
    );
#define BINARY_INCLUDE(buffername) \
extern "C"{\
    extern const unsigned char buffername[]; \
    extern const unsigned char* buffername##_end; \
    extern const int buffername##_size; \
}
#endif

BINARY_INCLUDE(magicsData);
#ifndef HCE
BINARY_INCLUDE(baseModel);
#endif