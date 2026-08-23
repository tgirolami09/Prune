// code from https://rmeguro.com/blogs/sparse-nnue.html
#include <array>
#include "arch.hpp"
#include "simd_definitions.hpp"
using namespace std;
#ifdef __ARM_NEON__
using simdsmol = uint16x8_t;
#else
using simdsmol = __m128i;
#endif
static inline simdsmol loadsimd(const uint16_t* pointer) {
#ifdef __ARM_NEON__
    return vld1q_u16(pointer);
#else
    return _mm_load_si128(reinterpret_cast<const __m128i*>(pointer));
#endif
}
static inline void storesimd(uint16_t* pointer, simdsmol vec) {
#ifdef __ARM_NEON__
    vst1q_u16(pointer, vec);
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pointer), vec);
#endif
}
static inline simdsmol simdsmoladd(simdsmol a, simdsmol b) {
#ifdef __ARM_NEON__
    return vaddq_u16(a, b);
#else
    return _mm_add_epi16(a, b);
#endif
}

static inline simdsmol simdset(uint16_t x) {
#ifdef __ARM_NEON__
    return vdupq_n_u16(x);
#else
    return _mm_set1_epi16(x);
#endif
}

struct SparseIterator {
    alignas(16) uint16_t indices[L1 / 4] = {0};
    int count_ = 0;
    simdsmol offset = simdset(0);
    alignas(16) static constexpr array<uint16_t, 256 * 8> nonzero_idx = [] {
        array<uint16_t, 256 * 8> idx{};

        for (int32_t i = 0; i < 256; i++) {
            int32_t nnz = 0;
            for (uint8_t mask = i; mask != 0; mask &= mask - 1)
                idx[i * 8 + (nnz++)] = countr_zero(mask);
        }

        return idx;
    }();

   public:
    int count() const;
    int index(int nnzidx) const;
    void add_nonzero(simd<8> ft_out0, simd<8> ft_out1);
};