#include "sparse.hpp"

void SparseIterator::add_nonzero(simd<8> ft_out0, simd<8> ft_out1){
    // VecU8 = __m256i on AVX2, __m512i on AVX512
    // ALIGNMENT = 32 on AVX2 or 64 on AVX512
    constexpr int32_t regw32 = SIZE / 8 / sizeof(int32_t);
    constexpr int32_t n_mask_bytes = 2 * regw32 / 8;

    uint32_t full_mask = (nonzero_mask(ft_out1) << regw32) | nonzero_mask(ft_out0);
    for (int32_t i = 0; i < n_mask_bytes; i++) {
        // get offset of up to 8 nonzero blocks
        const uint8_t mask = full_mask & 0xFF;
        full_mask >>= 8;
        const auto idxs = simdsmoladd(
            offset,
            loadsimd(&nonzero_idx[mask*8])
        );

        storesimd(&indices[count_], idxs);
        offset = simdsmoladd(offset, simdset(8));
        count_ += popcount(mask);
    }
}

int SparseIterator::count() const{
    return count_;
}

int SparseIterator::index(int nnzidx) const{
    return indices[nnzidx];
}