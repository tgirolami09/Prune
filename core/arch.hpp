#pragma once
const int INPUT_SIZE = 11 * 64;  // merged king planes
const int THREAT_SIZE = 60144;

constexpr int QA = 255;
constexpr int QB = 128;
constexpr int QC = 64;
constexpr int FT_BITS = 9;
constexpr int FT_LSHIFT = 16 - FT_BITS;

static constexpr inline int ilog2c(int n) {
    return (31 ^ __builtin_clz(n)) + !!(n & (n - 1));
}

static constexpr inline int _abs(int x) {
    return x < 0 ? -x : x;
}

constexpr int QA_bits = ilog2c(QA);
constexpr int QB_bits = ilog2c(QB);
constexpr int QC_bits = ilog2c(QC);
constexpr int L1shift = _abs(16 + QC_bits - FT_LSHIFT - QA_bits * 2 - QB_bits);

const int BUCKET = 8;
const int nbInputBuckets = 16;

const int L1 = 640;
const int L2 = 16;
const int L3 = 32;

const int SCALE = 283;
const int inputBuckets[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9,  9,  10, 10, 11, 11,
    12, 12, 13, 13, 12, 12, 13, 13, 14, 14, 15, 15, 14, 14, 15, 15,

};
const int DIVISOR = 32 / BUCKET;