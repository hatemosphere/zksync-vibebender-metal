#pragma once

#include "common.metal"

// Mersenne31 field arithmetic
// Based on https://github.com/Plonky3/Plonky3/tree/main/mersenne-31

namespace airbender {
namespace field {

struct base_field {
    static constant constexpr uint ORDER = (1u << 31) - 1;
    static constant constexpr uint MINUS_ONE = ORDER - 1;
    uint limb;

    static DEVICE_FORCEINLINE base_field zero() { return {0u}; }
    static DEVICE_FORCEINLINE base_field one() { return {1u}; }
    static DEVICE_FORCEINLINE base_field two() { return {2u}; }
    static DEVICE_FORCEINLINE base_field minus_one() { return {MINUS_ONE}; }

    static DEVICE_FORCEINLINE uint into_canonical_u32(const base_field value) {
        return value.limb == ORDER ? 0u : value.limb;
    }

    static DEVICE_FORCEINLINE base_field into_canonical(const base_field value) {
        return {into_canonical_u32(value)};
    }

    static DEVICE_FORCEINLINE base_field from_u32(const uint value) {
        const uint msb = value >> 31;
        const uint lsb = value & ORDER;
        return add({msb}, {lsb});
    }

    static DEVICE_FORCEINLINE base_field from_u32_max_minus_one(const uint value) {
        const uint msb = value >> 31;
        const uint lsb = value & ORDER;
        return {msb + lsb};
    }

    static DEVICE_FORCEINLINE base_field from_u62_max_minus_one(const ulong value) {
        const uint msb = static_cast<uint>(value >> 31);
        const uint lsb = static_cast<uint>(value & ORDER);
        return from_u32_max_minus_one(msb + lsb);
    }

    static DEVICE_FORCEINLINE base_field add(const base_field x, const base_field y) {
        return from_u32_max_minus_one(x.limb + y.limb);
    }

    static DEVICE_FORCEINLINE base_field neg(const base_field x) {
        return {ORDER - x.limb};
    }

    static DEVICE_FORCEINLINE base_field sub(const base_field x, const base_field y) {
        return add(x, neg(y));
    }

    static DEVICE_FORCEINLINE base_field mul(const base_field x, const base_field y) {
        const ulong product = static_cast<ulong>(x.limb) * static_cast<ulong>(y.limb);
        return from_u62_max_minus_one(product);
    }

    static DEVICE_FORCEINLINE base_field sqr(const base_field x) { return mul(x, x); }
    static DEVICE_FORCEINLINE base_field dbl(const base_field x) { return shl(x, 1); }

    static DEVICE_FORCEINLINE base_field shl(const base_field x, const uint shift) {
        const uint hi = (x.limb << shift) & ORDER;
        const uint lo = x.limb >> (31 - shift);
        return {hi | lo};
    }

    static DEVICE_FORCEINLINE base_field shr(const base_field x, const uint shift) {
        const uint hi = (x.limb << (31 - shift)) & ORDER;
        const uint lo = x.limb >> shift;
        return {hi | lo};
    }

    static DEVICE_FORCEINLINE base_field pow(const base_field x, uint power) {
        base_field result = one();
        base_field value = x;
        for (uint i = power;;) {
            if (i & 1u)
                result = mul(result, value);
            i >>= 1;
            if (!i)
                break;
            value = sqr(value);
        }
        return result;
    }

    static DEVICE_FORCEINLINE base_field pow_exp2(const base_field x, const uint log_p) {
        base_field result = x;
        for (uint i = 0; i < log_p; ++i)
            result = sqr(result);
        return result;
    }

    static DEVICE_FORCEINLINE base_field inv(const base_field x) {
        // inv(x) = x^(ORDER - 2) = x^0b1111111111111111111111111111101
        const base_field a = mul(pow_exp2(x, 2), x);   // x^0b101
        const base_field b = mul(sqr(a), a);            // x^0b1111
        const base_field c = mul(pow_exp2(b, 4), b);    // x^0b11111111
        const base_field d = pow_exp2(c, 4);            // x^0b111111110000
        const base_field e = mul(d, b);                 // x^0b111111111111
        const base_field f = mul(pow_exp2(d, 4), c);    // x^0b1111111111111111
        const base_field g = mul(pow_exp2(f, 12), e);   // x^0b1111111111111111111111111111
        const base_field h = mul(pow_exp2(g, 3), a);    // x^0b1111111111111111111111111111101
        return h;
    }
};

struct ext2_field {
    base_field coefficients[2];

    static DEVICE_FORCEINLINE base_field non_residue() { return base_field::minus_one(); }
    static DEVICE_FORCEINLINE ext2_field zero() { return {base_field::zero(), base_field::zero()}; }
    static DEVICE_FORCEINLINE ext2_field one() { return {base_field::one(), base_field::zero()}; }

    DEVICE_FORCEINLINE const thread base_field& base_coefficient_from_flat_idx(const uint idx) const thread {
        return coefficients[idx];
    }

    static DEVICE_FORCEINLINE base_field mul_by_non_residue(const base_field x) {
        return base_field::mul(x, non_residue());
    }

    static DEVICE_FORCEINLINE ext2_field add(const ext2_field x, const base_field y) {
        return {base_field::add(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext2_field add(const base_field x, const ext2_field y) {
        return {base_field::add(x, y.coefficients[0]), y.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext2_field add(const ext2_field x, const ext2_field y) {
        return {base_field::add(x.coefficients[0], y.coefficients[0]),
                base_field::add(x.coefficients[1], y.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext2_field sub(const ext2_field x, const base_field y) {
        return {base_field::sub(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext2_field sub(const base_field x, const ext2_field y) {
        return {base_field::sub(x, y.coefficients[0]), base_field::neg(y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext2_field sub(const ext2_field x, const ext2_field y) {
        return {base_field::sub(x.coefficients[0], y.coefficients[0]),
                base_field::sub(x.coefficients[1], y.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext2_field dbl(const ext2_field x) {
        return {base_field::dbl(x.coefficients[0]), base_field::dbl(x.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext2_field neg(const ext2_field x) {
        return {base_field::neg(x.coefficients[0]), base_field::neg(x.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext2_field mul(const ext2_field x, const base_field y) {
        return {base_field::mul(x.coefficients[0], y), base_field::mul(x.coefficients[1], y)};
    }
    static DEVICE_FORCEINLINE ext2_field mul(const base_field x, const ext2_field y) {
        return {base_field::mul(x, y.coefficients[0]), base_field::mul(x, y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext2_field mul(const ext2_field x, const ext2_field y) {
        const base_field a = base_field::mul(x.coefficients[0], y.coefficients[0]);
        const base_field b = base_field::mul(x.coefficients[1], y.coefficients[0]);
        const base_field c = base_field::mul(x.coefficients[0], y.coefficients[1]);
        const base_field d = base_field::mul(x.coefficients[1], y.coefficients[1]);
        const base_field e = base_field::add(a, mul_by_non_residue(d));
        const base_field f = base_field::add(b, c);
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext2_field sqr(const ext2_field x) {
        const base_field a = base_field::sqr(x.coefficients[0]);
        const base_field b = base_field::mul(x.coefficients[0], x.coefficients[1]);
        const base_field c = base_field::sqr(x.coefficients[1]);
        const base_field e = base_field::add(a, mul_by_non_residue(c));
        const base_field f = base_field::dbl(b);
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext2_field inv(const ext2_field x) {
        const base_field a = x.coefficients[0];
        const base_field b = x.coefficients[1];
        const base_field c = base_field::sub(base_field::sqr(a), mul_by_non_residue(base_field::sqr(b)));
        const base_field d = base_field::inv(c);
        const base_field e = base_field::mul(a, d);
        const base_field f = base_field::neg(base_field::mul(b, d));
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext2_field pow(const ext2_field x, uint power) {
        ext2_field result = one();
        ext2_field value = x;
        for (uint i = power;;) {
            if (i & 1u)
                result = mul(result, value);
            i >>= 1;
            if (!i)
                break;
            value = sqr(value);
        }
        return result;
    }

    static DEVICE_FORCEINLINE ext2_field shr(const ext2_field x, const uint shift) {
        return {base_field::shr(x.coefficients[0], shift), base_field::shr(x.coefficients[1], shift)};
    }
    static DEVICE_FORCEINLINE ext2_field shl(const ext2_field x, const uint shift) {
        return {base_field::shl(x.coefficients[0], shift), base_field::shl(x.coefficients[1], shift)};
    }
};

struct ext4_field {
    ext2_field coefficients[2];

    static DEVICE_FORCEINLINE ext2_field non_residue() {
        return {base_field::two(), base_field::one()};
    }
    static DEVICE_FORCEINLINE ext4_field zero() { return {ext2_field::zero(), ext2_field::zero()}; }
    static DEVICE_FORCEINLINE ext4_field one() { return {ext2_field::one(), ext2_field::zero()}; }

    DEVICE_FORCEINLINE const thread base_field& base_coefficient_from_flat_idx(const uint idx) const thread {
        return coefficients[(idx & 2u) >> 1].coefficients[idx & 1u];
    }

    static DEVICE_FORCEINLINE ext2_field mul_by_non_residue(const ext2_field x) {
        return ext2_field::mul(x, non_residue());
    }

    static DEVICE_FORCEINLINE ext4_field add(const ext4_field x, const base_field y) {
        return {ext2_field::add(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field add(const ext4_field x, const ext2_field y) {
        return {ext2_field::add(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field add(const base_field x, const ext4_field y) {
        return {ext2_field::add(x, y.coefficients[0]), y.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field add(const ext2_field x, const ext4_field y) {
        return {ext2_field::add(x, y.coefficients[0]), y.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field add(const ext4_field x, const ext4_field y) {
        return {ext2_field::add(x.coefficients[0], y.coefficients[0]),
                ext2_field::add(x.coefficients[1], y.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext4_field sub(const ext4_field x, const base_field y) {
        return {ext2_field::sub(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field sub(const ext4_field x, const ext2_field y) {
        return {ext2_field::sub(x.coefficients[0], y), x.coefficients[1]};
    }
    static DEVICE_FORCEINLINE ext4_field sub(const base_field x, const ext4_field y) {
        return {ext2_field::sub(x, y.coefficients[0]), ext2_field::neg(y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext4_field sub(const ext2_field x, const ext4_field y) {
        return {ext2_field::sub(x, y.coefficients[0]), ext2_field::neg(y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext4_field sub(const ext4_field x, const ext4_field y) {
        return {ext2_field::sub(x.coefficients[0], y.coefficients[0]),
                ext2_field::sub(x.coefficients[1], y.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext4_field dbl(const ext4_field x) {
        return {ext2_field::dbl(x.coefficients[0]), ext2_field::dbl(x.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext4_field neg(const ext4_field x) {
        return {ext2_field::neg(x.coefficients[0]), ext2_field::neg(x.coefficients[1])};
    }

    static DEVICE_FORCEINLINE ext4_field mul(const ext4_field x, const base_field y) {
        return {ext2_field::mul(x.coefficients[0], y), ext2_field::mul(x.coefficients[1], y)};
    }
    static DEVICE_FORCEINLINE ext4_field mul(const ext4_field x, const ext2_field y) {
        return {ext2_field::mul(x.coefficients[0], y), ext2_field::mul(x.coefficients[1], y)};
    }
    static DEVICE_FORCEINLINE ext4_field mul(const base_field x, const ext4_field y) {
        return {ext2_field::mul(x, y.coefficients[0]), ext2_field::mul(x, y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext4_field mul(const ext2_field x, const ext4_field y) {
        return {ext2_field::mul(x, y.coefficients[0]), ext2_field::mul(x, y.coefficients[1])};
    }
    static DEVICE_FORCEINLINE ext4_field mul(const ext4_field x, const ext4_field y) {
        const ext2_field a = ext2_field::mul(x.coefficients[0], y.coefficients[0]);
        const ext2_field b = ext2_field::mul(x.coefficients[1], y.coefficients[0]);
        const ext2_field c = ext2_field::mul(x.coefficients[0], y.coefficients[1]);
        const ext2_field d = ext2_field::mul(x.coefficients[1], y.coefficients[1]);
        const ext2_field e = ext2_field::add(a, mul_by_non_residue(d));
        const ext2_field f = ext2_field::add(b, c);
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext4_field sqr(const ext4_field x) {
        const ext2_field a = ext2_field::sqr(x.coefficients[0]);
        const ext2_field b = ext2_field::mul(x.coefficients[0], x.coefficients[1]);
        const ext2_field c = ext2_field::sqr(x.coefficients[1]);
        const ext2_field e = ext2_field::add(a, mul_by_non_residue(c));
        const ext2_field f = ext2_field::dbl(b);
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext4_field inv(const ext4_field x) {
        const ext2_field a = x.coefficients[0];
        const ext2_field b = x.coefficients[1];
        const ext2_field c = ext2_field::sub(ext2_field::sqr(a), mul_by_non_residue(ext2_field::sqr(b)));
        const ext2_field d = ext2_field::inv(c);
        const ext2_field e = ext2_field::mul(a, d);
        const ext2_field f = ext2_field::neg(ext2_field::mul(b, d));
        return {e, f};
    }

    static DEVICE_FORCEINLINE ext4_field pow(const ext4_field x, uint power) {
        ext4_field result = one();
        ext4_field value = x;
        for (uint i = power;;) {
            if (i & 1u)
                result = mul(result, value);
            i >>= 1;
            if (!i)
                break;
            value = sqr(value);
        }
        return result;
    }

    static DEVICE_FORCEINLINE ext4_field shr(const ext4_field x, const uint shift) {
        return {ext2_field::shr(x.coefficients[0], shift), ext2_field::shr(x.coefficients[1], shift)};
    }
    static DEVICE_FORCEINLINE ext4_field shl(const ext4_field x, const uint shift) {
        return {ext2_field::shl(x.coefficients[0], shift), ext2_field::shl(x.coefficients[1], shift)};
    }
};

// Type aliases matching CUDA code
using bf = base_field;
using e2f = ext2_field;
using e4f = ext4_field;

} // namespace field
} // namespace airbender
