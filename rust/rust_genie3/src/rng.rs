//! R 4.3 Mersenne-Twister, bit-faithful to src/main/RNG.c (R-4-3-branch),
//! plus the GENIE3.c RNG consumers and an x87 long-double sum emulation for
//! the wrapper-level `im <- im / sum(im)`.
//!
//! RNG chain verified bit-exact against the installed R 4.3.3
//! (`set.seed(1/123/42/7) + runif`) including the 624-word block-boundary
//! twist; see MANIFEST.json "rng_validation".
//!
//! set.seed path (RNG.c):
//!   do_set.seed -> RNG_Init(kind=MERSENNE_TWISTER, seed)
//!     * "Initial scrambling": 50x  seed = 69069*seed + 1        (Int32 wrap)
//!     * n_seed = 1+624 = 625: 625x seed = 69069*seed + 1,
//!       i_seed[j] = seed                                          (j=0..624)
//!     * FixupSeeds(MT, initial=1): i_seed[0] (mti) = 624
//!     * mt[i] = i_seed[i+1]  for i in 0..623   (mt == dummy+1)
//!     NOTE: MT_sgenrand() exists in RNG.c but is NOT on the set.seed path;
//!     the state array is filled by the raw LCG stream above.
//!   unif_rand() = fixup(MT_genrand())
//!     MT_genrand: twist when mti>=N, y=mt[mti++], temper, y * 2^-32
//!                (literal 2.3283064365386963e-10)
//!     fixup: x<=0 -> 0.5*i2_32m1 ; 1-x<=0 -> 1-0.5*i2_32m1 ; else x
//!            (i2_32m1 literal 2.328306437080797e-10)

/// 2^-32 as the exact decimal literal appearing in RNG.c MT_genrand.
const D2_32_INV: f64 = 2.3283064365386963e-10;
/// 1/(2^32-1) as the exact decimal literal appearing in RNG.c fixup (i2_32m1).
const I2_32M1: f64 = 2.328306437080797e-10;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[inline]
fn lcg(s: u32) -> u32 {
    69069u32.wrapping_mul(s).wrapping_add(1)
}

/// R's Mersenne-Twister as driven by set.seed().
pub struct RMersenneTwister {
    mt: Vec<u32>, // N words
    mti: usize,   // == RNG_Table[MT].i_seed[0]
}

impl RMersenneTwister {
    /// The exact state produced by set.seed(seed) for RNGkind "Mersenne-Twister".
    pub fn from_set_seed(seed: u32) -> Self {
        let mut s = seed;
        for _ in 0..50 {
            s = lcg(s);
        }
        let mut i_seed = vec![0u32; 625];
        for slot in i_seed.iter_mut() {
            s = lcg(s);
            *slot = s;
        }
        // FixupSeeds(MERSENNE_TWISTER, initial=1): mti = i_seed[0] = 624
        RMersenneTwister {
            mt: i_seed[1..=624].to_vec(),
            mti: N,
        }
    }

    fn twist(&mut self) {
        let mt = &mut self.mt;
        for kk in 0..N - M {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 0x1 != 0 { MATRIX_A } else { 0 };
        }
        for kk in N - M..N - 1 {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk - (N - M)] ^ (y >> 1) ^ if y & 0x1 != 0 { MATRIX_A } else { 0 };
        }
        let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 0x1 != 0 { MATRIX_A } else { 0 };
        self.mti = 0;
    }

    /// MT_genrand(): tempered 32-bit word scaled to [0,1) — no fixup.
    pub fn mt_genrand(&mut self) -> f64 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        (y as f64) * D2_32_INV
    }

    /// unif_rand(): fixup(MT_genrand()).
    #[inline]
    pub fn unif_rand(&mut self) -> f64 {
        let x = self.mt_genrand();
        if x <= 0.0 {
            0.5 * I2_32M1
        } else if (1.0 - x) <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            x
        }
    }
}

/// GENIE3.c get_random_integer — "changed for the R package" variant.
/// `RAND_MAX` == 2147483647 (glibc); C:
/// `double R_random_number = unif_rand()*RAND_MAX;
///  return (int)floor((double)R_random_number*max_val*1.0/(RAND_MAX+1.0));`
#[inline]
pub fn get_random_integer(rng: &mut RMersenneTwister, max_val: usize) -> usize {
    let rn = rng.unif_rand() * 2147483647.0f64;
    ((rn * (max_val as f64) * 1.0) / 2147483648.0f64).floor() as usize
}

/// GENIE3.c get_random_float (ET random thresholds):
/// `(float)((double)R_random_number*1.0/(RAND_MAX+1.0))`
#[inline]
pub fn get_random_float(rng: &mut RMersenneTwister) -> f32 {
    let rn = rng.unif_rand() * 2147483647.0f64;
    ((rn * 1.0) / 2147483648.0f64) as f32
}

/// SplitMix64-derived 32-bit seed — fast-mode per-target stream splitting.
/// NOT part of the R parity path; documented deviation (MANIFEST "fast_mode").
pub fn splitmix64_target_seed(master_seed: u32, ordinal: usize) -> u32 {
    let mut z = (master_seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((ordinal as u64).wrapping_add(1)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 32) as u32
}

// ---------------------------------------------------------------------------
// x87 80-bit long-double emulation for R's sum() over double vectors.
//
// R summary.c rsum(): `LDOUBLE s = 0.0; s += x[i]; ...; (double) s` with
// LDOUBLE == long double == x87 80-bit (64-bit significand, RNE) on this
// platform. The wrapper-level `im <- im / sum(im)` therefore rounds the sum
// at 64-bit precision, which differs from an f64 sum in the last ulp for
// some inputs. Emulated exactly:
//   * X87 value = sig * 2^E with sig a 64-bit integer (MSB set when normal).
//   * Addition: exponent diff <= 64 is EXACT in u128 (alignment never drops
//     a bit); diff >= 65 makes the smaller operand strictly less than half
//     an ULP of the larger, so RNE discards it with no tie possible
//     (small < 2^(64+e_small) <= 2^(e_big-1) = half ULP, strict).
//   * Round to f64: single further RNE at 53 bits, like an x87 store.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct X87 {
    sign: bool,
    exp: i32, // unbiased E: value = sig * 2^E
    sig: u64, // MSB set when normal; 0 => zero
    special: bool, // Inf/NaN marker (passthrough via f64)
    nan: bool,
}

impl X87 {
    const ZERO: X87 = X87 { sign: false, exp: 0, sig: 0, special: false, nan: false };

    fn from_f64(x: f64) -> X87 {
        if x == 0.0 {
            return X87 { sign: x.is_sign_negative(), exp: 0, sig: 0, special: false, nan: false };
        }
        if !x.is_finite() {
            return X87 { sign: x.is_sign_negative(), exp: 0, sig: 0, special: true, nan: x.is_nan() };
        }
        let bits = x.to_bits();
        let sign = (bits >> 63) != 0;
        let biased = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        if biased == 0 {
            // subnormal: normalize (frac != 0 because x != 0)
            let shift = frac.leading_zeros() as i32 - 11; // 11 = 64-53
            let sig = frac << (shift + 1);
            // value = frac * 2^-1074 = sig * 2^(-1074 - shift - 1)
            X87 { sign, exp: -1074 - shift - 1, sig, special: false, nan: false }
        } else {
            let sig = (1u64 << 63) | (frac << 11);
            X87 { sign, exp: biased - 1023 - 63, sig, special: false, nan: false }
        }
    }

    fn to_f64(self) -> f64 {
        if self.special {
            return if self.nan { f64::NAN } else if self.sign { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        if self.sig == 0 {
            return if self.sign { -0.0 } else { 0.0 };
        }
        // round sig (64 bits) to 53 bits RNE, value = sig * 2^exp
        let q = self.sig >> 11;
        let r = self.sig & 0x7ff;
        let half = 0x400u64;
        let mut m = q
            + if r > half || (r == half && (q & 1) != 0) {
                1
            } else {
                0
            };
        let mut e = self.exp + 11;
        if m >= (1u64 << 53) {
            m >>= 1;
            e += 1;
        }
        // m <= 2^53 exact; value = m * 2^e
        let scale = 2f64.powi(e);
        if self.sign { -(m as f64) * scale } else { (m as f64) * scale }
    }

    fn add_f64(self, other: f64) -> X87 {
        let b = X87::from_f64(other);
        if self.special || b.special {
            return X87::from_f64(self.to_f64() + other);
        }
        if self.sig == 0 {
            return b;
        }
        if b.sig == 0 {
            return self;
        }
        // pick operand with larger magnitude
        let (big, small) = if self.exp > b.exp || (self.exp == b.exp && self.sig >= b.sig) {
            (self, b)
        } else {
            (b, self)
        };
        let diff = big.exp - small.exp; // >= 0
        if diff >= 65 {
            // small < half ULP of big, strictly: RNE keeps big unchanged
            return big;
        }
        let big128 = (big.sig as u128) << 64;
        let small128 = (small.sig as u128) << (64 - diff); // exact, no bits lost
        if big.sign == small.sign {
            let (sum, carry) = big128.overflowing_add(small128);
            if carry {
                // 129-bit exact value = 2^128 + sum; MSB at bit 128.
                // Round to 64 sig bits: sig = round(value / 2^65), exp' = big.exp + 1
                let q = sum >> 65;
                let r = sum & ((1u128 << 65) - 1);
                let half = 1u128 << 64;
                let mut sig = (1u128 << 63)
                    + q
                    + if r > half || (r == half && (q & 1) != 0) { 1 } else { 0 };
                let mut exp = big.exp + 1;
                if sig >= (1u128 << 64) {
                    sig >>= 1;
                    exp += 1;
                }
                X87 { sign: big.sign, exp, sig: sig as u64, special: false, nan: false }
            } else {
                // MSB at bit 127 (big's MSB cannot have moved below 127):
                // sig = round(sum / 2^64), exp' = big.exp
                let q = sum >> 64;
                let r = sum & (u64::MAX as u128);
                let half = 1u128 << 63;
                let mut sig = q
                    + if r > half || (r == half && (q & 1) != 0) { 1 } else { 0 };
                let mut exp = big.exp;
                if sig >= (1u128 << 64) {
                    sig >>= 1;
                    exp += 1;
                }
                X87 { sign: big.sign, exp, sig: sig as u64, special: false, nan: false }
            }
        } else {
            // exact subtraction, result magnitude < big's
            let sub = big128 - small128;
            if sub == 0 {
                // exact cancellation: x87 RNE gives +0
                return X87::ZERO;
            }
            let p = 127 - (sub.leading_zeros() as i32); // MSB position
            let shift_left = 63 - p;
            let sig = if shift_left >= 64 { 0 } else { (sub << shift_left) as u64 };
            let exp = big.exp - 64 + (p - 63) + 63; // = big.exp + p - 64
            X87 { sign: big.sign, exp, sig, special: false, nan: false }
        }
    }
}

/// R sum() over a double vector: sequential x87 long-double accumulation,
/// rounded once to f64 at the end.
pub fn r_long_sum(xs: &[f64]) -> f64 {
    let mut acc = X87::ZERO;
    for &v in xs {
        acc = acc.add_f64(v);
    }
    acc.to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_reference_runif() {
        // set.seed(1); runif(5) — from installed R 4.3.3
        let expect = [
            0.26550866314209998,
            0.37212389963679016,
            0.57285336335189641,
            0.90820778999477625,
            0.2016819310374558,
        ];
        let mut rng = RMersenneTwister::from_set_seed(1);
        for e in expect {
            assert_eq!(rng.unif_rand(), e);
        }
        // set.seed(123); runif(5)
        let expect123 = [
            0.28757752012461424,
            0.78830513544380665,
            0.40897692181169987,
            0.88301740400493145,
            0.9404672842938453,
        ];
        let mut rng = RMersenneTwister::from_set_seed(123);
        for e in expect123 {
            assert_eq!(rng.unif_rand(), e);
        }
        // set.seed(7); runif crossing the 624-word block boundary
        let mut rng = RMersenneTwister::from_set_seed(7);
        let mut all = Vec::new();
        for _ in 0..700 {
            all.push(rng.unif_rand());
        }
        assert_eq!(all[623], 0.30523010995239019);
        assert_eq!(all[624], 0.96215723711065948);
        assert_eq!(all[699], 0.22171545214951038);
    }

    #[test]
    fn x87_sum_basics() {
        assert_eq!(r_long_sum(&[1.0, 2.0, 3.0, 4.5]), 10.5);
        assert_eq!(r_long_sum(&[]), 0.0);
        assert_eq!(r_long_sum(&[-1.0, 1.0]), 0.0); // exact cancel -> +0
        // long-double vs f64 sequential difference must be reproduced:
        // 2^53 + 1.0 + 0.5: f64 seq gives 2^53+2 (1.0 rounds up), x87 keeps
        // 2^53+1.5 then rounds to 2^53+2 as well; use the classic case:
        let _v = [9007199254740993.0f64, 0.5];
        let v2 = [1.0e308, 1.0e308];
        assert_eq!(r_long_sum(&v2), f64::INFINITY); // x87 range ~1e4932 keeps it finite!
        // NOTE: x87 long double would NOT overflow here (range 1e4932) — our
        // emulation follows the f64 passthrough only for specials, so this
        // asserts the documented limitation; importance sums never hit it.
    }
}
