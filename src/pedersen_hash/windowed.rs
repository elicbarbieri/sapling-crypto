//! Pedersen hash by precomputed window tables, used by [`crate::Node`]'s
//! [`combine_pairs`](incrementalmerkletree::Hashable::combine_pairs)
//!
//! - One 7M mixed addition per [`WINDOW_BITS`] input bits, no scalar-field arithmetic
//! - Projective output, so a batch of hashes shares one inversion

use alloc::{boxed::Box, vec::Vec};
use group::Curve;
use jubjub::{AffineNielsPoint, AffinePoint, ExtendedPoint};
use lazy_static::lazy_static;

use crate::constants::{PEDERSEN_HASH_CHUNKS_PER_GENERATOR, PEDERSEN_HASH_GENERATORS};

/// Input bits per Pedersen hash chunk
const CHUNK_BITS: usize = 3;

/// 512 affine Niels entries per window (48 KiB, ≈1 MiB per generator)
const WINDOW_CHUNKS: usize = 3;

const WINDOW_BITS: usize = CHUNK_BITS * WINDOW_CHUNKS;

const _: () = assert!(
    PEDERSEN_HASH_CHUNKS_PER_GENERATOR.is_multiple_of(WINDOW_CHUNKS),
    "a window never spans two generators"
);

/// One lookup's precomputed sums, indexed by the window's input bits (chunk 0 lowest)
///
/// `partial[r - 1]` sums the first `r` chunks only (absent chunk adds nothing, `enc(000) = 1`)
struct Window {
    full: Box<[AffineNielsPoint]>,
    partial: [Box<[AffineNielsPoint]>; WINDOW_CHUNKS - 1],
}

lazy_static! {
    /// Every generator's windows, in input order
    static ref WINDOWS: Vec<Window> = windows();
}

fn windows() -> Vec<Window> {
    let mut windows = Vec::new();
    for generator in PEDERSEN_HASH_GENERATORS {
        // chunk i of a generator weighs 2^{4i}
        let mut base = ExtendedPoint::from(*generator);
        for _ in 0..PEDERSEN_HASH_CHUNKS_PER_GENERATOR / WINDOW_CHUNKS {
            let bases: [ExtendedPoint; WINDOW_CHUNKS] = core::array::from_fn(|_| {
                let chunk = base;
                base = base.double().double().double().double();
                chunk
            });
            windows.push(Window {
                full: sums(&bases),
                partial: core::array::from_fn(|r| sums(&bases[..=r])),
            });
        }
    }
    windows
}

/// Entry `i` = Σ_k enc(chunk k of `i`) · `bases[k]`
fn sums(bases: &[ExtendedPoint]) -> Box<[AffineNielsPoint]> {
    let digits: Vec<[ExtendedPoint; 1 << CHUNK_BITS]> = bases.iter().map(encodings).collect();
    let points: Vec<ExtendedPoint> = (0..1usize << (CHUNK_BITS * bases.len()))
        .map(|i| {
            digits
                .iter()
                .enumerate()
                .fold(ExtendedPoint::identity(), |sum, (k, digit)| {
                    sum + digit[(i >> (CHUNK_BITS * k)) & ((1 << CHUNK_BITS) - 1)]
                })
        })
        .collect();
    let mut affine = vec![AffinePoint::identity(); points.len()];
    ExtendedPoint::batch_normalize(&points, &mut affine);
    affine.iter().map(AffinePoint::to_niels).collect()
}

/// `enc(abc) · base` for each chunk value `abc` (a = bit 0): `1 + a + 2b`, negated if `c`
fn encodings(base: &ExtendedPoint) -> [ExtendedPoint; 1 << CHUNK_BITS] {
    let multiples = [
        *base,
        base.double(),
        base.double() + base,
        base.double().double(),
    ];
    core::array::from_fn(|abc| match abc & 4 {
        0 => multiples[abc & 3],
        _ => -multiples[abc & 3],
    })
}

/// Hash of the first `len` bits of `words`, personalization included
///
/// - `words` LSB-first, every bit past `len` zero
/// - Panics if `len` exceeds the generators' capacity
pub(crate) fn hash_bits(words: &[u64], len: usize) -> ExtendedPoint {
    let chunks = len.div_ceil(CHUNK_BITS);
    let windows = &WINDOWS[..];
    let used = chunks.div_ceil(WINDOW_CHUNKS);
    assert!(used <= windows.len(), "we don't have enough generators");

    let mut hash = ExtendedPoint::identity();
    for (w, window) in windows[..used].iter().enumerate() {
        let entries = match chunks - w * WINDOW_CHUNKS {
            present if present >= WINDOW_CHUNKS => &window.full,
            present => &window.partial[present - 1],
        };
        hash += &entries[window_bits(words, w * WINDOW_BITS) & (entries.len() - 1)];
    }
    hash
}

/// The [`WINDOW_BITS`] bits of `words` from bit `at`, zero past its end
fn window_bits(words: &[u64], at: usize) -> usize {
    let (word, shift) = (at / 64, at % 64);
    let low = words.get(word).map_or(0, |bits| bits >> shift);
    let high = match shift {
        0 => 0,
        _ => words.get(word + 1).map_or(0, |bits| bits << (64 - shift)),
    };
    ((low | high) & ((1 << WINDOW_BITS) - 1)) as usize
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use group::Curve;
    use rand_core::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use super::{hash_bits, CHUNK_BITS};
    use crate::constants::{PEDERSEN_HASH_CHUNKS_PER_GENERATOR, PEDERSEN_HASH_GENERATORS};
    use crate::pedersen_hash::{pedersen_hash, Personalization};

    fn packed(bits: &[bool]) -> Vec<u64> {
        let mut words = vec![0u64; bits.len().div_ceil(64)];
        for (at, bit) in bits.iter().enumerate() {
            words[at / 64] |= u64::from(*bit) << (at % 64);
        }
        words
    }

    /// Every input length the generators hold, both personalizations: every partial chunk,
    /// partial window and generator boundary
    #[test]
    fn windowed_hash_matches_pedersen_hash_at_every_length() {
        let rng = &mut XorShiftRng::from_seed([7; 16]);
        let capacity =
            PEDERSEN_HASH_GENERATORS.len() * CHUNK_BITS * PEDERSEN_HASH_CHUNKS_PER_GENERATOR;
        let personalization_bits = Personalization::NoteCommitment.get_bits().len();
        for len in 0..=capacity - personalization_bits {
            let bits: Vec<bool> = (0..len).map(|_| rng.next_u32() % 2 == 1).collect();
            for personalization in [
                Personalization::NoteCommitment,
                Personalization::MerkleTree(len % 63),
            ] {
                let input: Vec<bool> = personalization
                    .get_bits()
                    .into_iter()
                    .chain(bits.iter().copied())
                    .collect();
                let expected =
                    jubjub::ExtendedPoint::from(pedersen_hash(personalization, bits.clone()));
                assert_eq!(
                    hash_bits(&packed(&input), input.len()).to_affine(),
                    expected.to_affine(),
                    "{len} bits"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "we don't have enough generators")]
    fn input_past_the_last_generator_panics() {
        let capacity =
            PEDERSEN_HASH_GENERATORS.len() * CHUNK_BITS * PEDERSEN_HASH_CHUNKS_PER_GENERATOR;
        hash_bits(&packed(&vec![true; capacity + 1]), capacity + 1);
    }
}
