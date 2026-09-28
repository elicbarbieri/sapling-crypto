use bitvec::{order::Lsb0, view::AsBits};
use group::{ff::PrimeField, Curve};
use incrementalmerkletree::{Hashable, Level};
use lazy_static::lazy_static;
use subtle::CtOption;

use alloc::vec::Vec;
use core::fmt;

use super::{
    note::ExtractedNoteCommitment,
    pedersen_hash::{pedersen_hash, windowed, Personalization},
};

pub const NOTE_COMMITMENT_TREE_DEPTH: u8 = 32;
pub type CommitmentTree =
    incrementalmerkletree::frontier::CommitmentTree<Node, NOTE_COMMITMENT_TREE_DEPTH>;
pub type IncrementalWitness =
    incrementalmerkletree::witness::IncrementalWitness<Node, NOTE_COMMITMENT_TREE_DEPTH>;
pub type MerklePath = incrementalmerkletree::MerklePath<Node, NOTE_COMMITMENT_TREE_DEPTH>;

lazy_static! {
    static ref UNCOMMITTED_SAPLING: bls12_381::Scalar = bls12_381::Scalar::one();
    static ref EMPTY_ROOTS: Vec<Node> = empty_roots();
}

fn empty_roots() -> Vec<Node> {
    let mut v = vec![Node::empty_leaf()];
    for d in 0..NOTE_COMMITMENT_TREE_DEPTH {
        let next = Node::combine(d.into(), &v[usize::from(d)], &v[usize::from(d)]);
        v.push(next);
    }
    v
}

/// Compute a parent node in the Sapling commitment tree given its two children.
pub fn merkle_hash(depth: usize, lhs: &[u8; 32], rhs: &[u8; 32]) -> [u8; 32] {
    merkle_hash_field(depth, lhs, rhs).to_repr()
}

fn merkle_hash_field(depth: usize, lhs: &[u8; 32], rhs: &[u8; 32]) -> jubjub::Base {
    let lhs = {
        let mut tmp = [false; 256];
        for (a, b) in tmp.iter_mut().zip(lhs.as_bits::<Lsb0>()) {
            *a = *b;
        }
        tmp
    };

    let rhs = {
        let mut tmp = [false; 256];
        for (a, b) in tmp.iter_mut().zip(rhs.as_bits::<Lsb0>()) {
            *a = *b;
        }
        tmp
    };

    jubjub::ExtendedPoint::from(pedersen_hash(
        Personalization::MerkleTree(depth),
        lhs.iter()
            .copied()
            .take(bls12_381::Scalar::NUM_BITS as usize)
            .chain(
                rhs.iter()
                    .copied()
                    .take(bls12_381::Scalar::NUM_BITS as usize),
            ),
    ))
    .to_affine()
    .get_u()
}

/// Length of [`Personalization::get_bits`]
const PERSONALIZATION_BITS: usize = 6;

/// Bits of each child a Merkle hash takes
const CHILD_BITS: usize = bls12_381::Scalar::NUM_BITS as usize;

/// Personalization ‖ lhs ‖ rhs
const MERKLE_INPUT_BITS: usize = PERSONALIZATION_BITS + 2 * CHILD_BITS;

/// Pairs per rayon task (one inversion amortised over 32 ≈ 9 µs hashes)
#[cfg(feature = "multicore")]
const PAIRS_PER_TASK: usize = 32;

/// One task's parents: every hash projective, then one batched inversion
fn parents(depth: u8, children: &[Node]) -> Vec<Node> {
    let points: Vec<jubjub::ExtendedPoint> = children
        .chunks_exact(2)
        .map(|pair| merkle_hash_point(depth, &pair[0].0, &pair[1].0))
        .collect();
    let mut affine = vec![jubjub::AffinePoint::identity(); points.len()];
    jubjub::ExtendedPoint::batch_normalize(&points, &mut affine);
    affine.iter().map(|parent| Node(parent.get_u())).collect()
}

/// [`merkle_hash_field`] by the windowed tables, still projective
fn merkle_hash_point(depth: u8, lhs: &jubjub::Base, rhs: &jubjub::Base) -> jubjub::ExtendedPoint {
    // all-ones = `Personalization::NoteCommitment`
    assert!(
        usize::from(depth) < (1 << PERSONALIZATION_BITS) - 1,
        "depth within MerkleTree(_)"
    );
    let mut words = [0u64; MERKLE_INPUT_BITS.div_ceil(64)];
    words[0] = u64::from(depth);
    place_child(&mut words, lhs, PERSONALIZATION_BITS);
    place_child(&mut words, rhs, PERSONALIZATION_BITS + CHILD_BITS);
    windowed::hash_bits(&words, MERKLE_INPUT_BITS)
}

/// ORs `child`'s little-endian bits into `words` from bit `at` (canonical, so under [`CHILD_BITS`])
fn place_child(words: &mut [u64], child: &jubjub::Base, at: usize) {
    for (i, limb) in child.to_bytes().chunks_exact(8).enumerate() {
        let limb = u64::from_le_bytes(limb.try_into().expect("8-byte limb"));
        let (word, shift) = ((at + 64 * i) / 64, (at + 64 * i) % 64);
        words[word] |= limb << shift;
        if shift > 0 {
            words[word + 1] |= limb >> (64 - shift);
        }
    }
}

/// The root of a Sapling commitment tree.
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub struct Anchor(jubjub::Base);

impl From<jubjub::Base> for Anchor {
    fn from(anchor_field: jubjub::Base) -> Anchor {
        Anchor(anchor_field)
    }
}

impl From<Node> for Anchor {
    fn from(anchor: Node) -> Anchor {
        Anchor(anchor.0)
    }
}

impl Anchor {
    /// The anchor of the empty Sapling note commitment tree.
    ///
    /// This anchor does not correspond to any valid anchor for a spend, so it
    /// may only be used for coinbase bundles or in circumstances where Sapling
    /// functionality is not active.
    pub fn empty_tree() -> Anchor {
        Anchor(Node::empty_root(NOTE_COMMITMENT_TREE_DEPTH.into()).0)
    }

    pub(crate) fn inner(&self) -> jubjub::Base {
        self.0
    }

    /// Parses a Sapling anchor from a byte encoding.
    pub fn from_bytes(bytes: [u8; 32]) -> CtOption<Anchor> {
        jubjub::Base::from_repr(bytes).map(Self)
    }

    /// Returns the byte encoding of this anchor.
    pub fn to_bytes(self) -> [u8; 32] {
        self.0.to_repr()
    }
}

/// A node within the Sapling commitment tree.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Node(jubjub::Base);

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("repr", &hex::encode(self.0.to_bytes()))
            .finish()
    }
}

impl Node {
    /// Creates a tree leaf from the given Sapling note commitment.
    pub fn from_cmu(value: &ExtractedNoteCommitment) -> Self {
        Node(value.inner())
    }

    /// Constructs a new note commitment tree node from a [`bls12_381::Scalar`]
    pub fn from_scalar(cmu: bls12_381::Scalar) -> Self {
        Self(cmu)
    }

    /// Parses a tree leaf from the bytes of a Sapling note commitment.
    ///
    /// Returns `None` if the provided bytes represent a non-canonical encoding.
    pub fn from_bytes(bytes: [u8; 32]) -> CtOption<Self> {
        jubjub::Base::from_repr(bytes).map(Self)
    }

    /// Returns the canonical byte representation of this node.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_repr()
    }

    /// Returns the wrapped value
    #[cfg(feature = "circuit")]
    pub(crate) fn inner(&self) -> &jubjub::Base {
        &self.0
    }
}

impl Hashable for Node {
    fn empty_leaf() -> Self {
        Node(*UNCOMMITTED_SAPLING)
    }

    fn combine(level: Level, lhs: &Self, rhs: &Self) -> Self {
        Node(merkle_hash_field(
            level.into(),
            &lhs.0.to_bytes(),
            &rhs.0.to_bytes(),
        ))
    }

    /// - Hashes by precomputed 9-bit window tables, one field inversion per batch
    /// - `multicore`: wide levels split across rayon tasks (one inversion each)
    /// - Panics if `children` has an odd length or `level` >= 63 (same as `combine`)
    fn combine_pairs(level: Level, children: &[Self]) -> Vec<Self> {
        assert!(children.len().is_multiple_of(2), "children pair up");
        let depth = u8::from(level);
        // below one task, no pool handoff (a lone pair = the root's serial chain)
        #[cfg(feature = "multicore")]
        if children.len() > 2 * PAIRS_PER_TASK {
            use rayon::prelude::*;
            return children
                .par_chunks(2 * PAIRS_PER_TASK)
                .flat_map_iter(|chunk| parents(depth, chunk))
                .collect();
        }
        parents(depth, children)
    }

    fn empty_root(level: Level) -> Self {
        EMPTY_ROOTS[<usize>::from(level)]
    }
}

impl From<Node> for bls12_381::Scalar {
    fn from(node: Node) -> Self {
        node.0
    }
}

#[cfg(any(test, feature = "test-dependencies"))]
pub(super) mod testing {
    use ff::Field;
    use proptest::prelude::*;
    use rand::{
        distr::{Distribution, StandardUniform},
        Rng,
    };

    use super::Node;
    use crate::note::testing::arb_cmu;

    prop_compose! {
        pub fn arb_node()(cmu in arb_cmu()) -> Node {
            Node::from_cmu(&cmu)
        }
    }

    impl Node {
        /// Return a random fake `MerkleHashOrchard`.
        pub fn random(rng: &mut impl Rng) -> Self {
            StandardUniform.sample(rng)
        }
    }

    impl Distribution<Node> for StandardUniform {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Node {
            Node::from_scalar(bls12_381::Scalar::random(rng))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use incrementalmerkletree::{Hashable, Level};
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use super::{Anchor, Node, NOTE_COMMITMENT_TREE_DEPTH};

    #[test]
    fn combine_pairs_equals_combine_per_pair() {
        let rng = &mut XorShiftRng::from_seed([5; 16]);
        for (level, pairs) in [(0u8, 0usize), (0, 1), (7, 3), (15, 64), (31, 33)] {
            let level = Level::from(level);
            let children: Vec<Node> = (0..2 * pairs).map(|_| Node::random(rng)).collect();
            let expected: Vec<Node> = children
                .chunks_exact(2)
                .map(|pair| Node::combine(level, &pair[0], &pair[1]))
                .collect();
            assert_eq!(
                Node::combine_pairs(level, &children),
                expected,
                "{level:?}, {pairs} pairs"
            );
        }
    }

    #[test]
    #[should_panic(expected = "children pair up")]
    fn combine_pairs_refuses_an_odd_count() {
        Node::combine_pairs(Level::from(0), &[Node::empty_leaf()]);
    }

    #[test]
    #[should_panic(expected = "depth within MerkleTree(_)")]
    fn combine_pairs_refuses_the_note_commitment_personalization() {
        Node::combine_pairs(Level::from(63), &[Node::empty_leaf(), Node::empty_leaf()]);
    }

    /// Published Sapling empty root (zcashd's `3e49b5f9…c2fb`, byte-reversed here)
    #[test]
    fn combine_pairs_rebuilds_the_published_empty_root() {
        let mut node = Node::empty_leaf();
        for level in 0..NOTE_COMMITMENT_TREE_DEPTH {
            node = Node::combine_pairs(Level::from(level), &[node, node])[0];
        }
        let root = Anchor::from(node).to_bytes();
        assert_eq!(
            hex::encode(root),
            "fbc2f4300c01f0b7820d00e3347c8da4ee614674376cbc45359daa54f9b5493e"
        );
    }
}
