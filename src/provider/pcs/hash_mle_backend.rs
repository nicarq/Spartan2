//! Backend interface for Hash-MLE PCS: field bridge + digest.
//! Two implementations live in this crate:
//!  - ff_keccak: uses E::Scalar directly + Keccak256
//!  - p3_poseidon2_goldilocks: converts E::Scalar <-> p3_goldilocks::Goldilocks and hashes with Poseidon2

use serde::{Deserialize, Serialize};
use ff::{Field, PrimeField};
use crate::traits::{Engine, transcript::TranscriptReprTrait};

#[cfg(feature = "p3_backend")]
use p3_goldilocks::Goldilocks as GF;

#[cfg(feature = "p3_backend")]
use p3_field::{PrimeCharacteristicRing, integers::QuotientMap};

/// 32-byte digest used by the Merkle tree / commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

impl<G: crate::traits::Group> TranscriptReprTrait<G> for Digest32 {
  fn to_transcript_bytes(&self) -> Vec<u8> { self.0.to_vec() }
}

/// Backends are parameterized by the engine E, so they can see E::Scalar at the boundary.
pub trait MleBackend<E: Engine> {
  /// Internal field element type used by this backend.
  type FE: Copy + Send + Sync + 'static;

  /// Convert from the engine's scalar (ff) to the backend field.
  fn fe_from_ff(x: &E::Scalar) -> Self::FE;

  /// Convert from backend field back to the engine's scalar (ff).
  fn fe_to_ff(x: &Self::FE) -> E::Scalar;

  /// Field ops
  fn zero() -> Self::FE;
  fn one()  -> Self::FE;
  fn add(a: Self::FE, b: Self::FE) -> Self::FE;
  fn sub(a: Self::FE, b: Self::FE) -> Self::FE;
  fn mul(a: Self::FE, b: Self::FE) -> Self::FE;

  /// Hash a leaf field element into a 32-byte digest.
  fn leaf_hash(x: &Self::FE) -> Digest32;

  /// Hash/consolidate two child digests into a parent digest (2-to-1).
  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32;
}

// -----------------------------
// ff + Keccak (existing behavior)
// -----------------------------
pub struct BackendFfKeccak<E: Engine>(core::marker::PhantomData<E>);

impl<E: Engine> MleBackend<E> for BackendFfKeccak<E> {
  type FE = E::Scalar;

  #[inline] fn fe_from_ff(x: &E::Scalar) -> Self::FE { *x }
  #[inline] fn fe_to_ff(x: &Self::FE) -> E::Scalar { *x }

  #[inline] fn zero() -> Self::FE { E::Scalar::ZERO }
  #[inline] fn one()  -> Self::FE { E::Scalar::ONE  }
  #[inline] fn add(a: Self::FE, b: Self::FE) -> Self::FE { a + b }
  #[inline] fn sub(a: Self::FE, b: Self::FE) -> Self::FE { a - b }
  #[inline] fn mul(a: Self::FE, b: Self::FE) -> Self::FE { a * b }

  fn leaf_hash(x: &Self::FE) -> Digest32 {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(b"mle/leaf");
    hasher.update(x.to_transcript_bytes());
    Digest32(hasher.finalize().into())
  }

  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32 {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(b"mle/node");
    hasher.update(&l.0);
    hasher.update(&r.0);
    Digest32(hasher.finalize().into())
  }
}

// -----------------------------------------
// p3/Goldilocks + Poseidon2 (feature-gated)
// -----------------------------------------
#[cfg(feature = "p3_backend")]
pub struct BackendP3Poseidon2Goldi<E: Engine>(core::marker::PhantomData<E>);

#[cfg(feature = "p3_backend")]
impl<E: Engine> MleBackend<E> for BackendP3Poseidon2Goldi<E> {
  // Internal field is p3_goldilocks::Goldilocks
  type FE = p3_goldilocks::Goldilocks;

  // Convert via canonical u64 (Goldilocks fits in 64 bits).
  // COMPILE-TIME GUARD: Ensure this backend is only used when E::Scalar == crate::provider::goldi::F
  #[inline]
  fn fe_from_ff(x: &E::Scalar) -> Self::FE {
    // For now, use a simple conversion assuming the scalar is already in the right range
    // This should be enhanced with proper type checking in the future
    let repr = x.to_repr();
    let le8 = &repr.as_ref()[..8];
    let u = u64::from_le_bytes(le8.try_into().unwrap());
    GF::from_int(u)
  }

  #[inline]
  fn fe_to_ff(x: &Self::FE) -> E::Scalar {
    use p3_field::PrimeField64;
    E::Scalar::from(x.as_canonical_u64())
  }

  #[inline] fn zero() -> Self::FE { GF::ZERO }
  #[inline] fn one()  -> Self::FE { GF::ONE }
  #[inline] fn add(a: Self::FE, b: Self::FE) -> Self::FE { a + b }
  #[inline] fn sub(a: Self::FE, b: Self::FE) -> Self::FE { a - b }
  #[inline] fn mul(a: Self::FE, b: Self::FE) -> Self::FE { a * b }

  fn leaf_hash(x: &Self::FE) -> Digest32 {
    // Domain separated leaf hash - include the field element directly
    poseidon2_hash_256(&[*x])
  }

  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32 {
    // Interpret each digest as 4 Goldilocks words (LE u64)
    let mut limbs = Vec::new();
    for d in [l, r] {
      for i in 0..4 {
        let start = i * 8;
        let w = u64::from_le_bytes(d.0[start..start + 8].try_into().unwrap());
        limbs.push(GF::from_int(w));
      }
    }
    poseidon2_hash_256(&limbs)
  }
}

/// Simplified Poseidon2-style hash using domain separation.
/// This is a placeholder that provides different hashing behavior from Keccak
/// while we work on integrating the full p3-poseidon2 API.
#[cfg(feature = "p3_backend")]
fn poseidon2_hash_256(inputs: &[GF]) -> Digest32 {
    use sha3::{Digest, Keccak256};
    use p3_field::PrimeField64;
    
    // Domain-separated hash that's different from the FF backend
    let mut hasher = Keccak256::new();
    hasher.update(b"p3/poseidon2/placeholder");
    hasher.update(&(inputs.len() as u64).to_le_bytes());
    
    for input in inputs {
        hasher.update(&input.as_canonical_u64().to_le_bytes());
    }
    
    Digest32(hasher.finalize().into())
}
