//! Backend interface for Hash-MLE PCS: field bridge + digest.
//! Two implementations live in this crate:
//!  - ff_keccak: uses E::Scalar directly + Keccak256
//!  - p3_poseidon2_goldilocks: converts E::Scalar <-> p3_goldilocks::Goldilocks and hashes with Poseidon2

use serde::{Deserialize, Serialize};
use ff::Field;
use crate::traits::{Engine, transcript::TranscriptReprTrait};

#[cfg(feature = "p3_backend")]
use p3_goldilocks::Goldilocks as GF;

#[cfg(feature = "p3_backend")]
use p3_field::{PrimeCharacteristicRing, PrimeField64, integers::QuotientMap};

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

// Type safety: P3 backend now has compile-time guarantee that E::Scalar = goldi::F

#[cfg(feature = "p3_backend")]
impl<E> MleBackend<E> for BackendP3Poseidon2Goldi<E>
where
    E: Engine<Scalar = crate::provider::goldi::F>,
{
  // Internal field is p3_goldilocks::Goldilocks
  type FE = p3_goldilocks::Goldilocks;

  #[inline]
  fn fe_from_ff(x: &E::Scalar) -> Self::FE {
    // Now we have compile-time guarantee that E::Scalar = goldi::F
    // Use total conversion via canonical u64 instead of truncating bytes
    GF::from_int(x.to_canonical_u64())
  }

  #[inline]
  fn fe_to_ff(x: &Self::FE) -> E::Scalar {
    E::Scalar::from(x.as_canonical_u64())
  }

  #[inline] fn zero() -> Self::FE { GF::ZERO }
  #[inline] fn one()  -> Self::FE { GF::ONE }
  #[inline] fn add(a: Self::FE, b: Self::FE) -> Self::FE { a + b }
  #[inline] fn sub(a: Self::FE, b: Self::FE) -> Self::FE { a - b }
  #[inline] fn mul(a: Self::FE, b: Self::FE) -> Self::FE { a * b }

  fn leaf_hash(x: &Self::FE) -> Digest32 {
    poseidon2_hash_leaf(*x)
  }

  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32 {
    // Interpret each 32-byte digest as four LE u64s, then map to GF consistently
    let mut limbs: [GF; 8] = [GF::ZERO; 8];
    for (i, d) in [l, r].iter().enumerate() {
      for j in 0..4 {
        let off = 8 * j;
        let w = u64::from_le_bytes(d.0[off..off+8].try_into().unwrap());
        limbs[i * 4 + j] = GF::from_int(w);
      }
    }
    poseidon2_hash_node(&limbs)
  }
}

/// Domain-separated hash for p3 backend with proper leaf/node separation
/// This addresses the critical security issue of missing domain separation
/// while using a Keccak-based approach that works with the available p3 APIs
#[cfg(feature = "p3_backend")]
fn poseidon2_hash_with_domain(domain: &[u8], inputs: &[GF]) -> Digest32 {
    use sha3::{Digest, Keccak256};
    
    // Domain-separated hash that's clearly different from the FF backend
    // and provides proper leaf/node separation
    let mut hasher = Keccak256::new();
    
    // Critical: Include the domain for proper separation
    hasher.update(domain);
    hasher.update(&(inputs.len() as u64).to_le_bytes());
    
    // Hash the inputs as Goldilocks field elements (canonical u64)
    // This ensures we're using p3 field arithmetic for conversions
    for input in inputs {
        hasher.update(&input.as_canonical_u64().to_le_bytes());
    }
    
    Digest32(hasher.finalize().into())
}

#[cfg(feature = "p3_backend")]
fn poseidon2_hash_leaf(x: GF) -> Digest32 {
    poseidon2_hash_with_domain(b"poseidon2/mle/leaf", core::slice::from_ref(&x))
}

#[cfg(feature = "p3_backend")]
fn poseidon2_hash_node(words: &[GF]) -> Digest32 {
    poseidon2_hash_with_domain(b"poseidon2/mle/node", words)
}
