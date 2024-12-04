use ark_ec::pairing::Pairing;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::*;
use ark_ff::fields::Field;
use ark_mnt6_753::*;
use ark_serialize::*;
use ark_std::{
    rand::{Rng, SeedableRng},
    UniformRand,
};
use blake2::{Blake2b512, Digest};
use generic_array::GenericArray;
use num_traits::identities::Zero;
use rand_chacha::ChaChaRng;
use rayon::prelude::*;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use typenum::consts::U64;

/// The accumulator supports circuits with 2^21 multiplication gates.
// const TAU_POWERS_LENGTH: usize = 1 << 22;
const TAU_POWERS_LENGTH: usize = 1 << 5;
/// More tau powers are needed in G1 because the Groth16 H query
/// includes terms of the form tau^i * (tau^m - 1) = tau^(i+m) - tau^i
/// where the largest i = m - 2, requiring the computation of tau^(2m - 2)
/// and thus giving us a vector length of 2^22 - 1.
const TAU_POWERS_G1_LENGTH: usize = (TAU_POWERS_LENGTH << 1) - 1;

pub struct Sizes<P: Pairing> {
    g1_uncompressed_byte_size: usize,
    g2_uncompressed_byte_size: usize,
    g1_compressed_byte_size: usize,
    g2_compressed_byte_size: usize,
    _curve: PhantomData<P>,
}

impl<P: Pairing> Default for Sizes<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Pairing> Sizes<P> {
    pub fn new() -> Self {
        let g1 = <P as Pairing>::G1Affine::zero();
        let g2 = <P as Pairing>::G2Affine::zero();
        Self {
            g1_uncompressed_byte_size: g1.uncompressed_size(),
            g2_uncompressed_byte_size: g2.uncompressed_size(),
            g1_compressed_byte_size: g1.compressed_size(),
            g2_compressed_byte_size: g2.compressed_size(),
            _curve: PhantomData,
        }
    }

    /// The size of the accumulator on disk.
    pub fn accumulator_byte_size_with_hash(&self) -> usize {
        (TAU_POWERS_G1_LENGTH * self.g1_uncompressed_byte_size) // alpha in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // gamma in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // beta in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // delta in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // eta1 in g1
        + 32 // lengths of vectors
        + self.g2_uncompressed_byte_size // beta in g2
        + self.g2_uncompressed_byte_size // delta in g2
        + self.g2_uncompressed_byte_size // eta1 in g2
        + 64 // blake2b hash of previous contribution
    }

    /// The "public key" is used to verify a contribution was correctly
    /// computed.
    pub fn public_key_size(&self) -> usize {
        PublicKey::default().uncompressed_size()
    }

    /// The size of the contribution on disk.
    pub fn contribution_byte_size(&self) -> usize {
        (TAU_POWERS_G1_LENGTH * self.g1_uncompressed_byte_size) // alpha in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // gamma in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // beta in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // delta in g1
        + (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // eta1 in g1
        + self.g2_uncompressed_byte_size // beta in g2
        + self.g2_uncompressed_byte_size // delta in g2
        + self.g2_uncompressed_byte_size // eta1 in g2
        + 64 // blake2b hash of input accumulator
        + self.public_key_size() // public key
    }
}

/// Compute BLAKE2b("")
pub fn blank_hash() -> GenericArray<u8, U64> {
    Blake2b512::new().finalize()
}

/// Abstraction over a reader which hashes the data being read.
pub struct HashReader<R: Read> {
    reader: R,
    hasher: Blake2b512,
}

impl<R: Read> HashReader<R> {
    /// Construct a new `HashReader` given an existing `reader` by value.
    pub fn new(reader: R) -> Self {
        HashReader {
            reader,
            hasher: Blake2b512::default(),
        }
    }

    /// Destroy this reader and return the hash of what was read.
    pub fn into_hash(self) -> GenericArray<u8, U64> {
        self.hasher.finalize()
    }
}

impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes = self.reader.read(buf)?;

        if bytes > 0 {
            self.hasher.update(&buf[0..bytes]);
        }

        Ok(bytes)
    }
}

/// Abstraction over a writer which hashes the data being written.
pub struct HashWriter<W: Write> {
    writer: W,
    hasher: Blake2b512,
}

impl<W: Write> HashWriter<W> {
    /// Construct a new `HashWriter` given an existing `writer` by value.
    pub fn new(writer: W) -> Self {
        HashWriter {
            writer,
            hasher: Blake2b512::default(),
        }
    }

    /// Destroy this writer and return the hash of what was written.
    pub fn into_hash(self) -> GenericArray<u8, U64> {
        self.hasher.finalize()
    }
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes = self.writer.write(buf)?;

        if bytes > 0 {
            self.hasher.update(&buf[0..bytes]);
        }

        Ok(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// Hashes to G2 using the first 32 bytes of `digest`. Panics if `digest` is less
/// than 32 bytes.
fn hash_to_g2(digest: &[u8]) -> G2Projective {
    assert!(digest.len() >= 32);

    let mut seed = [0; 32];
    seed.copy_from_slice(&digest[..32]);

    ChaChaRng::from_seed(seed).gen()
}

/// Contains terms of the form (s<sub>1</sub>, s<sub>1</sub><sup>x</sup>, H(s<sub>1</sub><sup>x</sup>)<sub>2</sub>, H(s<sub>1</sub><sup>x</sup>)<sub>2</sub><sup>x</sup>)
/// for all x in τ, α and β, and some s chosen randomly by its creator. The function H "hashes into" the group G2. No points in the public key may be the identity.
///
/// The elements in G2 are used to verify transformations of the accumulator. By its nature, the public key proves
/// knowledge of τ, α and β.
///
/// It is necessary to verify `same_ratio`((s<sub>1</sub>, s<sub>1</sub><sup>x</sup>), (H(s<sub>1</sub><sup>x</sup>)<sub>2</sub>, H(s<sub>1</sub><sup>x</sup>)<sub>2</sub><sup>x</sup>)).
#[derive(Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PublicKey {
    // tau_g1: (G1Affine, G1Affine),
    alpha_g1: (G1Affine, G1Affine),
    gamma_g1: (G1Affine, G1Affine),
    beta_g1: (G1Affine, G1Affine),
    delta_g1: (G1Affine, G1Affine),
    eta1_g1: (G1Affine, G1Affine),
    // tau_g2: G2Affine,
    alpha_g2: G2Affine,
    gamma_g2: G2Affine,
    beta_g2: G2Affine,
    delta_g2: G2Affine,
    eta1_g2: G2Affine,
}

/// Contains the secrets τ, α and β that the participant of the ceremony must destroy.
pub struct PrivateKey {
    // tau: Fr,
    alpha: Fr,
    gamma: Fr,
    beta: Fr,
    delta: Fr,
    eta1: Fr,
}

/// Constructs a keypair given an RNG and a 64-byte transcript `digest`.
pub fn keypair<R: Rng>(rng: &mut R, digest: &[u8]) -> (PublicKey, PrivateKey) {
    assert_eq!(digest.len(), 64);

    // let tau = Fr::rand(rng);
    let alpha = Fr::rand(rng);
    let gamma = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let delta = Fr::rand(rng);
    let eta1 = Fr::rand(rng);

    let mut op = |x, personalization: u8| {
        // Sample random g^s
        let g1_s = G1Projective::rand(rng).into_affine();
        // Compute g^{s*x}
        let g1_s_x = (g1_s * x).into_affine();
        // Compute BLAKE2b(personalization | transcript | g^s | g^{s*x})
        let h = {
            let mut h = Blake2b512::default();
            h.update([personalization]);
            h.update(digest);
            g1_s.serialize_uncompressed(&mut h).unwrap();
            g1_s_x.serialize_uncompressed(&mut h).unwrap();
            h.finalize()
        };
        // Hash into G2 as g^{s'}
        let g2_s = hash_to_g2(h.as_ref()).into_affine();
        // Compute g^{s'*x}
        let g2_s_x = (g2_s * x).into_affine();

        ((g1_s, g1_s_x), g2_s_x)
    };

    // let pk_tau = op(tau, 0);
    let pk_alpha = op(alpha, 1);
    let pk_gamma = op(alpha, 1);
    let pk_beta = op(beta, 2);
    let pk_delta = op(delta, 3);
    let pk_eta1 = op(eta1, 4);

    (
        PublicKey {
            alpha_g1: pk_alpha.0,
            gamma_g1: pk_alpha.0,
            beta_g1: pk_beta.0,
            delta_g1: pk_delta.0,
            eta1_g1: pk_eta1.0,
            alpha_g2: pk_alpha.1,
            gamma_g2: pk_gamma.1,
            beta_g2: pk_beta.1,
            delta_g2: pk_delta.1,
            eta1_g2: pk_eta1.1,
        },
        PrivateKey {
            // tau,
            alpha,
            gamma,
            beta,
            delta,
            eta1,
        },
    )
}

#[derive(PartialEq, Eq, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Accumulator {
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    pub delta_g2: G2Affine,
    pub eta1_g2: G2Affine,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    /// Constructs an "initial" accumulator with τ = 1, α = 1, β = 1.
    pub fn new() -> Self {
        Accumulator {
            beta_g2: G2Affine::generator(),
            gamma_g2: G2Affine::generator(),
            delta_g2: G2Affine::generator(),
            eta1_g2: G2Affine::generator(),
        }
    }

    /// Transforms the accumulator with a private key.
    /// tau, tau^2, tau^3,...
    /// t, t^2, t^3,...
    /// tau^t, (tau^2)^(t^2),...
    pub fn transform(&mut self, key: &PrivateKey) {
        /// Exponentiate a large number of points, with an optional coefficient to be applied to the
        /// exponent.
        fn batch_exp<C: SWCurveConfig>(
            bases: &mut [Affine<C>],
            exp: &[C::ScalarField],
            coeff: Option<&C::ScalarField>,
        ) {
            assert_eq!(bases.len(), exp.len());

            // Perform wNAF over multiple cores, placing results into `projective`.
            let projective: Vec<_> = bases
                .par_iter()
                .zip(exp)
                .map(|(base, exp)| {
                    let mut exp = *exp;
                    if let Some(coeff) = coeff {
                        exp *= coeff;
                    }

                    // PITODO: base * exp, check if arkworks does that efficiently already
                    // or whether we need to use some scalar-mul thingy
                    *base * exp
                })
                .collect();

            // Perform batch normalization
            // Turn it all back into affine points
            let affine = Projective::<C>::normalize_batch(&projective);
            bases.copy_from_slice(&affine);
        }

        self.beta_g2 = (self.beta_g2 * key.beta).into_affine();
        self.delta_g2 = (self.delta_g2 * key.beta).into_affine();
        self.eta1_g2 = (self.eta1_g2 * key.beta).into_affine();
    }
}

/// Verifies a transformation of the `Accumulator` with the `PublicKey`, given a 64-byte transcript `digest`.
pub fn verify_transform(
    before: &Accumulator,
    after: &Accumulator,
    key: &PublicKey,
    digest: &[u8],
) -> bool {
    assert_eq!(digest.len(), 64);

    let compute_g2_s = |g1_s: G1Affine, g1_s_x: G1Affine, personalization: u8| {
        let mut h = Blake2b512::default();
        h.update([personalization]);
        h.update(digest);
        g1_s.serialize_uncompressed(&mut h).unwrap();
        g1_s_x.serialize_uncompressed(&mut h).unwrap();
        hash_to_g2(h.finalize().as_ref()).into_affine()
    };
    let alpha_g2_s = compute_g2_s(key.alpha_g1.0, key.alpha_g1.1, 1);
    let gamma_g2_s = compute_g2_s(key.gamma_g1.0, key.gamma_g1.1, 4);
    let beta_g2_s = compute_g2_s(key.beta_g1.0, key.beta_g1.1, 2);
    let delta_g2_s = compute_g2_s(key.delta_g1.0, key.delta_g1.1, 3);
    let eta1_g2_s = compute_g2_s(key.eta1_g1.0, key.eta1_g1.1, 3);

    if !same_ratio::<MNT6_753>(key.alpha_g1, (alpha_g2_s, key.alpha_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.gamma_g1, (gamma_g2_s, key.gamma_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.beta_g1, (beta_g2_s, key.beta_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.delta_g1, (delta_g2_s, key.delta_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.eta1_g1, (eta1_g2_s, key.eta1_g2)) {
        return false;
    }
    true
}

/// Checks if pairs have the same ratio.
fn same_ratio<P: Pairing>(g1: (P::G1Affine, P::G1Affine), g2: (P::G2Affine, P::G2Affine)) -> bool {
    P::pairing(g1.0, g2.1) == P::pairing(g1.1, g2.0)
}
