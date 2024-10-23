use ark_ec::pairing::Pairing;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::*;

use ark_mnt6_753::*;
use ark_std::UniformRand;
use num_traits::identities::Zero;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

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
        (TAU_POWERS_G1_LENGTH * self.g1_uncompressed_byte_size) + // g1 tau powers
        (TAU_POWERS_LENGTH * self.g2_uncompressed_byte_size) + // g2 tau powers
        (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) + // alpha tau powers
        (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // beta tau powers
        + 32 // lengths of vectors
        + self.g2_uncompressed_byte_size // beta in g2
        + 64 // blake2b hash of previous contribution
    }


    ///// The "public key" is used to verify a contribution was correctly
    /// computed.
    pub fn public_key_size(&self) -> usize {
        PublicKey::default().uncompressed_size()
    }

    /// The size of the contribution on disk.
    pub fn contribution_byte_size(&self) -> usize {
        (TAU_POWERS_G1_LENGTH * self.g1_compressed_byte_size) + // g1 tau powers
        (TAU_POWERS_LENGTH * self.g2_compressed_byte_size) + // g2 tau powers
        (TAU_POWERS_LENGTH * self.g1_compressed_byte_size) + // alpha tau powers
        (TAU_POWERS_LENGTH * self.g1_compressed_byte_size) // beta tau powers
        + self.g2_compressed_byte_size // beta in g2
        + 64 // blake2b hash of input accumulator
        + self.public_key_size() // public key
    }
}

/// Checks if pairs have the same ratio.
fn same_ratio<P: Pairing>(g1: (P::G1Affine, P::G1Affine), g2: (P::G2Affine, P::G2Affine)) -> bool {
    P::pairing(g1.0, g2.1) == P::pairing(g1.1, g2.0)
}

/// Computes a random linear combination over v1/v2.
///
/// Checking that many pairs of elements are exponentiated by
/// the same `x` can be achieved (with high probability) with
/// the following technique:
///
/// Given v1 = [a, b, c] and v2 = [as, bs, cs], compute
/// (a*r1 + b*r2 + c*r3, (as)*r1 + (bs)*r2 + (cs)*r3) for some
/// random r1, r2, r3. Given (g, g^s)...
///
/// e(g, (as)*r1 + (bs)*r2 + (cs)*r3) = e(g^s, a*r1 + b*r2 + c*r3)
///
/// ... with high probability.
fn merge_pairs<C: SWCurveConfig>(v1: &[Affine<C>], v2: &[Affine<C>]) -> (Affine<C>, Affine<C>) {
    use rand::thread_rng;

    assert_eq!(v1.len(), v2.len());

    let chunk_size = (v1.len() / num_cpus::get()) + 1;

    let s = Arc::new(Mutex::new(Projective::<C>::zero()));
    let sx = Arc::new(Mutex::new(Projective::<C>::zero()));

    v1.par_chunks(chunk_size)
        .zip(v2.par_chunks(chunk_size))
        .for_each(|(v1, v2)| {
            let s = s.clone();
            let sx = sx.clone();

            // We do not need to be overly cautious of the RNG
            // used for this check.
            let rng = &mut thread_rng();

            let mut local_s = Projective::<C>::zero();
            let mut local_sx = Projective::<C>::zero();

            for (v1, v2) in v1.iter().zip(v2.iter()) {
                let rho = C::ScalarField::rand(rng);
                let v1 = *v1 * rho;
                let v2 = *v2 * rho;

                local_s += v1;
                local_sx += v2;
            }

            *s.lock().unwrap() += local_s;
            *sx.lock().unwrap() += local_sx;
        });

    let s = s.lock().unwrap().into_affine();
    let sx = sx.lock().unwrap().into_affine();

    (s, sx)
}

/// Construct a single pair (s, s^x) for a vector of
/// the form [1, x, x^2, x^3, ...].
fn power_pairs<C: SWCurveConfig>(v: &[Affine<C>]) -> (Affine<C>, Affine<C>) {
    merge_pairs(&v[0..(v.len() - 1)], &v[1..])
}

////////////////////////
//// 2nd Week ////
////////////////////////
//----------------------------------------------
use ark_serialize::CanonicalSerialize;
// use ark_serialize::*;
use blake2::{Blake2b512, Digest};
use generic_array::GenericArray;
use std::fs::OpenOptions;
use std::io::BufWriter;
use typenum::consts::U64;

/// Compute BLAKE2b("")
pub fn blank_hash() -> GenericArray<u8, U64> {
    Blake2b512::new().finalize()
}

/// The accumulator supports circuits with 2^n multiplication gates.
const TAU_POWERS_LENGTH: usize = 1 << 5;
/// More tau powers are needed in G1 because the Groth16 H query
/// includes terms of the form tau^i * (tau^m - 1) = tau^(i+m) - tau^i
/// where the largest i = m - 2, requiring the computation of tau^(2m - 2)
/// and thus giving us a vector length of 2^22 - 1.
const TAU_POWERS_G1_LENGTH: usize = (TAU_POWERS_LENGTH << 1) - 1;

/// The `Accumulator` is an object that participants of the ceremony contribute
/// randomness to. This object contains powers of trapdoor `tau` in G1 and in G2 over
/// fixed generators, and additionally in G1 over two other generators of exponents
/// `alpha` and `beta` over those fixed generators. In other words:
///
/// * (τ, τ<sup>2</sup>, ..., τ<sup>2<sup>22</sup> - 2</sup>, α, ατ, ατ<sup>2</sup>, ..., ατ<sup>2<sup>21</sup> - 1</sup>, β, βτ, βτ<sup>2</sup>, ..., βτ<sup>2<sup>21</sup> - 1</sup>)<sub>1</sub>
/// * (β, τ, τ<sup>2</sup>, ..., τ<sup>2<sup>21</sup> - 1</sup>)<sub>2</sub>
#[derive(PartialEq, Eq, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Accumulator {
    /// tau^0, tau^1, tau^2, ..., tau^{TAU_POWERS_G1_LENGTH - 1}
    pub tau_powers_g1: Vec<G1Affine>,
    /// tau^0, tau^1, tau^2, ..., tau^{TAU_POWERS_LENGTH - 1}
    pub tau_powers_g2: Vec<G2Affine>,
    /// alpha * tau^0, alpha * tau^1, alpha * tau^2, ..., alpha * tau^{TAU_POWERS_LENGTH - 1}
    pub alpha_tau_powers_g1: Vec<G1Affine>,
    /// beta * tau^0, beta * tau^1, beta * tau^2, ..., beta * tau^{TAU_POWERS_LENGTH - 1}
    pub beta_tau_powers_g1: Vec<G1Affine>,
    /// beta
    pub beta_g2: G2Affine,
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
            tau_powers_g1: vec![G1Affine::generator(); TAU_POWERS_G1_LENGTH],
            tau_powers_g2: vec![G2Affine::generator(); TAU_POWERS_LENGTH],
            alpha_tau_powers_g1: vec![G1Affine::generator(); TAU_POWERS_LENGTH],
            beta_tau_powers_g1: vec![G1Affine::generator(); TAU_POWERS_LENGTH],
            beta_g2: G2Affine::generator(),
        }
    }

    /// Transforms the accumulator with a private key.
    /// tau, tau^2, tau^3,...
    /// t, t^2, t^3,...
    /// tau^t, (tau^2)^(t^2),...
    pub fn transform(&mut self, key: &PrivateKey) {
        // Construct the powers of tau
        let mut taupowers = vec![Fr::zero(); TAU_POWERS_G1_LENGTH];
        let chunk_size = TAU_POWERS_G1_LENGTH / num_cpus::get();

        // Construct exponents in parallel
        taupowers
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, taupowers)| {
                let mut acc = key.tau.pow([(i * chunk_size) as u64]);

                for t in taupowers {
                    *t = acc;
                    acc *= key.tau;
                }
            });

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

        batch_exp(&mut self.tau_powers_g1, &taupowers[0..], None);
        batch_exp(
            &mut self.tau_powers_g2,
            &taupowers[0..TAU_POWERS_LENGTH],
            None,
        );
        batch_exp(
            &mut self.alpha_tau_powers_g1,
            &taupowers[0..TAU_POWERS_LENGTH],
            Some(&key.alpha),
        );
        batch_exp(
            &mut self.beta_tau_powers_g1,
            &taupowers[0..TAU_POWERS_LENGTH],
            Some(&key.beta),
        );
        self.beta_g2 = (self.beta_g2 * key.beta).into_affine();
    }
}

//----------------------------------------------
//----------------------------------------------
////////////////////////
//// 3rd Week ////
////////////////////////
// use ark_mnt6_753::MNT6_753;
use ark_serialize::{CanonicalDeserialize, Compress, Validate};
// use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

use std::env;
// use std::fs::OpenOptions;
use std::io::{self, BufReader, Read, Write};
// use std::io::{self, BufReader, BufWriter, Read, Write};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

use ark_ff::fields::Field;
// use ark_mnt6_753::*;
//----------------------------------------------

// /// The size of the accumulator on disk.
// pub fn accumulator_byte_size_with_hash(&self) -> usize {
//     (TAU_POWERS_G1_LENGTH * self.g1_uncompressed_byte_size) + // g1 tau powers
//     (TAU_POWERS_LENGTH * self.g2_uncompressed_byte_size) + // g2 tau powers
//     (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) + // alpha tau powers
//     (TAU_POWERS_LENGTH * self.g1_uncompressed_byte_size) // beta tau powers
//     + 32 // lengths of vectors
//     + self.g2_uncompressed_byte_size // beta in g2
//     + 64 // blake2b hash of previous contribution
// }

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
    tau_g1: (G1Affine, G1Affine),
    alpha_g1: (G1Affine, G1Affine),
    beta_g1: (G1Affine, G1Affine),
    tau_g2: G2Affine,
    alpha_g2: G2Affine,
    beta_g2: G2Affine,
}

/// Contains the secrets τ, α and β that the participant of the ceremony must destroy.
pub struct PrivateKey {
    tau: Fr,
    alpha: Fr,
    beta: Fr,
}

/// Constructs a keypair given an RNG and a 64-byte transcript `digest`.
pub fn keypair<R: Rng>(rng: &mut R, digest: &[u8]) -> (PublicKey, PrivateKey) {
    assert_eq!(digest.len(), 64);

    let tau = Fr::rand(rng);
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);

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

    let pk_tau = op(tau, 0);
    let pk_alpha = op(alpha, 1);
    let pk_beta = op(beta, 2);

    (
        PublicKey {
            tau_g1: pk_tau.0,
            alpha_g1: pk_alpha.0,
            beta_g1: pk_beta.0,
            tau_g2: pk_tau.1,
            alpha_g2: pk_alpha.1,
            beta_g2: pk_beta.1,
        },
        PrivateKey { tau, alpha, beta },
    )
}

// Verifies a transformation of the `Accumulator` with the `PublicKey`, given a 64-byte transcript `digest`.
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

    let tau_g2_s = compute_g2_s(key.tau_g1.0, key.tau_g1.1, 0);
    let alpha_g2_s = compute_g2_s(key.alpha_g1.0, key.alpha_g1.1, 1);
    let beta_g2_s = compute_g2_s(key.beta_g1.0, key.beta_g1.1, 2);

    // Check the proofs-of-knowledge for tau/alpha/beta
    if !same_ratio::<MNT6_753>(key.tau_g1, (tau_g2_s, key.tau_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.alpha_g1, (alpha_g2_s, key.alpha_g2)) {
        return false;
    }
    if !same_ratio::<MNT6_753>(key.beta_g1, (beta_g2_s, key.beta_g2)) {
        return false;
    }

    // Check the correctness of the generators for tau powers
    if after.tau_powers_g1[0] != G1Affine::generator() {
        return false;
    }
    if after.tau_powers_g2[0] != G2Affine::generator() {
        return false;
    }

    // Did the participant multiply the previous tau by the new one?
    if !same_ratio::<MNT6_753>(
        (before.tau_powers_g1[1], after.tau_powers_g1[1]),
        (tau_g2_s, key.tau_g2),
    ) {
        return false;
    }

    // Did the participant multiply the previous alpha by the new one?
    if !same_ratio::<MNT6_753>(
        (before.alpha_tau_powers_g1[0], after.alpha_tau_powers_g1[0]),
        (alpha_g2_s, key.alpha_g2),
    ) {
        return false;
    }

    // Did the participant multiply the previous beta by the new one?
    if !same_ratio::<MNT6_753>(
        (before.beta_tau_powers_g1[0], after.beta_tau_powers_g1[0]),
        (beta_g2_s, key.beta_g2),
    ) {
        return false;
    }
    if !same_ratio::<MNT6_753>(
        (before.beta_tau_powers_g1[0], after.beta_tau_powers_g1[0]),
        (before.beta_g2, after.beta_g2),
    ) {
        return false;
    }

    // Are the powers of tau correct?
    if !same_ratio::<MNT6_753>(
        power_pairs(&after.tau_powers_g1),
        (after.tau_powers_g2[0], after.tau_powers_g2[1]),
    ) {
        return false;
    }
    if !same_ratio::<MNT6_753>(
        (after.tau_powers_g1[0], after.tau_powers_g1[1]),
        power_pairs(&after.tau_powers_g2),
    ) {
        return false;
    }
    if !same_ratio::<MNT6_753>(
        power_pairs(&after.alpha_tau_powers_g1),
        (after.tau_powers_g2[0], after.tau_powers_g2[1]),
    ) {
        return false;
    }
    if !same_ratio::<MNT6_753>(
        power_pairs(&after.beta_tau_powers_g1),
        (after.tau_powers_g2[0], after.tau_powers_g2[1]),
    ) {
        return false;
    }

    true
}
//----------------------------------------------
//----------------------------------------------
fn main() {
    println!("Starting the ceremmony...");

    use rand::thread_rng;

    let rng = &mut thread_rng();

    let s = Fr::rand(rng);
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let g1_s = (g1 * s).into_affine();
    let g2_s = (g2 * s).into_affine();

    // Print the result of same_ratio::<MNT6_753>((g1, g1_s), (g2, g2_s))
    let result1 = same_ratio::<MNT6_753>((g1, g1_s), (g2, g2_s));
    println!("Same ratio true check result1: {}", result1);

    let result2 = same_ratio::<MNT6_753>((g1_s, g1), (g2, g2_s));
    println!("Same ratio false check result2: {}", result2);

    use ark_std::One;

    let rng = &mut thread_rng();

    let mut v = vec![];
    let x = Fr::rand(rng);
    let mut acc = Fr::one();
    for _ in 0..33 {
        v.push((G1Affine::generator() * acc).into_affine());
        acc *= x;
    }
    // Print the vector of points
    // println!("Vector of G1 points (v): {:?}", v);

    let gx = (G2Affine::generator() * x).into_affine();

    assert!(same_ratio::<MNT6_753>(
        power_pairs(&v),
        (G2Affine::generator(), gx)
    ));

    v[1] = (v[1] * Fr::rand(rng)).into_affine();

    assert!(!same_ratio::<MNT6_753>(
        power_pairs(&v),
        (G2Affine::generator(), gx)
    ));

    // Replace println! with write statements to file
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create("output.txt").expect("Unable to create file");
    writeln!(file, "G1 Generator: {:?}", g1).expect("Unable to write to file");
    writeln!(file, "G2 Generator: {:?}", g2).expect("Unable to write to file");
    writeln!(
        file,
        "Same ratio check result (g1, g1_s) vs (g2, g2_s): {}",
        result1
    )
    .expect("Unable to write to file");
    writeln!(
        file,
        "Same ratio check result (g1_s, g1) vs (g2, g2_s): {}",
        result2
    )
    .expect("Unable to write to file");
    writeln!(file, "Vector of G1 points (v): {:?}", v).expect("Unable to write to file");
    // writeln!(file, "Power pairs result: {:?}", power_pairs_result)
    // .expect("Unable to write to file");
    writeln!(file, "Gx: {:?}", gx).expect("Unable to write to file");
    ////////////////////////
    //// 2nd Week ////
    ////////////////////////
    let writer = OpenOptions::new()
        .read(false)
        .write(true)
        .create_new(true)
        .open("challenge")
        .expect("unable to create `./challenge`");

    let mut writer = BufWriter::new(writer);

    // Write a blank BLAKE2b hash:
    writer
        .write_all(blank_hash().as_slice())
        .expect("unable to write blank hash to `./challenge`");

    let acc = Accumulator::new();
    acc.serialize_uncompressed(&mut writer)
        .expect("unable to write fresh accumulator to `./challenge`");
    writer.flush().expect("unable to flush accumulator to disk");

    println!("Wrote a fresh accumulator to `./challenge`");

    ////////////////////////
    //// 3rd Week ////
    ////////////////////////
    // Create an RNG based on a mixture of system randomness and user provided randomness
    let mut rng = {
        use blake2::{Blake2b512, Digest};
        use rand::{rngs::OsRng, Rng, SeedableRng};
        use rand_chacha::ChaChaRng;

        let h = {
            let mut system_rng = OsRng::default();
            let mut h = Blake2b512::default();

            // Gather 1024 bytes of entropy from the system
            for _ in 0..1024 {
                let r: u8 = system_rng.gen();
                h.update([r]);
            }

            // Ask the user to provide some information for additional entropy
            let mut user_input = String::new();
            println!("Type some random text and press [ENTER] to provide additional entropy...");
            io::stdin()
                .read_line(&mut user_input)
                .expect("expected to read some random text from the user");

            // Hash it all up to make a seed
            h.update(user_input.as_bytes());
            h.finalize()
        };

        // Interpret the first 32 bytes of the digest as 8 32-bit words
        let mut seed = [0; 32];
        seed.copy_from_slice(&h[..32]);

        ChaChaRng::from_seed(seed)
    };
    // Print the path before trying to open itxcşklşkl

    match env::current_dir() {
        Ok(path) => println!(
            "-------------------------The current directory is: {}",
            path.display()
        ),
        Err(e) => println!("Error getting current directory: {}", e),
    }

    // Try to load `./challenge` from disk.
    let reader = OpenOptions::new()
        .read(true)
        .open("challenge")
        .expect("unable open `./challenge` in this directory");

    {
        let metadata = reader
            .metadata()
            .expect("unable to get filesystem metadata for `./challenge`");
        if metadata.len() != (Sizes::<MNT6_753>::new().accumulator_byte_size_with_hash() as u64) {
            panic!(
                "The size of `./challenge` should be {}, but it's {}, so something isn't right.",
                Sizes::<MNT6_753>::new().accumulator_byte_size_with_hash(),
                metadata.len()
            );
        }
    }

    let reader = BufReader::new(reader);
    let mut reader = HashReader::new(reader);

    // Create `./response` in this directory
    let writer = OpenOptions::new()
        .read(false)
        .write(true)
        .create_new(true)
        .open("response")
        .expect("unable to create `./response` in this directory");

    let writer = BufWriter::new(writer);
    let mut writer = HashWriter::new(writer);

    println!("Reading `./challenge` into memory...");

    // Read the BLAKE2b hash of the previous contribution
    {
        // We don't need to do anything with it, but it's important for
        // the hash chain.
        let mut tmp = [0; 64];
        reader
            .read_exact(&mut tmp)
            .expect("unable to read BLAKE2b hash of previous contribution");
    }

    // Load the current accumulator into memory
    let mut current_accumulator =
        Accumulator::deserialize_with_mode(&mut reader, Compress::No, Validate::No)
            .expect("unable to read uncompressed accumulator");

    // Get the hash of the current accumulator
    let current_accumulator_hash = reader.into_hash();

    // Construct our keypair using the RNG we created above
    let (pub_key, priv_key) = keypair(&mut rng, current_accumulator_hash.as_ref());

    // Perform the transformation
    println!("Computing, this could take a while...");
    current_accumulator.transform(&priv_key);
    println!("Writing your contribution to `./response`...");

    // Write the hash of the input accumulator
    writer
        .write_all(current_accumulator_hash.as_ref())
        .expect("unable to write BLAKE2b hash of input accumulator");

    // Write the transformed accumulator (in compressed form, to save upload bandwidth for disadvantaged
    // players.)
    current_accumulator
        .serialize_compressed(&mut writer)
        .expect("unable to write transformed accumulator");

    // Write the public key
    pub_key
        .serialize_uncompressed(&mut writer)
        .expect("unable to write public key");

    // Get the hash of the contribution, so the user can compare later
    let contribution_hash = writer.into_hash();

    print!(
        "Done!\n\n\
              Your contribution has been written to `./response`\n\n\
              The BLAKE2b hash of `./response` is:\n"
    );

    for line in contribution_hash.as_slice().chunks(16) {
        print!("\t");
        for section in line.chunks(4) {
            for b in section {
                print!("{:02x}", b);
            }
            print!(" ");
        }
        println!();
    }

    println!("\n");
}