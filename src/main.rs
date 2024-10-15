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
    _curve: PhantomData<P>,
}

impl<P: Pairing> Default for Sizes<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Pairing> Sizes<P> {
    pub fn new() -> Self {
        Self {
            _curve: PhantomData,
        }
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
use ark_serialize::*;
use blake2::{Blake2b512, Digest};
use generic_array::GenericArray;
use std::fs::OpenOptions;
use std::io::BufWriter;
use typenum::consts::U64;

/// Compute BLAKE2b("")
pub fn blank_hash() -> GenericArray<u8, U64> {
    Blake2b512::new().finalize()
}

/// The accumulator supports circuits with 2^21 multiplication gates.
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
}

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
}