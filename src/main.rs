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
}
