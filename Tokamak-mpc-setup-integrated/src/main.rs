#![allow(non_snake_case)]

use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_serialize::CanonicalSerialize;

use blake2::Digest;
use icicle_bls12_381::curve::{CurveCfg, G2CurveCfg, ScalarField};
use icicle_core::curve::Curve;
use icicle_core::traits::{Arithmetic, FieldImpl};
use libs::field_structures::Tau;
use libs::group_structures::{G1serde, Sigma2};
use libs::iotools::{from_coef_vec_to_g1serde_vec, SetupParams};
use rand_chacha::rand_core::SeedableRng;
use std::ops::Mul;
use std::time::Instant;

pub mod utils;

 

fn main() {
    let start1 = Instant::now();

    // Generate random affine points on the elliptic curve (G1 and G2)
    println!("Generating random generator points...");
    let g1_gen = CurveCfg::generate_random_affine_points(1)[0];
    let g2_gen = G2CurveCfg::generate_random_affine_points(1)[0];

    // Generate a random secret parameter tau (x and y only, no z as per the paper)
    println!("Generating random tau parameter...");
    let tau = Tau::gen();

    // Load setup parameters from a JSON file
    println!("Loading setup parameters...");
    let setup_file_name = "setupParams.json";
    let setup_params = SetupParams::from_path(setup_file_name).unwrap();

    // Extract key parameters from setup_params
    let s_max = setup_params.s_max; // The maximum number of placements.

    // Verify s_max is a power of two
    if !s_max.is_power_of_two() {
        panic!("s_max is not a power of two.");
    }

    let mut resultMatrix = vec![ScalarField::zero(); (4+1)* s_max * s_max].into_boxed_slice();
    let smax_squared = s_max * s_max;
    for k in 0..=4 {
        let alpha = tau.alpha.pow(k);
        for i in 0..=s_max-1 {
            let x = tau.x.pow(i);
            for j in 0..=s_max-1 {
                let y = tau.y.pow(j);
                let idx = k * smax_squared + i * s_max + j;
                resultMatrix [idx] = alpha * x * y;
            }
        }
    }
    println!("Finished computing result matrix. len {}", resultMatrix.len());
    let mut res = vec![G1serde::zero(); resultMatrix.len()].into_boxed_slice();
    from_coef_vec_to_g1serde_vec(
        &resultMatrix,
        &g1_gen,
        &mut res,
    );
    let sigma2 = Sigma2::gen(&tau,&g2_gen);

    //TODO

    println!("Finished in {:?}", start1.elapsed());

}
