// use ark_ec::{AffineRepr, CurveGroup};
// use ark_ff::PrimeField;
// use ark_mnt6_753::{Fr, G1Affine, G2Affine};
// use ark_serialize::CanonicalSerialize;
// use deneme::{compute1, compute2};
// use std::fs::OpenOptions;
// use std::io::{BufWriter, Write};


use ark_bls12_381::{Fr, G1Affine, G2Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use deneme::{compute1, compute2};

use rand::thread_rng;
use ark_ff::UniformRand;

use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;

pub fn initialize_all() {
    println!("Initializing Type-1 (α) and Type-2 (α * β) parameters...");
    let rnd_string = "rndString";

    // Type-1 (α)
    let alpha_g1_pre = G1Affine::identity();
    let (alpha_g1_out, alpha_j_g1, y) = compute1(alpha_g1_pre, rnd_string);

    // Type-2 (α * β)
    let alpha_beta_prev = G1Affine::identity();
    let (alpha_beta_out, alpha_j_g1_2, beta_j_g1, alpha_beta_g2, y_alpha, y_beta) =
        compute2(alpha_beta_prev, rnd_string);

    // Save all accumulators to a single file
    save_to_single_file(
        "accumulator_all_types.bin",
        &[
            &alpha_g1_pre, &alpha_g1_out, &alpha_j_g1, // Type-1 (G1)
            &alpha_beta_prev, &alpha_beta_out, &alpha_j_g1_2, &beta_j_g1, // Type-2 (G1)
        ],
        &[
            &y, &alpha_beta_g2, &y_alpha, &y_beta // All G2 elements
        ],
    );
}

fn save_to_single_file<T: CanonicalSerialize, U: CanonicalSerialize>(
    file_path: &str,
    g1_data: &[&T],
    g2_data: &[&U],
) {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .expect("Failed to open file");

    let mut writer = BufWriter::new(&mut file);

    // Save G1 Elements
    for item in g1_data {
        let mut serialized = Vec::new();
        item.serialize_compressed(&mut serialized).expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    // Save G2 Elements
    for item in g2_data {
        let mut serialized = Vec::new();
        item.serialize_compressed(&mut serialized).expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    println!("All values successfully written to {}", file_path);
}

fn main() {
    initialize_all();
}
