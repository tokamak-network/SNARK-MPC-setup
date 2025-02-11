use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_mnt6_753::{Fr, G1Affine, G2Affine};
use ark_serialize::CanonicalSerialize;
use deneme::compute1;
use rand::thread_rng;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

pub fn initialize_alpha() {
    println!("Initializing first alpha...");

    let rnd_string = "rndString"; // Ensuring consistent input string for compute functions
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);
    let alpha_g1_pre = G1Affine::generator();

    // Compute the initial proof using the same input string
    let (alpha_g1_out, alpha_j_g1, y) = compute1(alpha_g1_pre, rnd_string);

    // Save output values for comVer
    save_alpha_to_file(&alpha_g1_pre);
    save_alpha_to_file(&alpha_g1_out);
    save_alpha_to_file(&alpha_j_g1);
    save_alpha_to_file(&y);
}

fn save_alpha_to_file<T: CanonicalSerialize>(data: &T) {
    let file_path = "accumulator.bin"; // Ensure correct filename
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true) // Change from truncate to append
        .open(file_path)
        .expect("Unable to open or create accumulator.bin");

    let mut writer = BufWriter::new(&mut file);
    let mut serialized = Vec::new();
    data.serialize_compressed(&mut serialized)
        .expect("Failed to serialize data");

    writer
        .write_all(&serialized)
        .expect("Unable to write data to file");
    println!("Initial parameters saved to accumulator for comVer.");
}

fn main() {
    initialize_alpha();
}
