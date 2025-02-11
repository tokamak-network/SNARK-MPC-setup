use ark_ec::AffineRepr;
use ark_mnt6_753::{G1Affine, G2Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use deneme::{compute1, verify1};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};

pub fn verifyCompute() -> bool {
    println!("\nVerifying previous proof and generating new one...");
    let rnd_string = "rndString";
    // Read previous values from accumulator
    match read_previous_values() {
        Some((alpha_g1_pre, alpha_g1_out, alpha_j_g1, y)) => {
            // Verify previous proof using stored y value
            let is_valid = verify1(alpha_g1_pre, alpha_g1_out, alpha_j_g1, rnd_string, y);

            if !is_valid {
                println!("Previous proof verification failed. Aborting computation.");
                return false;
            }

            // Compute new proof
            let (new_alpha_g1_out, new_alpha_j_g1, new_y) = compute1(alpha_g1_out, rnd_string);
            // println!("New computed alpha: {:?}", new_alpha_g1_out);

            // Save new values to accumulator
            save_alpha_to_file(&new_alpha_g1_out);
            save_alpha_to_file(&new_alpha_j_g1);
            save_alpha_to_file(&new_y);

            true
        }
        None => {
            println!("Error: No previous proof found. Please run initialize.rs first.");
            false
        }
    }
}

fn save_alpha_to_file<T: CanonicalSerialize>(data: &T) {
    let file_path = "resp_accumulator.bin";
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .expect("Unable to open or create accumulator.bin");

    let mut writer = BufWriter::new(&mut file);
    let mut serialized = Vec::new();
    data.serialize_compressed(&mut serialized)
        .expect("Failed to serialize data");

    writer
        .write_all(&serialized)
        .expect("Unable to write data to file");
    println!("Alpha saved to accumulator.");
}

fn read_previous_values() -> Option<(G1Affine, G1Affine, G1Affine, G2Affine)> {
    let file_path = "accumulator.bin";

    // Check if the file exists
    let mut file = File::open(file_path).ok()?;
    let mut buffer = Vec::new();
    if file.read_to_end(&mut buffer).is_err() || buffer.is_empty() {
        println!("Error: Could not read from accumulator file or file is empty.");
        return None;
    }

    let mut cursor = &buffer[..];

    let alpha_g1_pre = G1Affine::deserialize_compressed(&mut cursor).ok()?;
    let alpha_g1_out = G1Affine::deserialize_compressed(&mut cursor).ok()?;
    let alpha_j_g1 = G1Affine::deserialize_compressed(&mut cursor).ok()?;
    let y = G2Affine::deserialize_compressed(&mut cursor).ok()?;

    Some((alpha_g1_pre, alpha_g1_out, alpha_j_g1, y))
}

fn main() {
    let _ = verifyCompute();
}
