// use ark_ec::AffineRepr;
// use ark_mnt6_753::{G1Affine, G2Affine};
// use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
// use deneme::{compute1, compute2, verify1, verify2};
// use std::fs::OpenOptions;
// use std::io::{BufWriter, Read, Write};

use ark_bls12_381::{G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use deneme::{compute1, compute2, verify1, verify2};


use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;

pub fn verify_and_update() -> bool {
    println!("Verifying previous proofs...");

    let rnd_string = "rndString";

    // 1️⃣ Read Previous Accumulators
    if let Some((g1_values, g2_values)) = read_combined_values("accumulator_all_types.bin", 7, 4) {
        let [alpha_g1_pre, alpha_g1_out, alpha_j_g1,
             alpha_beta_prev, alpha_beta_out, alpha_j_g1_2, beta_j_g1]: [G1Affine; 7] = 
             g1_values.try_into().unwrap();

        let [y, alpha_beta_g2, y_alpha, y_beta]: [G2Affine; 4] = 
            g2_values.try_into().unwrap();

        // 2️⃣ Verify Type-1 (α)
        if !verify1(alpha_g1_pre, alpha_g1_out, alpha_j_g1, rnd_string, y) {
            println!("❌ Type-1 (α) verification failed.");
            return false;
        }

        // 3️⃣ Verify Type-2 (α * β)
        if !verify2(
            alpha_beta_prev, alpha_beta_out, alpha_j_g1_2, beta_j_g1,
            alpha_beta_g2, y_alpha, y_beta, rnd_string,
        ) {
            println!("❌ Type-2 (α * β) verification failed.");
            return false;
        }

        println!("✅ All verifications passed.");

        // 4️⃣ Compute New Random Parameters
        let (new_alpha_g1_out, new_alpha_j_g1, new_y) = compute1(alpha_g1_pre, rnd_string);
        let (new_alpha_beta_out, new_alpha_j_g1_2, new_beta_j_g1, new_alpha_beta_g2, new_y_alpha, new_y_beta) =
            compute2(alpha_beta_prev, rnd_string);

        // 5️⃣ Update Combined File with New Parameters
        save_to_combined_file(
            "accumulator_all_types.bin",
            &[
                &alpha_g1_pre, &new_alpha_g1_out, &new_alpha_j_g1, // Type-1 (G1)
                &alpha_beta_prev, &new_alpha_beta_out, &new_alpha_j_g1_2, &new_beta_j_g1, // Type-2 (G1)
            ],
            &[
                &new_y, &new_alpha_beta_g2, &new_y_alpha, &new_y_beta, // All G2
            ],
        );

        println!("✅ Accumulator file updated with new random parameters.");
        true
    } else {
        println!("❌ Failed to read accumulator file.");
        false
    }
}

fn read_combined_values(
    file_path: &str,
    g1_count: usize,
    g2_count: usize,
) -> Option<(Vec<G1Affine>, Vec<G2Affine>)> {
    let mut file = std::fs::File::open(file_path).ok()?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).ok()?;

    let mut cursor = &buffer[..];

    let g1_values = (0..g1_count)
        .filter_map(|_| G1Affine::deserialize_compressed(&mut cursor).ok())
        .collect::<Vec<_>>();

    let g2_values = (0..g2_count)
        .filter_map(|_| G2Affine::deserialize_compressed(&mut cursor).ok())
        .collect::<Vec<_>>();

    if g1_values.len() == g1_count && g2_values.len() == g2_count {
        Some((g1_values, g2_values))
    } else {
        None
    }
}

fn save_to_combined_file<T: CanonicalSerialize, U: CanonicalSerialize>(
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
    if verify_and_update() {
        println!("✅ Verification and update completed successfully!");
    } else {
        println!("❌ Verification or update failed.");
    }
}
