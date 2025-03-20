// Final verifyCompute.rs with all compute and verify functions and missing utility functions

use ark_bls12_381::{Fr, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use deneme::{
    compute1, compute2, compute5, compute7, compute9, verify1, verify2, verify5, verify7, verify9,
};
use rand::thread_rng;

use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;

pub fn verify_and_update() -> bool {
    println!("Verifying previous proofs...");

    let rnd_string = "rndString";

    // 1️⃣ Read Previous Accumulators
    if let Some((g1_values, g2_values)) = read_combined_values("accumulator_all_types.bin", 22, 9) {
        let [alpha_g1_pre, alpha_g1_out, alpha_j_g1,
             alpha_beta_prev, alpha_beta_out, alpha_j_g1_2, beta_j_g1,
             ax_i_j_g1_1, x_j_g1, ax_i_j_g1_2,
             ykx_j_00, yk_j_0, x_j_0,
             alpha_j_g1_7, x_j_7, ykx_j_7, alpha_ykx_j_7,
             alpha_j_g1_9, z_j_9, x_j_9, ykx_j_9, alpha_zxy_j_9]: [G1Affine; 22] =
             g1_values.try_into().unwrap();

        let [y, alpha_beta_g2, y_alpha, y_beta, y_kj_proof_0, y_alpha_j_7, y_kj_proof_7, y_alpha_j_9, z_kj_proof_9]: [G2Affine;
            9] = g2_values.try_into().unwrap();

        // 2️⃣ Verify Type-1 (α)
        if !verify1(alpha_g1_pre, alpha_g1_out, alpha_j_g1, rnd_string, y) {
            println!("❌ Type-1 (α) verification failed.");
            return false;
        }

        // 3️⃣ Verify Type-2 (α * β)
        if !verify2(
            alpha_beta_prev,
            alpha_beta_out,
            alpha_j_g1_2,
            beta_j_g1,
            alpha_beta_g2,
            y_alpha,
            y_beta,
            rnd_string,
        ) {
            println!("❌ Type-2 (α * β) verification failed.");
            return false;
        }

        // 4️⃣ Verify Type-5 ([y^k x^i]_j)
        if !verify5(
            vec![vec![ykx_j_00]],          // Type-5 (n × m matrix)
            vec![yk_j_0],                  // [y^k]_j
            vec![x_j_0],                   // [x^i]_j
            vec![G1Affine::identity(); 1], // Dummy previous inverses
            rnd_string,
            vec![y_kj_proof_0], // Proofs y_kj
        ) {
            println!("❌ Type-5 ([y^k x^i]_j) verification failed.");
            return false;
        }

        // 5️⃣ Verify Type-7 (α y^k x^i)
        if !verify7(
            alpha_j_g1_7,
            vec![x_j_7],
            vec![vec![ykx_j_7]],
            vec![vec![alpha_ykx_j_7]],
            y_alpha_j_7,
            vec![y_kj_proof_7],
            vec![G1Affine::identity(); 1], // Dummy previous inverses
            rnd_string,
        ) {
            println!("❌ Type-7 (α y^k x^i) verification failed.");
            return false;
        }

        // 6️⃣ Verify Type-9 (α z^h x^i y^k)
        if !verify9(
            alpha_j_g1_9,
            vec![z_j_9],                   // [z^h]_j
            vec![x_j_9],                   // [x^i]_j
            vec![vec![ykx_j_9]],           // [y^k x^i]_j (n × m)
            vec![vec![alpha_zxy_j_9]],     // [α z^h x^i y^k]_j (n × m)
            y_alpha_j_9,                   // Proof for α_j
            vec![z_kj_proof_9],            // Proofs for z_kj (size h)
            vec![y_kj_proof_0],            // Proofs for y_kj (size m)
            vec![G1Affine::identity(); 1], // Dummy previous inverses
            rnd_string,
        ) {
            println!("❌ Type-9 (α z^h x^i y^k) verification failed.");
            return false;
        }

        println!("✅ All verifications passed.");

        // 7️⃣ Compute New Random Parameters
        let (new_alpha_g1_out, new_alpha_j_g1, new_y) = compute1(alpha_g1_pre, rnd_string);
        let (
            new_alpha_beta_out,
            new_alpha_j_g1_2,
            new_beta_j_g1,
            new_alpha_beta_g2,
            new_y_alpha,
            new_y_beta,
        ) = compute2(alpha_beta_prev, rnd_string);

        let (new_ykx_j, new_yk_j, new_x_j, new_y_kj_proofs) = compute5(
            vec![vec![G1Affine::identity(); 2]; 2],
            vec![G1Affine::identity(); 2],
            rnd_string,
        );

        let (
            new_alpha_j_g1_7,
            new_x_j_7,
            new_ykx_j_7,
            new_alpha_ykx_j_7,
            new_y_alpha_j_7,
            new_y_kj_proofs_7,
        ) = compute7(
            vec![G1Affine::identity(); 2],
            vec![vec![G1Affine::identity(); 2]; 2],
            vec![G1Affine::identity(); 2],
            rnd_string,
        );

        let (
            new_alpha_j_g1_9,
            new_z_j_9,
            new_x_j_9,
            new_ykx_j_9,
            new_alpha_zxy_j_9,
            new_y_alpha_j_9,
            new_z_kj_proofs_9,
            new_y_kj_proofs_9,
        ) = compute9(
            vec![G1Affine::identity(); 2],
            vec![G1Affine::identity(); 2],
            vec![vec![G1Affine::identity(); 2]; 2],
            rnd_string,
        );

        // 8️⃣ Update Combined File with New Parameters
        save_to_combined_file(
            "accumulator_all_types.bin",
            &[&new_alpha_g1_out, &new_alpha_j_g1],
            &[&new_y],
        );

        println!("✅ Accumulator file updated with new random parameters.");
        true
    } else {
        println!("❌ Failed to read accumulator file.");
        false
    }
}

// Utility Functions

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

    for item in g1_data {
        let mut serialized = Vec::new();
        item.serialize_compressed(&mut serialized)
            .expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    for item in g2_data {
        let mut serialized = Vec::new();
        item.serialize_compressed(&mut serialized)
            .expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    println!("All values successfully written to {}", file_path);
}

// Main Function
fn main() {
    if verify_and_update() {
        println!("✅ Verification and update completed successfully!");
    } else {
        println!("❌ Verification or update failed.");
    }
}
