use ark_bls12_381::{Fr, G1Affine, G2Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_serialize::CanonicalSerialize;
use deneme::{compute1, compute2, compute4, compute5, compute7, compute8, compute9}; // ✅ Added compute8
use rand::thread_rng;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;

pub fn initialize_all() {
    println!("Initializing Type-1 (α), Type-2 (α * β), Type-4 (μ * x^i), Type-5 ([y^k x^i]_j), Type-7 (α y^k x^i), Type-8 (α y^k x^i f(x)), and Type-9 (α z^h x^i y^k) parameters...");
    let rnd_string = "rndString";

    // Type-1 (α)
    let alpha_g1_pre = G1Affine::identity();
    let (alpha_g1_out, alpha_j_g1, y) = compute1(alpha_g1_pre, rnd_string);

    // Type-2 (α * β)
    let alpha_beta_prev = G1Affine::identity();
    let (alpha_beta_out, alpha_j_g1_2, beta_j_g1, alpha_beta_g2, y_alpha, y_beta) =
        compute2(alpha_beta_prev, rnd_string);

    // Type-4 (μ * x^i)
    let ax_prev_inv = vec![G1Affine::identity(); 3]; // Dummy previous inverses
    let (ax_i_j_g1, x_j_g1, ax_i_j_g1_clone, alpha_j_g1, y_alpha_4) =
        compute4(ax_prev_inv.clone(), rnd_string);

    // Type-5 ([y^k x^i]_j)
    let ykx_inv = vec![vec![G1Affine::identity(); 2]; 2]; // Dummy inverses for testing
    let x_inv = vec![G1Affine::identity(); 2];
    let (ykx_j, yk_j, x_j, y_kj_proofs) = compute5(ykx_inv.clone(), x_inv.clone(), rnd_string);

    // Type-7 (α y^k x^i)
    let ax_inv = vec![G1Affine::identity(); 2];
    let (alpha_j_g1_7, x_j_7, ykx_j_7, alpha_ykx_j, y_alpha_j_7, y_kj_proofs_7) =
        compute7(ax_inv.clone(), ykx_inv.clone(), x_inv.clone(), rnd_string);

    // Type-8 (α y^k x^i f(x)) ✅ Added
    let (ykx_j_8, y_j_8, x_j_8, alpha_ykx_j_8, alpha_j_g1_8, y_alpha_j_8, y_kj_proofs_8) =
        compute8(ykx_inv.clone(), yk_j.clone(), x_inv.clone(), rnd_string);

    // Type-9 (α z^h x^i y^k)
    let z_inv = vec![G1Affine::identity(); 2]; // Dummy inverses for testing
    let (
        alpha_j_g1_9,
        z_j,
        x_j_9,
        ykx_j_9,
        alpha_zxy_j_9,
        y_alpha_j_9,
        z_kj_proofs_9,
        y_kj_proofs_9,
    ) = compute9(z_inv.clone(), x_inv.clone(), ykx_inv.clone(), rnd_string);

    // Save all accumulators to a single file
    save_to_single_file(
        "accumulator_all_types.bin",
        &[
            &alpha_g1_pre,
            &alpha_g1_out,
            &alpha_j_g1, // Type-1 (G1)
            &alpha_beta_prev,
            &alpha_beta_out,
            &alpha_j_g1_2,
            &beta_j_g1, // Type-2 (G1)
            &ax_i_j_g1_clone[0],
            &x_j_g1[0],
            &ax_i_j_g1_clone[1], // Type-4 (G1)
            &ykx_j[0][0],
            &yk_j[0],
            &x_j[0], // Type-5 (G1)
            &alpha_j_g1_7[0],
            &x_j_7[0],
            &ykx_j_7[0][0],
            &alpha_ykx_j[0][0], // Type-7 (G1)
            &alpha_j_g1_8,
            &ykx_j_8[0][0],
            &y_j_8[0],
            &x_j_8[0],
            &alpha_ykx_j_8[0][0], // Type-8 (G1)
            &alpha_j_g1_9,
            &z_j[0],
            &x_j_9[0],
            &ykx_j_9[0][0],
            &alpha_zxy_j_9[0][0], // Type-9 (G1)
        ],
        &[
            &y,
            &alpha_beta_g2,
            &y_alpha,
            &y_beta,
            &y_alpha_4, // All G2 elements
            &y_kj_proofs[0],
            &y_alpha_j_7,
            &y_kj_proofs_7[0], // Type-5 and Type-7 (G2)
            &y_alpha_j_8,
            &y_kj_proofs_8[0], // Type-8 (G2)
            &y_alpha_j_9,
            &z_kj_proofs_9[0],
            &y_kj_proofs_9[0], // Type-9 (G2)
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
        item.serialize_compressed(&mut serialized)
            .expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    // Save G2 Elements
    for item in g2_data {
        let mut serialized = Vec::new();
        item.serialize_compressed(&mut serialized)
            .expect("Serialization failed");
        writer.write_all(&serialized).expect("Write failed");
    }

    println!("All values successfully written to {}", file_path);
}

fn main() {
    initialize_all();
}
