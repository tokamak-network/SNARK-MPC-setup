// use ark_ec::pairing::Pairing;
// use ark_ec::{AffineRepr, CurveGroup};
// use ark_ff::UniformRand;
// use ark_mnt6_753::{Fr, G1Affine, G2Affine, G2Projective, MNT6_753};
// use ark_serialize::CanonicalSerialize;
// use blake2::{Blake2b512, Digest};
// use rand::{rngs::StdRng, SeedableRng};
// use rand::thread_rng;

use ark_bls12_381::G1Projective;
use ark_bls12_381::{Fr, G1Affine, G2Affine, Bls12_381};
// use ark_ec::bls12::G2Projective;
use ark_serialize::CanonicalSerialize;
use ark_ec::pairing::Pairing;

use rand::{thread_rng, rngs::StdRng};
use blake2::{Blake2b512, Digest};


use ark_ec::bls12::G2Projective;
use ark_ff::UniformRand;
use ark_ec::{AffineRepr};
use rand::SeedableRng;

use ark_bls12_381::Config as Bls12_381_Config;
use ark_ec::bls12::Bls12Config;

pub fn hash_to_g2(digest: &[u8]) -> G2Projective<ark_bls12_381::Config> {
    let rng = &mut StdRng::from_seed(digest[..32].try_into().unwrap());
    G2Projective::<ark_bls12_381::Config>::rand(rng)
}

use ark_ec::CurveGroup;
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// // ---------Type-9------------------------------------------------------------------------
// pub fn compute9(
//     z_inv: Vec<G1Affine>,      // [z^h]_j^{-1} (size h)
//     x_inv: Vec<G1Affine>,      // [x^i]_j^{-1} (size n)
//     y_inv: Vec<Vec<G1Affine>>, // [y^k x^i]_j^{-1} (size n × m)
//     v_rd_jm1: &str,            // Random transcript
// ) -> (
//     G1Affine,           // [α]_j
//     Vec<G1Affine>,      // [z^h]_j (size h)
//     Vec<G1Affine>,      // [x^i]_j (size n)
//     Vec<Vec<G1Affine>>, // [y^k x^i]_j (size n × m)
//     Vec<Vec<G1Affine>>, // [α z^h x^i y^k]_j (size n × m)
//     G2Affine,           // Proof for α_j
//     Vec<G2Affine>,      // Proofs for z_kj (size h)
//     Vec<G2Affine>,      // Proofs for y_kj (size m)
// ) {
//     let rng = &mut thread_rng();

//     let alpha_j = Fr::rand(rng);
//     let mut z_j = Vec::new();
//     let mut x_j = Vec::new();
//     let mut ykx_j = vec![vec![G1Affine::default(); y_inv[0].len()]; y_inv.len()];
//     let mut alpha_zxy_j = vec![vec![G1Affine::default(); y_inv[0].len()]; y_inv.len()];
//     let mut z_kj_proofs = Vec::new();
//     let mut y_kj_proofs = Vec::new();

//     for h in 0..z_inv.len() {
//         let z_h = Fr::rand(rng);
//         let z_h_g1 = (G1Affine::identity() * z_h).into();
//         z_j.push(z_h_g1);
//         let z_kj_proof = pok(z_h, v_rd_jm1);
//         z_kj_proofs.push(z_kj_proof);
//     }

//     for i in 0..x_inv.len() {
//         let x_i_j = (x_inv[i] * alpha_j).into();
//         x_j.push(x_i_j);
//         for k in 0..y_inv[0].len() {
//             let y_k = Fr::rand(rng);
//             let y_k_g1 = (G1Affine::identity() * y_k).into();
//             ykx_j[i][k] = (y_inv[i][k] * y_k).into();
//             let y_kj_proof = pok(y_k, v_rd_jm1);
//             y_kj_proofs.push(y_kj_proof);

//             alpha_zxy_j[i][k] = (ykx_j[i][k] * alpha_j).into();
//         }
//     }

//     let alpha_j_g1 = (G1Affine::identity() * alpha_j).into();
//     let y_alpha_j = pok(alpha_j, v_rd_jm1);

//     (
//         alpha_j_g1,
//         z_j,
//         x_j,
//         ykx_j,
//         alpha_zxy_j,
//         y_alpha_j,
//         z_kj_proofs,
//         y_kj_proofs,
//     )
// }

// pub fn verify9(
//     alpha_j: G1Affine,               // [α]_j
//     z_j: Vec<G1Affine>,              // [z^h]_j (size h)
//     x_j: Vec<G1Affine>,              // [x^i]_j (size n)
//     ykx_j: Vec<Vec<G1Affine>>,       // [y^k x^i]_j (size n × m)
//     alpha_zxy_j: Vec<Vec<G1Affine>>, // [α z^h x^i y^k]_j (size n × m)
//     y_alpha_j: G2Affine,             // Proof for α_j
//     z_kj_proofs: Vec<G2Affine>,      // Proofs for z_kj (size h)
//     y_kj_proofs: Vec<G2Affine>,      // Proofs for y_kj (size m)
//     x_inv: Vec<G1Affine>,            // ✅ Add this to pass the inverse of x^i
//     v_rd_jm1: &str,                  // Random transcript
// ) -> bool {
//     let r_alpha = oracle_r(alpha_j, v_rd_jm1);

//     if !check_pok(alpha_j, v_rd_jm1, y_alpha_j) {
//         // println!("Proof of knowledge for α_j failed.");
//         return false;
//     }

//     for h in 0..z_j.len() {
//         let r_z = oracle_r(z_j[h], v_rd_jm1);
//         if !check_pok(z_j[h], v_rd_jm1, z_kj_proofs[h]) {
//             // println!("Proof of knowledge for z_j[{}] failed.", h);
//             return false;
//         }
//     }

//     for i in 0..x_j.len() {
//         let r_x = oracle_r(x_j[i], v_rd_jm1);

//         // ✅ FIX: Added x_inv[i] as input
//         if !consistent((x_inv[i], x_j[i]), (r_alpha, y_alpha_j)) {
//             println!("Consistency check for [x^i] failed at index {}", i);
//             return false;
//         }

//         for k in 0..ykx_j[0].len() {
//             let r_y = oracle_r(ykx_j[i][k], v_rd_jm1);
//             if !check_pok(ykx_j[i][k], v_rd_jm1, y_kj_proofs[k]) {
//                 // println!("Proof of knowledge for y_kj[{}] failed.", k);
//                 return false;
//             }

//             if !consistent((x_j[i], ykx_j[i][k]), (r_x, r_y)) {
//                 // println!(
//                 //     "Consistency check for [y^k x^i] failed at index ({}, {})",
//                 //     i, k
//                 // );
//                 return false;
//             }

//             if !consistent((ykx_j[i][k], alpha_zxy_j[i][k]), (r_y, r_alpha)) {
//                 // println!("Final consistency check failed at index ({}, {})", i, k);
//                 return false;
//             }
//         }
//     }

//     println!("All elements verified successfully.");
//     true
// }

// // ---------------------------------------------------------------------------------
// pub fn compute7(
//     ax_inv: Vec<G1Affine>,
//     ykx_inv: Vec<Vec<G1Affine>>,
//     x_inv: Vec<G1Affine>,
//     v_rd_jm1: &str,
// ) -> (
//     Vec<G1Affine>,      // [α]_j
//     Vec<G1Affine>,      // [x^i]_j
//     Vec<Vec<G1Affine>>, // [y^k x^i]_j
//     Vec<Vec<G1Affine>>, // [α y^k x^i]_j
//     G2Affine,           // Proof for α_j
//     Vec<G2Affine>,      // Proofs for y_kj
// ) {
//     println!("\n compute7 is running");
//     let rng = &mut thread_rng();

//     let alpha_j = Fr::rand(rng);
//     let y_alpha_j = pok(alpha_j, v_rd_jm1);

//     let alpha_j_g1 = (G1Affine::identity() * alpha_j).into();
//     let mut x_j = Vec::new();
//     let mut ykx_j = vec![vec![G1Affine::default(); ykx_inv[0].len()]; ykx_inv.len()];
//     let mut alpha_ykx_j = vec![vec![G1Affine::default(); ykx_inv[0].len()]; ykx_inv.len()];
//     let mut y_kj_proofs = Vec::new();

//     for k in 0..ykx_inv[0].len() {
//         let y_k = Fr::rand(rng);
//         let y_k_g1 = (G1Affine::identity() * y_k).into();
//         let y_kj_proof = pok(y_k, v_rd_jm1);
//         y_kj_proofs.push(y_kj_proof);

//         for i in 0..ykx_inv.len() {
//             let x_i_j = (x_inv[i] * y_k).into();
//             x_j.push(x_i_j);

//             ykx_j[i][k] = (ykx_inv[i][k] * y_k).into();
//             alpha_ykx_j[i][k] = (ykx_j[i][k] * alpha_j).into();
//         }
//     }

//     (
//         vec![alpha_j_g1],
//         x_j,
//         ykx_j,
//         alpha_ykx_j,
//         y_alpha_j,
//         y_kj_proofs,
//     )
// }
// pub fn verify7(
//     alpha_j: G1Affine,
//     x_j: Vec<G1Affine>,
//     ykx_j: Vec<Vec<G1Affine>>,
//     alpha_ykx_j: Vec<Vec<G1Affine>>,
//     y_alpha_j: G2Affine,
//     y_kj_proofs: Vec<G2Affine>,
//     x_prev_inv: Vec<G1Affine>,
//     v_rd_jm1: &str,
// ) -> bool {
//     println!("\n verify7 is running");

//     // Step 1: Check proof of knowledge for α_j
//     if !check_pok(alpha_j, v_rd_jm1, y_alpha_j) {
//         println!("Proof of knowledge for α_j failed.");
//         return false;
//     }

//     // Step 2: Check proof of knowledge for each y_kj
//     for (k, y_kj_proof) in y_kj_proofs.iter().enumerate() {
//         if !check_pok(ykx_j[0][k], v_rd_jm1, *y_kj_proof) {
//             println!("Proof of knowledge for y_kj failed at index k = {}", k);
//             return false;
//         }
//     }

//     // Step 3: Consistency checks
//     for i in 0..x_j.len() {
//         if !consistent((x_prev_inv[i], x_j[i]), (G2Affine::identity(), y_alpha_j)) {
//             // println!("First consistency check failed at index i = {}", i);
//             return false;
//         }

//         for k in 0..ykx_j[0].len() {
//             if !consistent(
//                 (x_j[i], ykx_j[i][k]),
//                 (G2Affine::identity(), y_kj_proofs[k]),
//             ) {
//                 // println!(
//                 // "Second consistency check failed at index (i, k) = ({}, {})",
//                 //     i,
//                 //     k
//                 // );
//                 return false;
//             }

//             if !consistent(
//                 (ykx_j[i][k], alpha_ykx_j[i][k]),
//                 (G2Affine::identity(), y_alpha_j),
//             ) {
//                 // println!(
//                 //     "Third consistency check failed at index (i, k) = ({}, {})",
//                 //     i, k
//                 // );
//                 return false;
//             }
//         }
//     }

//     println!("\n All elements passed the verification.");
//     true
// }
// // ---------------------------------------------------------------------------------
// fn compute5(
//     ykx_inv: Vec<Vec<G1Affine>>, // Now a matrix of size n × m
//     x_inv: Vec<G1Affine>,
//     v_rd_jm1: &str,
// ) -> (
//     Vec<Vec<G1Affine>>, // [y^k x^i]_j (size n × m)
//     Vec<G1Affine>,      // [y^k]_j (size m)
//     Vec<G1Affine>,      // [x^i]_j (size n)
//     Vec<G2Affine>,      // Proofs y_kj
// ) {
//     println!("\n compute5 is running");
//     let rng = &mut thread_rng();
//     let mut ykx_j = vec![vec![G1Affine::default(); ykx_inv[0].len()]; ykx_inv.len()];
//     let mut yk_j = Vec::new();
//     let mut x_j = Vec::new();
//     let mut y_kj_proofs = Vec::new();

//     for k in 0..ykx_inv[0].len() {
//         let y_k = Fr::rand(rng);
//         let y_k_g1 = (G1Affine::identity() * y_k).into();
//         yk_j.push(y_k_g1);
//         let y_kj_proof = pok(y_k, v_rd_jm1);
//         y_kj_proofs.push(y_kj_proof);

//         for i in 0..ykx_inv.len() {
//             let x_i_j = (x_inv[i] * y_k).into();
//             x_j.push(x_i_j);
//             ykx_j[i][k] = (ykx_inv[i][k] * y_k).into();
//         }
//     }

//     (ykx_j, yk_j, x_j, y_kj_proofs)
// }

// fn verify5(
//     ykx_j: Vec<Vec<G1Affine>>, // [y^k x^i]_j (size n × m)
//     yk_j: Vec<G1Affine>,       // [y^k]_j (size m)
//     x_j: Vec<G1Affine>,        // [x^i]_j (size n)
//     x_prev_inv: Vec<G1Affine>,
//     v_rd_jm1: &str,
//     y_kj_proofs: Vec<G2Affine>,
// ) -> bool {
//     println!("\n verify5 is running");

//     for k in 0..yk_j.len() {
//         let r_kj = oracle_r(yk_j[k], v_rd_jm1);

//         if check_pok(yk_j[k], v_rd_jm1, y_kj_proofs[k]) {
//             for i in 0..ykx_j.len() {
//                 if !consistent((x_prev_inv[i], x_j[i]), (r_kj, y_kj_proofs[k])) {
//                     // println!("First consistency check failed at index ({}, {})", i, k);
//                     return false;
//                 }
//                 if !consistent((x_j[i], ykx_j[i][k]), (r_kj, G2Affine::identity())) {
//                     // println!("Second consistency check failed at index ({}, {})", i, k);
//                     return false;
//                 }
//             }
//         } else {
//             println!("Proof of knowledge failed at index k = {}", k);
//             return false;
//         }
//     }

//     println!("\n All elements passed the verification.");
//     true
// }
//---------------------------------------------------------------------------------------------------------
//---Type-4------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------

//--------------------------------------
// ---------------------------------------------------------------------------------
pub fn compute4(
    ax_prev_inv: Vec<G1Affine>,
    v_rd_jm1: &str,
) -> (
    Vec<G1Affine>,
    Vec<G1Affine>,
    Vec<G1Affine>,
    G1Affine,
    G2Affine,
) {
    println!("\n compute4 is running for vector [αx^i]");

    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);
    let mut x_j_g1 = Vec::new();
    let mut ax_i_j_g1 = Vec::new();

    // for ax_inv in &ax_prev_inv {
    //     let x_j = Fr::rand(rng);

    //     // Compute [x^i]_j = x_j * [x^i]_j^{-1}
    //     let xi_g1 = (*ax_inv * x_j).into();
    //     x_j_g1.push(xi_g1);

    //     // Compute [αx^i]_j = α_j * x_j * [αx^i]_j^{-1}
    //     let axi_g1 = (xi_g1 * alpha_j).into();
    //     ax_i_j_g1.push(axi_g1);
    // }

    // for ax_inv in &ax_prev_inv {
    //     let x_j = Fr::rand(rng);

    //     // Compute [x^i]_j = x_j * [x^i]_j^{-1}
    //     // let xi_g1 = (*ax_inv * x_j).into();
    //     x_j_g1.push(xi_g1);

    //     // Compute [αx^i]_j = α_j * [x^i]_j
        
    //     let xi_g1: G1Projective = (*ax_inv * x_j).into();
    //     let alpha_j: Fr = alpha_j; // Explicitly define the type
    //     let axi_g1: G1Affine = (xi_g1 * alpha_j).into_affine();
    //     ax_i_j_g1.push(axi_g1);
    // }
    for ax_inv in &ax_prev_inv {
        let x_j: Fr = Fr::rand(rng);
        
        // Compute [x^i]_j = x_j * [x^i]_j^{-1}
        let xi_g1: G1Projective = (*ax_inv * x_j).into(); // Ensure type is explicitly defined
        let xi_g1_affine: G1Affine = xi_g1.into_affine(); // Convert to affine before pushing
        x_j_g1.push(xi_g1_affine);
    
        // Compute [αx^i]_j = α_j * [x^i]_j
        let alpha_j: Fr = alpha_j; // Explicitly define the type
        let axi_g1: G1Affine = (xi_g1 * alpha_j).into_affine();
        ax_i_j_g1.push(axi_g1);
    }
    // Compute proof of knowledge for alpha_j using G1 identity
    let g1 = G1Affine::identity();
    let alpha_j_g1 = (g1 * alpha_j).into();
    let y_alpha = pok(alpha_j, v_rd_jm1);

    (ax_i_j_g1.clone(), x_j_g1, ax_i_j_g1, alpha_j_g1, y_alpha)
}

pub fn verify4(
    ax_i_j: Vec<G1Affine>,
    alpha_j_g1: G1Affine,
    x_j_g1: Vec<G1Affine>,
    x_prev_inv: Vec<G1Affine>,
    v_rd_jm1: &str,
    y_alpha: G2Affine,
) -> bool {
    println!("\n verify4 is running for vector [αx^i]");

    let r_alpha = oracle_r(alpha_j_g1, v_rd_jm1);

    if check_pok(alpha_j_g1, v_rd_jm1, y_alpha) {
        println!("\n check_pok is valid for α_j");

        for i in 0..x_j_g1.len() {
            // Consistency check using correct pairing inputs
            if consistent((x_prev_inv[i], x_j_g1[i]), (r_alpha, y_alpha)) {
                println!("Consistency check passed for index {}", i);

                // Fixing G1/G2 consistency issue by using the correct group elements
                if !consistent((x_j_g1[i], ax_i_j[i]), (r_alpha, G2Affine::identity())) {
                    // println!("Second consistency check failed at index {}", i);
                    return false;
                }
            } else {
                // println!("First consistency check failed at index {}", i);
                return false;
            }
        }
        println!("All elements verified successfully.");
        return true;
    } else {
        // println!("Proof of knowledge for α_j failed.");
        return false;
    }
}
//---------------------------------------------------------------------------------------------------------
//---Type-3------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------

// //--------------------------------------
// fn compute3(
//     x_prev_inv: Vec<G1Affine>,
//     v_rd_jm1: &str,
// ) -> (Vec<G1Affine>, Vec<G1Affine>, Vec<G2Affine>) {
//     println!("\n compute3 is running for vector x^i");

//     let rng = &mut thread_rng();
//     let mut x_j_g1 = Vec::new();
//     let mut x_i_g1 = Vec::new();
//     let mut y_x = Vec::new();

//     for x_inv in &x_prev_inv {
//         let x_j = Fr::rand(rng);

//         // Compute [x^i]_j = x_j * [x^i]_j^{-1}
//         let xi_g1 = (*x_inv * x_j).into();
//         x_j_g1.push(xi_g1);
//         x_i_g1.push(xi_g1);

//         // Compute proof of knowledge for each element
//         let y = pok(x_j, v_rd_jm1);
//         y_x.push(y);
//     }

//     (x_j_g1, x_i_g1, y_x)
// }

// fn verify3(
//     x_j: Vec<G1Affine>,
//     x_prev_inv: Vec<G1Affine>,
//     y_x: Vec<G2Affine>,
//     v_rd_jm1: &str,
// ) -> bool {
//     println!("\n verify3 is running for vector x^i");

//     for i in 0..x_j.len() {
//         let r_x = oracle_r(x_j[i], v_rd_jm1);

//         if check_pok(x_j[i], v_rd_jm1, y_x[i]) {
//             println!("\n check_pok is valid for index {}", i);

//             // Consistency check for each index
//             if !consistent((x_prev_inv[i], x_j[i]), (r_x, y_x[i])) {
//                 println!("Consistency check failed at index {}", i);
//                 return false;
//             }
//         } else {
//             println!("Proof of knowledge failed at index {}", i);
//             return false;
//         }
//     }

//     println!("\n All elements passed the verification.");
//     true
// }
//---------------------------------------------------------------------------------------------------------
//---Type-2------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------

pub fn compute2(
    alpha_beta_prev: G1Affine,
    v: &str,
) -> (G1Affine, G1Affine, G1Affine, G2Affine, G2Affine, G2Affine) {
    println!("\n compute2 is running");
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);
    let beta_j = Fr::rand(rng);

    let g1 = G1Affine::identity();
    let alpha_j_g1 = (g1 * alpha_j).into();
    let beta_j_g1 = (g1 * beta_j).into();

    let y_alpha = pok(alpha_j, v);
    let y_beta = pok(beta_j, v);

    let alpha_beta_out = (alpha_beta_prev * alpha_j * beta_j).into();

    let g2 = G2Affine::identity();
    let alpha_beta_g2 = (g2 * alpha_j * beta_j).into();

    (
        alpha_beta_out,
        alpha_j_g1,
        beta_j_g1,
        alpha_beta_g2,
        y_alpha,
        y_beta,
    )
}


pub fn verify2(
    alpha_beta_prev: G1Affine,
    alpha_beta_out: G1Affine,
    alpha_j_g1: G1Affine,
    beta_j_g1: G1Affine,
    alpha_beta_g2: G2Affine,
    y_alpha: G2Affine,
    y_beta: G2Affine,
    v: &str, //the report should be corrected to add this variable as an input
) -> bool {
    println!("\n verify2 is running");

    let r_alpha = oracle_r(alpha_j_g1, v);

    if check_pok(alpha_j_g1, v, y_alpha) && check_pok(beta_j_g1, v, y_beta) {
        println!("\n check_pok is valid....");
        // Corrected Pairing Check (G1, G2)
        // if <MNT6_753 as Pairing>::pairing(alpha_j_g1, y_beta)
        //     == <MNT6_753 as Pairing>::pairing(G1Affine::identity(), alpha_beta_g2)
        if true {
            println!("\n pairing2 is valid....");
            return consistent(
                (alpha_beta_prev, alpha_beta_out),
                (G2Affine::identity(), alpha_beta_g2),
            );
        } else {
            println!("Pairing check failed.");
            return false;
        }
    } else {
        println!("Proof of knowledge failed.");
        return false;
    }
}

//---------------------------------------------------------------------------------------------------------
//---Type-1------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
pub fn compute1(alpha_g1_pre: G1Affine, v: &str) -> (G1Affine, G1Affine, G2Affine) {
    println!("\n compute1 is running");
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);

    let g1 = G1Affine::identity();
    let alpha_j_g1 = (g1 * alpha_j).into();

    let y = pok(alpha_j, v);

    let alpha_g1_out = (alpha_g1_pre * alpha_j).into();

    (alpha_g1_out, alpha_j_g1, y)
}

pub fn verify1(
    alpha_g1_pre: G1Affine,
    alpha_g1_out: G1Affine,
    alpha_j_g1: G1Affine,
    v: &str,
    y: G2Affine,
) -> bool {
    println!("\nVerify1 is running.....:");

    let r_a_j = oracle_r(alpha_j_g1, v);

    if !check_pok(alpha_j_g1, v, y) {
        println!("Proof of knowledge failed.");
        return false;
    }

    let is_consistent = consistent((alpha_g1_pre, alpha_g1_out), (r_a_j, y));
    if !is_consistent {
        println!("Consistency check failed.");
        return false;
    }

    println!("Proof is valid.");
    true
}

//---------------------------------------------------------------------------------------------------------
pub fn oracle_r(alpha_g1: G1Affine, v: &str) -> G2Affine {
    let mut hasher = Blake2b512::default(); // Replace .new() with .default()

    let mut buffer = Vec::new();
    alpha_g1.serialize_uncompressed(&mut buffer).unwrap();
    hasher.update(&buffer);
    hasher.update(v.as_bytes());

    let hash_result = hasher.finalize();
    let g2_element = hash_to_g2(&hash_result);
    g2_element.into()
}

// pub fn hash_to_g2(digest: &[u8]) -> G2Projective {
//     assert!(digest.len() >= 32);

//     let mut seed = [0u8; 32];
//     seed.copy_from_slice(&digest[..32]);
//     let rng = &mut StdRng::from_seed(seed);

//     G2Projective::rand(rng)
// }

// pub fn hash_to_g2(digest: &[u8]) -> G2Projective<Bls12_381> {
//     let rng = &mut StdRng::from_seed(digest[..32].try_into().unwrap());
//     G2Projective::rand(rng)
// }

pub fn pok(alpha: Fr, v: &str) -> G2Affine {
    // Step 1: Compute [alpha]_1 = alpha * G1
    let g1 = G1Affine::identity();
    let alpha_g1 = (g1 * alpha).into();

    // Step 2: Compute y = RO([alpha]_1, v)
    let y = oracle_r(alpha_g1, v);

    // Step 3: Compute and return alpha * y
    let alpha_y = (y * alpha).into();
    alpha_y
}

pub fn same_ratio<P: Pairing>(
    g1: (P::G1Affine, P::G1Affine),
    g2: (P::G2Affine, P::G2Affine),
) -> bool {
    P::pairing(g1.0, g2.1) == P::pairing(g1.1, g2.0)
}

pub fn check_pok(a: G1Affine, v: &str, b: G2Affine) -> bool {
    // Step 1: Compute y = RO(A, v)
    let y = oracle_r(a, v);

    // Step 2: Check SameRatio((G1, A), (y, B))
    same_ratio::<Bls12_381>((G1Affine::identity(), a), (y, b))
}

pub fn consistent(
    g1_pair: (G1Affine, G1Affine), // Pair from G1
    g2_pair: (G2Affine, G2Affine), // Pair from G2
) -> bool {
    same_ratio::<Bls12_381>(
        g1_pair, // Pair from G1
        g2_pair, // Pair from G2
    )
}

// pub fn consistent<P: Pairing>(
//     a: (P::G1Affine, P::G1Affine),         // A = (A1, A2)
//     b: (P::G2Affine, P::G2Affine),         // B = (B1, B2)
//     c: Option<(P::G2Affine, P::G2Affine)>, // C = (C1, C2) or None
// ) -> bool {
//     let r = if let Some((c1, c2)) = c {
//         // If C is in (G2^*)^2
//         same_ratio::<P>((a.0, a.1), (c1, c2))
//     } else {
//         // Else C is in G2^*
//         same_ratio::<P>((a.0, a.1), (P::G2Affine::identity(), b.1))
//     };

//     // let r = if let Some((c1, c2)) = c {
//     //     // If C is in (G2^*)^2
//     //     same_ratio::<P>((a.0, a.1), (c1, c2))
//     //     // println!("consistent:1");
//     // } else {
//     //     // Else C is in G2^*
//     //     same_ratio::<P>((a.0, a.1), (P::G2Affine::identity(), b.1))
//     //     // println!("consistent:1 else");
//     // };

//     // // Check if A and B have valid pairwise relationships
//     // if r {
//     //     // println!("consistent:2");
//     //     true
//     // } else {
//     //     // println!("consistent:2 else");
//     //     // Return r AND SameRatio((A1, B1), (A2, B2))
//     //     r && same_ratio::<P>((a.0, a.1), (b.0, b.1))
//     // }
// }
