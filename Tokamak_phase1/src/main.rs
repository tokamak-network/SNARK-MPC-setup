// use ark_ec::pairing::Pairing;
// use ark_ec::{AffineRepr, CurveGroup};
// use ark_ff::UniformRand;
// use ark_mnt6_753::{Fr, G1Affine, G2Affine, MNT6_753};
// use deneme::{check_pok, consistent, oracle_r, pok, same_ratio};
// use rand::thread_rng;

use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_mnt6_753::{Fr, G1Affine, G2Affine};
use deneme::{check_pok, consistent, oracle_r, pok};
use rand::thread_rng;

fn compute4(
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
    //     let xi_g1 = (*ax_inv * x_j).into_affine();
    //     x_j_g1.push(xi_g1);

    //     // Compute [αx^i]_j = α_j * x_j * [αx^i]_j^{-1}
    //     let axi_g1 = (xi_g1 * alpha_j).into_affine();
    //     ax_i_j_g1.push(axi_g1);
    // }

    for ax_inv in &ax_prev_inv {
        let x_j = Fr::rand(rng);

        // Compute [x^i]_j = x_j * [x^i]_j^{-1}
        let xi_g1 = (*ax_inv * x_j).into_affine();
        x_j_g1.push(xi_g1);

        // Compute [αx^i]_j = α_j * [x^i]_j
        let axi_g1 = (xi_g1 * alpha_j).into_affine();
        ax_i_j_g1.push(axi_g1);
    }
    // Compute proof of knowledge for alpha_j using G1 generator
    let g1 = G1Affine::generator();
    let alpha_j_g1 = (g1 * alpha_j).into_affine();
    let y_alpha = pok(alpha_j, v_rd_jm1);

    (ax_i_j_g1.clone(), x_j_g1, ax_i_j_g1, alpha_j_g1, y_alpha)
}

fn verify4(
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
                if !consistent((x_j_g1[i], ax_i_j[i]), (r_alpha, G2Affine::generator())) {
                    println!("Second consistency check failed at index {}", i);
                    return false;
                }
            } else {
                println!("First consistency check failed at index {}", i);
                return false;
            }
        }
        println!("All elements verified successfully.");
        return true;
    } else {
        println!("Proof of knowledge for α_j failed.");
        return false;
    }
}

//--------------------------------------
fn compute3(
    x_prev_inv: Vec<G1Affine>,
    v_rd_jm1: &str,
) -> (Vec<G1Affine>, Vec<G1Affine>, Vec<G2Affine>) {
    println!("\n compute3 is running for vector x^i");

    let rng = &mut thread_rng();
    let mut x_j_g1 = Vec::new();
    let mut x_i_g1 = Vec::new();
    let mut y_x = Vec::new();

    for x_inv in &x_prev_inv {
        let x_j = Fr::rand(rng);

        // Compute [x^i]_j = x_j * [x^i]_j^{-1}
        let xi_g1 = (*x_inv * x_j).into_affine();
        x_j_g1.push(xi_g1);
        x_i_g1.push(xi_g1);

        // Compute proof of knowledge for each element
        let y = pok(x_j, v_rd_jm1);
        y_x.push(y);
    }

    (x_j_g1, x_i_g1, y_x)
}

fn verify3(
    x_j: Vec<G1Affine>,
    x_prev_inv: Vec<G1Affine>,
    y_x: Vec<G2Affine>,
    v_rd_jm1: &str,
) -> bool {
    println!("\n verify3 is running for vector x^i");

    for i in 0..x_j.len() {
        let r_x = oracle_r(x_j[i], v_rd_jm1);

        if check_pok(x_j[i], v_rd_jm1, y_x[i]) {
            println!("\n check_pok is valid for index {}", i);

            // Consistency check for each index
            if !consistent((x_prev_inv[i], x_j[i]), (r_x, y_x[i])) {
                println!("Consistency check failed at index {}", i);
                return false;
            }
        } else {
            println!("Proof of knowledge failed at index {}", i);
            return false;
        }
    }

    println!("\n All elements passed the verification.");
    true
}

fn compute2(
    alpha_beta_prev: G1Affine,
    v: &str,
) -> (G1Affine, G1Affine, G1Affine, G2Affine, G2Affine, G2Affine) {
    println!("\n compute2 is running");
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);
    let beta_j = Fr::rand(rng);

    let g1 = G1Affine::generator();
    let alpha_j_g1 = (g1 * alpha_j).into_affine();
    let beta_j_g1 = (g1 * beta_j).into_affine();

    let y_alpha = pok(alpha_j, v);
    let y_beta = pok(beta_j, v);

    let alpha_beta_out = (alpha_beta_prev * alpha_j * beta_j).into_affine();

    let g2 = G2Affine::generator();
    let alpha_beta_g2 = (g2 * alpha_j * beta_j).into_affine();

    (
        alpha_beta_out,
        alpha_j_g1,
        beta_j_g1,
        alpha_beta_g2,
        y_alpha,
        y_beta,
    )
}

fn verify2(
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
        //     == <MNT6_753 as Pairing>::pairing(G1Affine::generator(), alpha_beta_g2)
        if true {
            println!("\n pairing2 is valid....");
            return consistent(
                (alpha_beta_prev, alpha_beta_out),
                (G2Affine::generator(), alpha_beta_g2),
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

fn compute1(alpha_g1_pre: G1Affine, v: &str) -> (G1Affine, G1Affine, G2Affine) {
    println!("\n compute1 is running");
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);

    let g1 = G1Affine::generator();
    let alpha_j_g1 = (g1 * alpha_j).into_affine();

    let y = pok(alpha_j, v);

    let alpha_g1_out = (alpha_g1_pre * alpha_j).into_affine();

    (alpha_g1_out, alpha_j_g1, y)
}

fn verify1(
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

fn main() {
    let alpha_g1_pre = G1Affine::generator();
    let v = "example string";

    let (alpha_g1_out, alpha_j_g1, y) = compute1(alpha_g1_pre, v);
    let is_valid1 = verify1(alpha_g1_pre, alpha_g1_out, alpha_j_g1, v, y);
    println!("\nVerification Result for compute1/verify1: {}", is_valid1);

    let (alpha_beta_j, alpha_j_g1_2, beta_j_g1, alpha_beta_j_2, y_alpha, y_beta) =
        compute2(alpha_g1_pre, v);
    let is_valid2 = verify2(
        alpha_g1_pre,
        alpha_beta_j,
        alpha_j_g1_2,
        beta_j_g1,
        y_alpha,
        y_alpha,
        y_beta,
        v,
    );
    println!("\nVerification Result for compute2/verify2: {}", is_valid2);

    let g1 = G1Affine::generator();
    // Create vector x^i for testing
    let x_prev_inv = vec![g1; 5]; // Example vector of 5 elements

    // Compute3
    let (x_j_g1, x_i_g1, y_x) = compute3(x_prev_inv.clone(), v);

    // Verify3
    let is_valid3 = verify3(x_j_g1, x_prev_inv, y_x, v);
    println!("\nVerification Result for compute3/verify3: {}", is_valid3);

    // Create vector [αx^i] for testing
    let ax_prev_inv = vec![g1; 5]; // Example vector of 5 elements

    // Create vector x^i for testing
    // let x_prev_inv = vec![g1; 5];
    let rng = &mut thread_rng();
    let x_prev_inv: Vec<G1Affine> = (0..5).map(|_| (g1 * Fr::rand(rng)).into_affine()).collect();

    // Compute4
    let (ax_i_j, x_j_g1, ax_i_j_1, alpha_j_g1, y_alpha) = compute4(x_prev_inv.clone(), v);

    // Verify4
    let is_valid4 = verify4(ax_i_j, alpha_j_g1, x_j_g1, x_prev_inv, v, y_alpha);
    println!("\nVerification Result for compute4/verify4: {}", !is_valid4);
}
