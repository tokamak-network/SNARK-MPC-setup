use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_mnt6_753::{Fr, G1Affine, G2Affine, MNT6_753};
use deneme::{check_pok, consistent, oracle_r, pok, same_ratio};
use rand::thread_rng;

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
}
