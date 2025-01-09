use ark_ec::{AffineRepr, CurveGroup}; // Import AffineRepr to resolve `generator` issue
use ark_ff::UniformRand;
use ark_mnt6_753::{Fr, G1Affine, G2Affine};
use deneme::{check_pok, consistent, oracle_r, pok, same_ratio}; // Import the required functions
use rand::thread_rng;

fn compute1(alpha_g1_pre: G1Affine, v: &str) -> (G1Affine, G1Affine, G2Affine) {
    println!("\n compute1 is running");
    let rng = &mut thread_rng();
    let alpha_j = Fr::rand(rng);

    // Step 1: Compute alpha_j_g1 = alpha_j * G1
    let g1 = G1Affine::generator(); // Using the generator from AffineRepr
    let alpha_j_g1 = (g1 * alpha_j).into_affine();

    // Step 2: Compute y = POK(alpha_j, v)
    let y = pok(alpha_j, v);

    // Step 3: Compute [alpha^j]_1 = alpha_j * [alpha^(j-1)]_1
    let alpha_g1_out = (alpha_g1_pre * alpha_j).into_affine();

    // println!(
    //     "Compute1 Output: ([alpha^(j-1)]_1 = {:?}, [alpha^j]_1 = {:?}, y = {:?})",
    //     alpha_g1_pre, alpha_g1_out, y
    // );

    (alpha_g1_out, alpha_j_g1, y)
}

fn verify1(
    alpha_g1_pre: G1Affine, // [alpha^(j-1)]_1
    alpha_g1_out: G1Affine, // [alpha^j]_1
    alpha_j_g1: G1Affine,   // [alpha_j]_1
    v: &str,                // Public input
    y: G2Affine,            // Proof y
) -> bool {
    println!("\nVerify1 is running.....:");

    // Step 1: Compute r_a,j = RO([alpha^j]_1, v)
    let r_a_j = oracle_r(alpha_j_g1, v);

    // Step 2: Check the proof of knowledge (POK)
    if !check_pok(alpha_j_g1, v, y) {
        println!("Proof of knowledge failed.");
        return false;
    }

    // Step 3: Verify consistency using the consistent function
    // let is_consistent = consistent::<ark_mnt6_753::MNT6_753>(
    //     (alpha_g1_pre, alpha_g1_out), // A = ([alpha^(j-1)]_1, [alpha^j]_1)
    //     (r_a_j, y),                   // B = (r_a,j, y)
    //     None,                         // C = None
    // );

    let is_consistent = consistent(
        (alpha_g1_pre, alpha_g1_out), // Pair from G1
        (r_a_j, y),                   // Pair from G2
    );
    // let is_consistent = same_ratio::<ark_mnt6_753::MNT6_753>(
    //     (alpha_g1_pre, alpha_g1_out), // Pair from G1
    //     (r_a_j, y),                   // Pair from G2
    // );
    if !is_consistent {
        println!("Consistency check failed.");
        return false;
    }

    println!("Proof is valid.");
    true
}

// fn test_oracle_r() {
//     // Generate a random G1Affine point (alpha_g1)
//     let rng = &mut thread_rng();
//     let alpha = Fr::rand(rng);
//     let g1 = G1Affine::generator();
//     let alpha_g1 = (g1 * alpha).into_affine();

//     // Public input string (v)
//     let v = "example string";

//     // Compute the oracle result twice
//     let result1 = oracle_r(alpha_g1, v);
//     let result2 = oracle_r(alpha_g1, v);

//     // Print the results for debugging
//     println!("Oracle Result 1: {:?}", result1);
//     println!("Oracle Result 2: {:?}", result2);

//     // Check if the results are consistent
//     if result1 == result2 {
//         println!("Oracle function is consistent.");
//     } else {
//         println!("Oracle function is inconsistent!");
//     }
// }

fn main() {
    // test_oracle_r();

    // Initial input
    let alpha_g1_pre = G1Affine::generator(); // Using the generator from AffineRepr
    let v = "example string";

    // Compute proof using compute1
    let (alpha_g1_out, alpha_j_g1, y) = compute1(alpha_g1_pre, v);

    // println!(
    //     "Main Output: alpha_g1_out = {:?}, alpha_j_g1 = {:?}, y = {:?}",
    //     alpha_g1_out, alpha_j_g1, y
    // );

    // Verify proof using verify1
    let is_valid = verify1(alpha_g1_pre, alpha_g1_out, alpha_j_g1, v, y);

    println!("\nVerification Result: {}", is_valid);
}