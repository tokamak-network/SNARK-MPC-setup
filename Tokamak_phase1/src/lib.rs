use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use ark_mnt6_753::{Fr, G1Affine, G2Affine, G2Projective, MNT6_753};
use ark_serialize::CanonicalSerialize;
use blake2::{Blake2b512, Digest};
use rand::{rngs::StdRng, SeedableRng};


pub fn oracle_r(alpha_g1: G1Affine, v: &str) -> G2Affine {
    let mut hasher = Blake2b512::new();

    let mut buffer = Vec::new();
    alpha_g1.serialize_uncompressed(&mut buffer).unwrap();
    hasher.update(&buffer);
    hasher.update(v.as_bytes());

    let hash_result = hasher.finalize();
    let g2_element = hash_to_g2(&hash_result);
    g2_element.into_affine()
}

pub fn hash_to_g2(digest: &[u8]) -> G2Projective {
    assert!(digest.len() >= 32);

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest[..32]);
    let rng = &mut StdRng::from_seed(seed);

    G2Projective::rand(rng)
}

pub fn pok(alpha: Fr, v: &str) -> G2Affine {
    // Step 1: Compute [alpha]_1 = alpha * G1
    let g1 = G1Affine::generator();
    let alpha_g1 = (g1 * alpha).into_affine();

    // Step 2: Compute y = RO([alpha]_1, v)
    let y = oracle_r(alpha_g1, v);

    // Step 3: Compute and return alpha * y
    let alpha_y = (y * alpha).into_affine();
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
    same_ratio::<MNT6_753>((G1Affine::generator(), a), (y, b))
}

pub fn consistent(
    g1_pair: (G1Affine, G1Affine), // Pair from G1
    g2_pair: (G2Affine, G2Affine), // Pair from G2
) -> bool {
    same_ratio::<MNT6_753>(
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
//         same_ratio::<P>((a.0, a.1), (P::G2Affine::generator(), b.1))
//     };

//     // let r = if let Some((c1, c2)) = c {
//     //     // If C is in (G2^*)^2
//     //     same_ratio::<P>((a.0, a.1), (c1, c2))
//     //     // println!("consistent:1");
//     // } else {
//     //     // Else C is in G2^*
//     //     same_ratio::<P>((a.0, a.1), (P::G2Affine::generator(), b.1))
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
