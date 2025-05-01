use ark_ff::{BigInt, BigInteger384, PrimeField, ToConstraintField, UniformRand};
use ark_serialize::{CanonicalSerialize, Compress};
use blake2::crypto_mac::generic_array::typenum::U64;
use blake2::crypto_mac::generic_array::GenericArray;
use blake2::{Blake2b, Digest};
use icicle_bls12_381::curve::{CurveCfg, G1Affine as IcicleG1Affine, G1Projective as IcicleG1Projective, G2Affine as IcicleG2Affine, G2CurveCfg, G2Projective as IcicleG2Projective, ScalarField};
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_core::traits::{Arithmetic, FieldImpl};
use libs::field_structures::Tau;
use libs::group_structures::{
    icicle_g1_affine_to_ark, icicle_g2_affine_to_ark, pairing, G1serde, G2serde,
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::ops::Mul;
use ark_bls12_381::{Config, Fr, G1Affine as ArkG1Affine,G1Projective as ArkG1Projective, G2Affine as ArkG2Affine, G2Projective as ArkG2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use rand::Rng;
//A = [A1 B1], B = [A2, B2], C = [C1, C2]
pub fn consistent(A : &[G1serde], B : &[G2serde], C : &[G2serde]) -> bool {
    let G2 = icicle_g2_generator();
    if C.len() == 2 {
        same_ratio(A[0],A[1],C[0],C[1]) && same_ratio(A[0],A[1],B[0],B[1])
    } else {
        same_ratio(A[0],A[1],G2serde(G2),C[1])
    }
}

pub fn check_pok(A : G1serde, G1 : G1serde, B: G2serde, v: &[u8])-> bool {
    let y = ro(&G1serde(A.0),v);
    same_ratio(G1, A, G2serde(IcicleG2Affine::from(y)), B)
}

pub fn pok(alpha : ScalarField, G1 :G1serde, v: &[u8]) ->IcicleG2Projective {
    let alphaG1 = G1.0.to_projective().mul(alpha);
    let y = ro(&G1serde(IcicleG1Affine::from(alphaG1)), v);
    y.mul(alpha)
}

pub fn same_ratio(g1_0: G1serde, g1_1: G1serde, g2_0: G2serde, g2_1: G2serde) -> bool {
    let pair1 = pairing(&[g1_0], &[g2_1]);
    let pair2 = pairing(&[g1_1], &[g2_0]);
    pair1.eq(&pair2)
}
pub fn blank_hash() -> GenericArray<u8, U64> {
    Blake2b::new().result()
}

fn serialize_g1_affine_compressed(point: &IcicleG1Affine) -> [u8; 48] {
    let mut buf = Vec::new();

    icicle_g1_affine_to_ark(point).serialize_with_mode(
        &mut buf,
        Compress::Yes,
    ).expect("Serialization failed");

    let mut out = [0u8; 48];
    out.copy_from_slice(&buf);
    out
}
pub fn hash_to_g2(digest: &[u8]) -> Projective<G2CurveCfg> {
    assert!(digest.len() >= 32, "Digest must be at least 32 bytes");

    // Use the first 32 bytes as seed
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest[..32]);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let g2 = icicle_g2_generator();

    let limbs: [u32; 8] = rng.gen();
    let scalar = ScalarField::from(limbs);
    g2.to_projective().mul(scalar)
}
/// Hashes to G2 and returns a compressed 96-byte representation.
pub fn hash_to_g2_compressed(digest: &[u8]) -> [u8; 96] {
    let point = hash_to_g2(digest);
    let pointArk = icicle_g2_affine_to_ark(&IcicleG2Affine::from(point));
    // Serialize to compressed form (96 bytes)
    let mut buf = Vec::new();
    pointArk
        .serialize_with_mode(&mut buf, Compress::Yes)
        .unwrap();

    let mut out = [0u8; 96];
    out.copy_from_slice(&buf);
    out
}

pub fn ro(a: &G1serde, v: &[u8]) -> Projective<G2CurveCfg> {
    let mut h = Blake2b::default();
    h.input(v);
    h.input(serialize_g1_affine_compressed(&a.0));
    hash_to_g2(h.result().as_ref())
}

pub fn icicle_g1_generator() -> IcicleG1Affine {
    let x_limbs: [u32; 12] = [0xdb22c6bb,
        0xfb3af00a,
        0xf97a1aef,
        0x6c55e83f,
        0x171bac58,
        0xa14e3a3f,
        0x9774b905,
        0xc3688c4f,
        0x4fa9ac0f,
        0x2695638c,
        0x3197d794,
        0x17f1d3a7];
    let y_limbs: [u32; 12] = [
        1187375073,
        212476713,
        2726857444,
        3493644100,
        738505709,
        14358731,
        3587181302,
        4243972245,
        1948093156,
        2694721773,
        3819610353,
        146011265,
    ];
    // Build the G1Affine point from limbs
     IcicleG1Affine::from_limbs(x_limbs, y_limbs)
}
pub fn icicle_g2_generator() -> IcicleG2Affine {
    let x_limbs: [u32; 24] = [
        3240213944,
        3565180616,
        2818948079,
        195822374,
        2061750647,
        3025210212,
        4198513410,
        3336862420,
        767889489,
        638059815,
        4035906193,
        38445746,
        1560554366,
        3853286661,
        328490327,
        860680466,
        3699331145,
        3050987963,
        2569057818,
        1500238032,
        2284277605,
        2108478368,
        1383178080,
        333458272,
    ];

    let y_limbs: [u32; 24] = [
        146286593,
        3784529030,
        1001169545,
        2453326284,
        1365299500,
        1833081449,
        2361250727,
        2919078826,
        3660461338,
        2362035654,
        1920822801,
        216388903,
        4032788926,
        2863204191,
        1558977953,
        1060572455,
        1462671787,
        645173931,
        2242339759,
        3409848446,
        734170009,
        850186928,
        782709964,
        101106848,
    ];
    // Build the G1Affine point from limbs
    IcicleG2Affine::from_limbs(x_limbs, y_limbs)
}

#[test]
pub fn test_ro() {
    let g1_gen = icicle_g1_generator();
    let v = [99u8; 64];
    let out1 = ro(&G1serde(g1_gen), &v);
    let out2 = ro(&G1serde(g1_gen), &v);
    assert_eq!(out1,out2)
}
#[test]
fn testG2Generator() {
    let g2Ice = icicle_g2_generator();
    let res = icicle_g2_affine_to_ark(&g2Ice);
    let arcG2 = ArkG2Affine::generator();
    assert_eq!(res, arcG2);
}
#[test]
fn testG1Generator() {
    // Build the G1Affine point from limbs
    let g1Ice = icicle_g1_generator();
    let res = icicle_g1_affine_to_ark(&g1Ice);
    let arcG1 = ArkG1Affine::generator();
    assert_eq!(res, arcG1);
}

#[test]
fn test_hash_to_g2_compressed_deterministic() {
    let digest = [99u8; 64];
    let out1 = hash_to_g2_compressed(&digest);
    let out2 = hash_to_g2_compressed(&digest);
    assert_eq!(out1.len(), 96);
    assert_eq!(out2.len(), 96);
    assert_eq!(out1, out2, "Deterministic input should yield same output");
}

#[test]
fn test_hash_to_g2_compressed_unique() {
    let digest1 = [1u8; 64];
    let digest2 = [2u8; 64];
    let out1 = hash_to_g2_compressed(&digest1);
    let out2 = hash_to_g2_compressed(&digest2);
    assert_eq!(out1.len(), 96);
    assert_eq!(out2.len(), 96);
    assert_ne!(out1, out2, "Different inputs should yield different outputs");
}
#[test]
pub fn test_same_ratio() {
    let g1_gen = CurveCfg::generate_random_affine_points(1)[0].to_projective();
    let g2_gen = G2CurveCfg::generate_random_affine_points(1)[0].to_projective();

    let tau = Tau::gen();

    let x2G1 = IcicleG1Affine::from(g1_gen.mul(tau.x.pow(2)));
    let xyG1 = IcicleG1Affine::from(g1_gen.mul(tau.x).mul(tau.y));

    let y2G2 = IcicleG2Affine::from(g2_gen.mul(tau.y.pow(2)));
    let xyG2 = IcicleG2Affine::from(g2_gen.mul(tau.y).mul(tau.x));

    let result = same_ratio(G1serde(x2G1), G1serde(xyG1), G2serde(xyG2), G2serde(y2G2));
    assert_eq!(result, true)
}
#[test]
pub fn test_pok() {
    let G1 = icicle_g1_generator();

    let tau = Tau::gen();
    let v = [72u8; 64];
    let A = G1.to_projective().mul(tau.alpha);
    let cpok = pok(tau.alpha, G1serde(G1), &v);

    let result = check_pok(G1serde(IcicleG1Affine::from(A)),G1serde(G1),G2serde(IcicleG2Affine::from(cpok)),&v);
    assert_eq!(result,true)
}

fn main() {}
