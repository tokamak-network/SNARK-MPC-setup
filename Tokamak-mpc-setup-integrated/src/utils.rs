pub(crate) use crate::conversions::{
    hash_to_g2, icicle_g1_generator, icicle_g2_generator, serialize_g1_affine_compressed,
};
use ark_ec::{AffineRepr, PrimeGroup};
use blake2::{Blake2b, Digest};
use icicle_bls12_381::curve::{ScalarCfg, ScalarField};
use icicle_core::curve::Curve;
use icicle_core::traits::{Arithmetic, FieldImpl, GenerateRandom};
use libs::field_structures::Tau;
use libs::group_structures::{pairing, G1serde, G2serde};
use rand::Rng;
use rayon::join;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::Mul;
use std::sync::Mutex;
use std::time::Instant;
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::PairingOutput;
use lazy_static::lazy_static;
// Import rayon prelude

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SerialSerde {
    g1: Vec<G1serde>, //[xG1, x^2G1, x^3G1, ..., x^s_maxG1]
    g2: G2serde,      //xG2
}
impl SerialSerde {
    //xr, xr^2, xr^3...xr^n where n is the length of xr vector
    pub(crate) fn mul(&self, xr_powers: &Vec<ScalarField>) -> SerialSerde {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let serde = SerialSerde {
            g1: g1
                .par_iter()
                .zip(xr_powers.par_iter())
                .map(|(g1, xr)| g1.mul(*xr))
                .collect(),
            g2: g2.mul(xr_powers[0]),
        };
        serde
    }
    pub(crate) fn get_g1(&self, index: usize) -> G1serde {
        self.g1[index]
    }
    pub(crate) fn get_g2(&self) -> G2serde {
        self.g2
    }
    pub fn len_g1(&self) -> usize {
        self.g1.len()
    }

    pub fn new(s_max: usize) -> SerialSerde {
        let g1 = icicle_g1_generator();
        let g2 = icicle_g2_generator();
        SerialSerde {
            g1: vec![g1; s_max],
            g2,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairSerde {
    pub g1: G1serde, //xG1
    pub g2: G2serde, //xG2
}

impl PairSerde {
    pub(crate) fn mul(&self, p0: ScalarField) -> PairSerde {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let serde = PairSerde {
            g1: g1.mul(p0),
            g2: g2.mul(p0),
        };
        serde
    }
    pub fn new(g1: G1serde, g2: G2serde) -> PairSerde {
        PairSerde { g1, g2 }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proof2 {
    x_rG1: G1serde, //latest x_r contribution
    pok_x: G2serde,
    v: [u8; 32],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proof5 {
    proof2_alpha: Proof2,
    proof2_x: Proof2,
    proof2_y: Proof2,
}
impl Proof5 {
    /// Save the Proof5 to a JSON file
    pub fn save_to_json(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let json_str = serde_json::to_string_pretty(&self)
            .expect("JSON serialization failed");
        writer.write_all(json_str.as_bytes())?;
        Ok(())
    }

    /// Load the Accumulator from a JSON file and recalculate the hash
    pub fn load_from_json(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut json_str = String::new();
        reader.read_to_string(&mut json_str)?;
        let proof: Proof5 = serde_json::from_str(&json_str)
            .expect("JSON deserialization failed");
        Ok(proof)
    }
}
//type 1: compute1
pub fn compute1(prev_alpha: PairSerde, v: &[u8]) -> (PairSerde, G1serde, G2serde) {
    let alpha_r = next_random();
    let pok_alpha = pok(alpha_r, v);
    let alpha_rG1 = icicle_g1_generator().mul(alpha_r);
    let cur_alpha = prev_alpha.mul(alpha_r);
    (cur_alpha, alpha_rG1, pok_alpha)
}
//type 1: verify1
pub fn verify1(
    prev_alpha: PairSerde,
    cur_alpha: PairSerde,
    alpha_rG1: G1serde,
    alpha_pok: G2serde,
    v: &[u8],
) -> bool {
    let G1 = icicle_g1_generator();
    if check_pok(alpha_rG1, G1, alpha_pok, v) {
        let r_alpha = ro(&alpha_rG1, v);
        return consistent(
            &[prev_alpha.g1, cur_alpha.g1],
            &[prev_alpha.g2, cur_alpha.g2],
            &[r_alpha, alpha_pok],
        );
    }
    false
}


//type 2: compute2
//Let [x^i]^0 = G_1 or [x^i]^0 = G
pub fn compute2(prev_x: &SerialSerde, v: &[u8; 32]) -> (SerialSerde, Proof2) {
    let (cur_x, proof2, _) = compute2_temp(prev_x, v);
    (cur_x, proof2)
}

//type 2: verify2
pub fn verify2(prev_x: &SerialSerde, cur_x: &SerialSerde, proof2: &Proof2) -> bool {
    let g1 = icicle_g1_generator();

    if !check_pok(proof2.x_rG1, g1, proof2.pok_x, proof2.v.as_ref()) {
        return false;
    }

    let r_alpha = ro(&proof2.x_rG1, proof2.v.as_ref());

    if !consistent(
        &[prev_x.get_g1(0), cur_x.get_g1(0)],
        &[prev_x.get_g2(), cur_x.get_g2()],
        &[r_alpha, proof2.pok_x],
    ) {
        return false;
    }

    let g2 = icicle_g2_generator();

    // Parallelize loop consistency checks
    (1..cur_x.len_g1()).into_par_iter().all(|i| {
        consistent(
            &[cur_x.get_g1(i - 1), cur_x.get_g1(i)],
            &[],
            &[g2, cur_x.get_g2()],
        )
    })
}
pub fn verify2i(prev_x: &Vec<PairSerde>, cur_x: &Vec<PairSerde>, proof2: &Proof2) -> bool {
    let g1 = icicle_g1_generator();

    if !check_pok(proof2.x_rG1, g1, proof2.pok_x, proof2.v.as_ref()) {
        return false;
    }

    let r_alpha = ro(&proof2.x_rG1, proof2.v.as_ref());

    if !consistent(
        &[prev_x[0].g1, cur_x[0].g1],
        &[prev_x[0].g2, cur_x[0].g2],
        &[r_alpha, proof2.pok_x],
    ) {
        return false;
    }

    let g2 = icicle_g2_generator();

    // Parallelize the consistency checks across elements
    (1..cur_x.len())
        .into_par_iter()
        .all(|i| consistent(&[cur_x[i - 1].g1, cur_x[i].g1], &[], &[g2, cur_x[0].g2]))
}

fn compute2_temp(
    prev_x: &SerialSerde,
    v: &[u8; 32],
) -> (SerialSerde, Proof2, Vec<ScalarField>) {
    let x_r = next_random();
    let pok_x = pok(x_r, v);
    let x_rG1 = icicle_g1_generator().mul(x_r);
    let len_x = prev_x.len_g1();

    // Precompute the powers of x_r efficiently
    let mut x_powers = compute_powers(x_r, len_x);
    let cur_x = prev_x.mul(&x_powers);
    (cur_x, Proof2 { x_rG1, pok_x, v: *v }, x_powers)
}
fn compute2_tempi(
    prev_x: &Vec<PairSerde>,
    v: &[u8; 32],
) -> (Vec<PairSerde>, Proof2, Vec<ScalarField>) {
    let x_r = next_random();
    let pok_x = pok(x_r, v);
    let x_rG1 = icicle_g1_generator().mul(x_r);
    let len_x = prev_x.len();

    // Precompute the powers of x_r efficiently
    let mut x_powers = compute_powers(x_r, len_x);
    let cur_x: Vec<PairSerde> = prev_x
        .par_iter()
        .zip(x_powers.par_iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    (cur_x, Proof2 { x_rG1, pok_x, v: *v }, x_powers)
}

lazy_static! {
    static ref INDEX: Mutex<usize> = Mutex::new(0);
}
pub fn next_random() -> ScalarField {
    let scalars = [ScalarField::from_u32(3),ScalarField::from_u32(5),ScalarField::from_u32(7)];
    //ScalarCfg::generate_random(1)[0]
    let mut idx = INDEX.lock().unwrap();
    let next = *idx;
    *idx = (*idx + 1) % scalars.len(); // Wrap around to avoid panic
    
    println!("next random: {:?}", scalars[next]);
    scalars[next]
}

//type 3: compute3
//Let [x^i*y^k]
pub fn compute3(prev_xy: &Vec<G1serde>, prev_x: &Vec<PairSerde>, prev_y: &SerialSerde, v: &[u8; 32]) -> (Vec<G1serde>, Vec<PairSerde>, SerialSerde, Proof2, Proof2, Vec<ScalarField>, Vec<ScalarField>, Vec<ScalarField>) {
    let len_x = prev_x.len();

    let x_r = next_random();
    let proof2_x = Proof2 { x_rG1: icicle_g1_generator().mul(x_r), pok_x: pok(x_r, v), v: *v };

    // Precompute the powers of x_r efficiently
    let x_powers = compute_powers(x_r, len_x);
    let cur_x: Vec<PairSerde> = prev_x
        .par_iter()
        .zip(x_powers.par_iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    let (cur_y, proof2_y, y_powers) = compute2_temp(prev_y, v);

    let xy_powers = vector_product(&x_powers, &y_powers);
    let cur_xy: Vec<G1serde> = prev_xy
        .par_iter()
        .zip(xy_powers.par_iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    (cur_xy, cur_x, cur_y, proof2_x, proof2_y, x_powers, y_powers, xy_powers)
}


//type 3: verify3
pub fn verify3(prev_x: &Vec<PairSerde>, prev_y: &SerialSerde, cur_xy: &Vec<G1serde>, cur_x: &Vec<PairSerde>, cur_y: &SerialSerde, proof2_x: &Proof2, proof2_y: &Proof2) -> bool {
    if !verify2i(prev_x, cur_x, proof2_x) {
        return false;
    }
    if !verify2(prev_y, cur_y, proof2_y) {
        return false;
    }
    //check xy consistency
    let g2 = icicle_g2_generator();
    let len_y = cur_y.len_g1();

    (0..cur_x.len()).into_par_iter().all(|i| {
        let xi = cur_x[i];
        (0..len_y).into_par_iter().all(|k| {
            let ykG1 = cur_y.get_g1(k);
            let xyG1 = cur_xy[i * len_y + k];
            consistent(&[ykG1, xyG1], &[], &[g2, xi.g2])
        })
    })
}

//type 3: verify3i
pub fn verify3i(prev_x: &Vec<PairSerde>, prev_y: &Vec<PairSerde>, cur_xy: &Vec<G1serde>, cur_x: &Vec<PairSerde>, cur_y: &Vec<PairSerde>, proof2_x: &Proof2, proof2_y: &Proof2) -> bool {
    if !verify2i(prev_x, cur_x, proof2_x) {
        return false;
    }
    if !verify2i(prev_y, cur_y, proof2_y) {
        return false;
    }
    //check xy consistency
    let g2 = icicle_g2_generator();
    let len_y = cur_y.len();

    // Nested parallel loops for consistency checks
    (0..cur_x.len()).into_par_iter().all(|i| {
        let xi = cur_x[i];
        (0..len_y).into_par_iter().all(|k| {
            let ykG1 = cur_y[k].g1;
            let xyG1 = cur_xy[i * len_y + k];
            consistent(&[ykG1, xyG1], &[], &[g2, xi.g2])
        })
    })
}

#[test]
pub fn test_compute3() {
    //initialize
    let g1 = icicle_g1_generator();
    let g2 = icicle_g2_generator();

    let s_max1: usize = 4;
    let s_max2: usize = 5;

    let v = [34u8; 32];
    let mut prev_x = vec![PairSerde { g1: g1.clone(), g2: g2.clone() }; s_max1];
    let mut prev_y = SerialSerde::new(s_max2);
    let mut prev_xy = vec![g1; s_max1 * s_max2];

    // first participant
    let (cur_xy, cur_x, cur_y, proof2_x, proof2_y, _, _, _) = compute3(&prev_xy, &prev_x, &prev_y, &v);
    assert_eq!(verify3(&prev_x,&prev_y,&cur_xy,&cur_x,&cur_y,&proof2_x, &proof2_y), true,);
    prev_xy = cur_xy;
    prev_x = cur_x;
    prev_y = cur_y;

    let (cur_xy, cur_x, cur_y, proof2_x, proof2_y, _, _, _) = compute3(&prev_xy, &prev_x, &prev_y, &v);
    assert_eq!(verify3(&prev_x,&prev_y,&cur_xy,&cur_x,&cur_y,&proof2_x, &proof2_y), true,);
    prev_xy = cur_xy;
    prev_x = cur_x;
    prev_y = cur_y;

    let (cur_xy, cur_x, cur_y, proof2_x, proof2_y, _, _, _) = compute3(&prev_xy, &prev_x, &prev_y, &v);
    assert_eq!(verify3(&prev_x,&prev_y,&cur_xy,&cur_x,&cur_y,&proof2_x, &proof2_y), true,);
    prev_xy = cur_xy;
    prev_x = cur_x;
    prev_y = cur_y;
}


pub fn compute5(prev_alphaxy: &Vec<G1serde>, prev_xy: &Vec<G1serde>, prev_alphax: &Vec<G1serde>, prev_alphay: &Vec<G1serde>, prev_alpha: &Vec<PairSerde>, prev_x: &Vec<PairSerde>, prev_y: &SerialSerde, v: &[u8; 32])
                -> (Vec<G1serde>, Vec<G1serde>, Vec<G1serde>, Vec<G1serde>, Vec<PairSerde>, Vec<PairSerde>, SerialSerde, Proof5) {
    let (cur_xy, cur_x, cur_y, proof2_x, proof2_y, x_powers, y_powers, xy_powers) = compute3(&prev_xy, &prev_x, &prev_y, &v);
    let (cur_alpha, proof2_alpha, alpha_powers) = compute2_tempi(prev_alpha, v);

    let mut alphaxy_powers = vector_product(&alpha_powers, &xy_powers);
    let cur_alphaxy: Vec<G1serde> = prev_alphaxy
        .par_iter()
        .zip(alphaxy_powers.par_iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    let alphax_powers = vector_product(&alpha_powers, &x_powers);
    let cur_alphax: Vec<G1serde> = prev_alphax
        .par_iter()
        .zip(alphax_powers.par_iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    let alphay_powers = vector_product(&alpha_powers, &y_powers);
    let cur_alphay: Vec<G1serde> = prev_alphay
        .par_iter()
        .zip(alphay_powers.par_iter())
        .map(|(y, scalar)| y.mul(*scalar))
        .collect();

    (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, Proof5 { proof2_alpha, proof2_x, proof2_y })
}

//type 5: verify5
pub fn verify5(prev_alpha: &Vec<PairSerde>, prev_x: &Vec<PairSerde>, prev_y: &SerialSerde,
               cur_alphaxy: &Vec<G1serde>, cur_xy: &Vec<G1serde>, cur_alphax: &Vec<G1serde>, cur_alphay: &Vec<G1serde>, cur_alpha: &Vec<PairSerde>, cur_x: &Vec<PairSerde>, cur_y: &SerialSerde,
               proof5: &Proof5) -> bool {
    if !verify2i(prev_alpha, cur_alpha, &proof5.proof2_alpha) {
        return false;
    }
    if !verify3(prev_x, prev_y, cur_xy, cur_x, cur_y, &proof5.proof2_x, &proof5.proof2_y) {
        return false;
    }
    if !verify3(prev_alpha, prev_y, cur_alphay, cur_alpha, cur_y, &proof5.proof2_alpha, &proof5.proof2_y) {
        return false;
    }
    if !verify3i(prev_alpha, prev_x, cur_alphax, cur_alpha, cur_x, &proof5.proof2_alpha, &proof5.proof2_x) {
        return false;
    }
    let start = Instant::now();

    let len_x = prev_x.len();
    let len_y = prev_y.len_g1();
    let g2 = icicle_g2_generator();

    let result = cur_alpha.iter().enumerate().all(|(h, alpha)| {
        (0..cur_x.len()).into_par_iter().all(|i| {
            (0..cur_y.len_g1()).into_par_iter().all(|k| {
                let xy = cur_xy[i * len_y + k];
                let cur_alphaxy = cur_alphaxy[get_alphaxy_index(h, i, k, len_x, len_y)];
                consistent(&[xy, cur_alphaxy], &[], &[g2, alpha.g2])
            })
        })
    });
    println!("Time elapsed for verify for the last consistency: {:?}", start.elapsed().as_secs());
    result
}
//index = (h* len_x * len_y) + (i * len_y) + k;
fn get_alphaxy_index(h: usize, i: usize, k: usize, len_x: usize, len_y: usize) -> usize {
    h * len_x * len_y + i * len_y + k
}


#[test]
pub fn test_compute5() {
    //initialize
    let g1 = icicle_g1_generator();
    let g2 = icicle_g2_generator();

    let s_max0: usize = 4;  //alpha
    let s_max1: usize = 16; //x^i
    let s_max2: usize = 32; //y^k

    let v = [34u8; 32];
    let mut prev_alpha = vec![PairSerde { g1: g1.clone(), g2: g2.clone() }; s_max0];
    let mut prev_x = vec![PairSerde { g1: g1.clone(), g2: g2.clone() }; s_max1];
    let mut prev_y = SerialSerde::new(s_max2);
    let mut prev_xy = vec![g1; s_max1 * s_max2];
    let mut prev_alphax = vec![g1; s_max0 * s_max1];
    let mut prev_alphay = vec![g1; s_max0 * s_max2];

    let mut prev_alphaxy = vec![g1; s_max0 * s_max1 * s_max2];

    // first participant
    let (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, proof5) =
        compute5(&prev_alphaxy, &prev_xy, &prev_alphax, &prev_alphay, &prev_alpha, &prev_x, &prev_y, &v);


    assert_eq!(verify5(&prev_alpha,&prev_x,&prev_y,
               &cur_alphaxy,&cur_xy,&cur_alphax,&cur_alphay,&cur_alpha,&cur_x,&cur_y,&proof5), true,);

    prev_alpha = cur_alpha;
    prev_x = cur_x;
    prev_y = cur_y;
    prev_xy = cur_xy;
    prev_alphaxy = cur_alphaxy;
    prev_alphax = cur_alphax;
    prev_alphay = cur_alphay;

    // second participant
    let (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, proof5) =
        compute5(&prev_alphaxy, &prev_xy, &prev_alphax, &prev_alphay, &prev_alpha, &prev_x, &prev_y, &v);


    assert_eq!(verify5(&prev_alpha,&prev_x,&prev_y,
               &cur_alphaxy,&cur_xy,&cur_alphax,&cur_alphay,&cur_alpha,&cur_x,&cur_y,&proof5), true,);
}


//ab_g1 = [A1 B1], ab_g2 = [A2, B2], C = [C1, C2]
pub fn consistent(ab_g1: &[G1serde], ab_g2: &[G2serde], C: &[G2serde]) -> bool {
    let A1 = ab_g1[0];
    let B1 = ab_g1[1];
    let C1 = C[0];
    let C2 = C[1];

    if ab_g2.is_empty() {
        same_ratio(A1, B1, C1, C2)
    } else {
        let A2 = ab_g2[0];
        let B2 = ab_g2[1];

        let (res_ab, res_c) = join(
            || same_ratio(A1, B1, A2, B2),
            || same_ratio(A1, B1, C1, C2),
        );

        res_ab && res_c
    }
}

pub fn check_pok(A: G1serde, G1: G1serde, B: G2serde, v: &[u8]) -> bool {
    let y = ro(&G1serde(A.0), v);
    same_ratio(G1, A, y, B)
}

pub fn pok(alpha: ScalarField, v: &[u8]) -> G2serde {
    let g1 = icicle_g1_generator();
    let alphaG1 = g1.mul(alpha);
    let y = ro(&alphaG1, v);
    y.mul(alpha)
}

pub fn same_ratio(g1_0: G1serde, g1_1: G1serde, g2_0: G2serde, g2_1: G2serde) -> bool {
    let results : Vec<PairingOutput<Bls12_381>> = [(&[g1_0], &[g2_1]), (&[g1_1], &[g2_0])]
        .par_iter()
        .map(|(g1, g2)| pairing(*g1, *g2))
        .collect();

    results[0].eq(&results[1])
}

pub fn ro(a: &G1serde, v: &[u8]) -> G2serde {
    let mut h = Blake2b::default();
    h.input(v);
    h.input(serialize_g1_affine_compressed(&a.0));
    hash_to_g2(h.result().as_ref())
}

fn vector_product(a: &Vec<ScalarField>, b: &Vec<ScalarField>) -> Vec<ScalarField> {
    let mut result: Vec<ScalarField> = Vec::with_capacity(a.len() * b.len());
    for i in 0..a.len() {
        for j in 0..b.len() {
            result.push(a[i].mul(b[j]));
        }
    }
    result
}
fn compute_powers(x_r: ScalarField, len_x: usize) -> Vec<ScalarField> {
    let mut x_powers: Vec<ScalarField> = Vec::with_capacity(len_x);
    let mut current_power = ScalarField::one();
    for i in 0..len_x {
        current_power = current_power.mul(x_r);
        x_powers.push(current_power);
    }
    x_powers
}

#[test]
pub fn test_consistent_case1() {
    // a1*c2 == b1*c1
    let g1_gen = icicle_g1_generator();
    let g2_gen = icicle_g2_generator();
    let a1 = next_random();
    let b1 = next_random();
    let c1 = next_random();
    let c2 = b1 * c1 * a1.inv();

    let a1G = g1_gen.mul(a1);
    let b1G = g1_gen.mul(b1);

    let c1G = g2_gen.mul(c1);
    let c2G = g2_gen.mul(c2);

    assert_eq!(consistent(&[a1G, b1G], &[], &[c1G, c2G]), true)
}

#[test]
pub fn test_consistent_case3() {
    let g1_gen = icicle_g1_generator();
    let g2_gen = icicle_g2_generator();

    let a = next_random();

    let two = ScalarField::one() + ScalarField::one();
    let three = two + ScalarField::one();
    let six = three + three;

    let a1 = a.mul(two); //2a
    let b1 = a.pow(2).mul(six); //6*a^2

    let c1 = a.mul(three); //3a

    let a1G = g1_gen.mul(a1);
    let b1G = g1_gen.mul(b1);

    let c1G = g2_gen; //G2
    let c2G = g2_gen.mul(c1); //3a * G2

    assert_eq!(consistent(&[a1G, b1G], &[], &[c1G, c2G]), true)
}

#[test]
pub fn test_compute1() {
    //initialize
    let g1 = icicle_g1_generator();
    let g2 = icicle_g2_generator();

    let alpha_0G1 = g1.clone();
    let alpha_0G2 = g2.clone();

    let v = [34u8; 32];
    let mut prev_pair = PairSerde {
        g1: alpha_0G1,
        g2: alpha_0G2,
    };
    // first participant
    let (cur_pair, alphaG1, pok_alpha) = compute1(prev_pair.clone(), &v);

    assert_eq!(verify1(prev_pair, cur_pair, alphaG1, pok_alpha, &v), true,);
    prev_pair = cur_pair;
    let (cur_pair, alphaG1, pok_alpha) = compute1(prev_pair.clone(), &v);

    assert_eq!(verify1(prev_pair, cur_pair, alphaG1, pok_alpha, &v), true,);
    prev_pair = cur_pair;
    let (cur_pair, alphaG1, pok_alpha) = compute1(prev_pair.clone(), &v);

    //beacon verified
    assert_eq!(verify1(prev_pair, cur_pair, alphaG1, pok_alpha, &v), true,);
}

#[test]
pub fn test_compute2() {
    let s_max: usize = 16;

    let x = vec![ScalarField::one(); s_max];

    let v = [34u8; 32];
    let mut prev_x_serial = SerialSerde::new(s_max);

    // first participant
    let (cur_pair, proof2) = compute2(&prev_x_serial, &v);
    assert_eq!(verify2(&prev_x_serial, &cur_pair, &proof2), true,);
    prev_x_serial = cur_pair;

    //second participant
    let (cur_pair, proof2) = compute2(&prev_x_serial, &v);
    assert_eq!(verify2(&prev_x_serial, &cur_pair, &proof2), true,);
    prev_x_serial = cur_pair;

    //third participant
    let (cur_x_serial, proof2) = compute2(&prev_x_serial, &v);
    assert_eq!(verify2(&prev_x_serial, &cur_x_serial, &proof2), true,);
}
#[test]
pub fn test_consistent_case4() {
    let g1_gen = icicle_g1_generator();
    let g2_gen = icicle_g2_generator();

    let two = ScalarField::from_u32(2);

    //same_ratio(A1, B1, A2, B2) && same_ratio(A1, B1, G2serde(g2), C2)
    // a1*b2 == b1*a2
    // a1 * c2 == b1 * 1
    let a1 = next_random();
    let a2 = next_random();

    let b1 = a1 * two;
    let b2 = a2 * two;

    let c1 = a1;
    let c2 = c1 * two;

    let a1G = g1_gen.mul(a1);
    let b1G = g1_gen.mul(b1);

    let a2G = g2_gen.mul(a2);
    let b2G = g2_gen.mul(b2);

    let c1G = g2_gen.mul(c1);
    let c2G = g2_gen.mul(c2);

    assert_eq!(consistent(&[a1G, b1G], &[a2G, b2G], &[c1G, c2G]), true)
}

#[test]
pub fn test_same_ratio() {
    let g1_gen = icicle_g1_generator();
    let g2_gen = icicle_g2_generator();

    let tau = Tau::gen();

    let x2G1 = g1_gen.mul(tau.x.pow(2));
    let xyG1 = g1_gen.mul(tau.x).mul(tau.y);

    let y2G2 = g2_gen.mul(tau.y.pow(2));
    let xyG2 = g2_gen.mul(tau.y).mul(tau.x);

    let result = same_ratio(x2G1, xyG1, xyG2, y2G2);
    assert_eq!(result, true)
}
#[test]
pub fn test_pok() {
    let g1 = icicle_g1_generator();

    let tau = Tau::gen();
    let v = [72u8; 64];
    let A = g1.mul(tau.alpha);
    let cpok = pok(tau.alpha, &v);

    let result = check_pok(A, g1, cpok, &v);
    assert_eq!(result, true)
}

#[test]
pub fn test_ro() {
    let g1_gen = icicle_g1_generator();
    let v = [99u8; 64];
    let out1 = ro(&g1_gen, &v);
    let out2 = ro(&g1_gen, &v);
    assert_eq!(out1.0, out2.0)
}

fn main() {}
