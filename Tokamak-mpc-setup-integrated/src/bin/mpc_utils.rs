use icicle_bls12_381::curve::{CurveCfg, ScalarField};
use icicle_core::curve::Curve;
use icicle_core::traits::FieldImpl;
use icicle_runtime::memory::HostSlice;
use lazy_static::lazy_static;
use libs::bivariate_polynomial::{BivariatePolynomial, DensePolynomialExt};
use libs::field_structures::Tau;
use libs::group_structures::G1serde;
use libs::iotools::{from_coef_vec_to_g1serde_vec, read_global_wire_list_as_boxed_boxed_numbers, SetupParams, SubcircuitInfo};
use libs::vector_operations::gen_evaled_lagrange_bases;
use rayon::prelude::*;
use std::ops::{Add, Mul};
use std::sync::Mutex;
use std::time::Instant;
use crate::QAP;

lazy_static! {
    static ref FFT_MUTEX: Mutex<()> = Mutex::new(());
}
 
#[test]
fn test_qap_setup() {
    let s_max = 64;
    let l_pub = 65;
    let start1 = Instant::now();
    println!("start loading...");

    let qap = QAP::load_from_json("output/qap.bin").unwrap();
    let lap = start1.elapsed();
    println!("The qap load time: {:.6} seconds", lap.as_secs_f64());

    let l0_y =thread_safe_compute_langrange_i_poly(0,1, s_max);
    for j in 0..64 {
        let m_j_x =thread_safe_compute_langrange_i_poly(j,l_pub-1,1);

        //lookup with alpha x^i y^j
        if !qap.u_j_X[j].is_zero() {
            // let l0y_mul_ujx = qap.u_j_X[i].mul(&l0_y);
        }
        //lookup with alpha^2 x^i y^j
        if !qap.v_j_X[j].is_zero() {
            //let l0y_mul_vjx = qap.v_j_X[i].mul(&l0_y);
        }
        //lookup with alpha^3 x^i y^j
        if !qap.w_j_X[j].is_zero() {
            //let l0y_mul_wjx = qap.w_j_X[i].mul(&l0_y);
            println!("not zero");
           // qap.w_j_X[j].print();
        }

        //sum them
    }

}
#[test]
fn test_eval_lagrange_bases() {
    let g1_gen = CurveCfg::generate_random_affine_points(1)[0];
    let mut tau = Tau::gen();

    const S_MAX: usize = 128;

    let prev_x = [G1serde(g1_gen); S_MAX];
    let x_powers = compute_powers(tau.x, S_MAX);
    let cur_x: Vec<G1serde> = prev_x
        .iter()
        .zip(x_powers.iter())
        .map(|(x, scalar)| x.mul(*scalar))
        .collect();

    let mut result = vec![G1serde::zero(); S_MAX];
    eval_langrange_bases(&cur_x, &mut result);

    let mut x_evaled_vec = vec![ScalarField::zero(); S_MAX].into_boxed_slice();
    gen_evaled_lagrange_bases(&tau.x, S_MAX, &mut x_evaled_vec);

    let mut x_evaledCommit = vec![G1serde::zero(); S_MAX].into_boxed_slice();
    from_coef_vec_to_g1serde_vec(&x_evaled_vec, &g1_gen, &mut x_evaledCommit);

    assert_eq!(result.into_boxed_slice(), x_evaledCommit);
}
pub(crate) fn thread_safe_compute_langrange_i_poly(i: usize, max_x: usize, max_y: usize) -> DensePolynomialExt{
    let _guard = FFT_MUTEX.lock().unwrap(); // Lock before unsafe call

    let mut lag_coeffs = vec![ScalarField::zero(); max_x*max_y];
    compute_langrange_i_coeffs(i, max_x, max_y, &mut lag_coeffs);
    // Mutex guard dropped here
     DensePolynomialExt::from_coeffs(HostSlice::from_slice(&lag_coeffs), max_x, max_y)
 }
pub(crate) fn thread_safe_compute_langrange_i_coeffs(i: usize, max_x: usize, max_y: usize, res: &mut [ScalarField]) {
    let _guard = FFT_MUTEX.lock().unwrap(); // Lock before unsafe call
    compute_langrange_i_coeffs(i, max_x, max_y, res);
    // Mutex guard dropped here
}
// given xG1 = [x^i*G1, x^i*G1, ..., x_max^i*G1]
// it evaluates the lagrange bases foreach i = 0, ..., s_max-1
// return x_evaled_vec = [x_0^i, x_1^i, ..., x_s_max^i]
pub fn eval_langrange_bases(xG1: &Vec<G1serde>, x_evaled_vec: &mut Vec<G1serde>) {
    let s_max = xG1.len();
    assert_eq!(x_evaled_vec.len(), s_max);

    x_evaled_vec
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
            let mut lag_coeffs = vec![ScalarField::zero(); s_max];

            thread_safe_compute_langrange_i_coeffs(i, s_max, 1, &mut lag_coeffs);
            // evaluate lagrange base for i
            let result = xG1
                .iter()
                .zip(lag_coeffs.iter())
                .fold(G1serde::zero(), |acc, (x_g1, coeff_i)| {
                    acc.add(x_g1.mul(*coeff_i))
                });

            *out = result;
        });
}
// TODO update to accumulator [x^0,X^1,...x^len_x-1]
fn compute_powers(x_r: ScalarField, len_x: usize) -> Vec<ScalarField> {
    let mut x_powers: Vec<ScalarField> = Vec::with_capacity(len_x);
    let mut current_power = ScalarField::one();
    for i in 0..len_x {
        x_powers.push(current_power);
        current_power = current_power.mul(x_r);
    }
    x_powers
}

fn compute_langrange_i_coeffs(i: usize, max_x: usize, max_y: usize, res: &mut [ScalarField]) {
    let mut l_evals = vec![ScalarField::zero(); max_x * max_y];
    l_evals[i] = ScalarField::one();
    let lagrange_L_XY = DensePolynomialExt::from_rou_evals(
        HostSlice::from_slice(&l_evals),
        max_x,
        max_y,
        None,
        None,
    );
    let cached_val_pows = HostSlice::from_mut_slice(res);
    lagrange_L_XY.copy_coeffs(0, cached_val_pows);
}
