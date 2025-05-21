use icicle_bls12_381::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{Arithmetic, FieldImpl};
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};
use libs::bivariate_polynomial::{BivariatePolynomial, DensePolynomialExt};
use libs::field_structures::Tau;
use libs::group_structures::G1serde;
use libs::iotools::{SetupParams, SubcircuitInfo};
use mpc_setup::accumulator::Accumulator;
use mpc_setup::conversions::icicle_g1_generator;
use mpc_setup::mpc_utils::{
    thread_safe_compute_langrange_i_coeffs, thread_safe_compute_langrange_i_poly,
    poly_mult,
};
pub use mpc_setup::prepare::QAP;
use mpc_setup::utils::next_random;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use std::ops::{Mul, Sub};
use std::time::Instant;
 use ark_ff::Zero;
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::msm;
use icicle_core::msm::{msm, MSMConfig};

fn mains() {
    // Elements of the form {x^h y^i}_{h=0,i=0}^{max(2n-2,3m_D-3),2*s_max-2}
    //xy_powers: Box<[G1serde]>,
    //gamma_inv_o_inst: Box<[G1serde]>, // {γ^(-1)(L_t(y)o_j(x) + M_j(x))}_{t=0,j=0}^{1,l-1} where t=0 for j∈[0,l_in-1] and t=1 for j∈[l_in,l-1]
    let mut tau = Tau::gen();

    // Load setup parameters from JSON file
    let setup_file_name = "setupParams.json";
    let mut setup_params = SetupParams::from_path(setup_file_name).unwrap();

    // Extract key parameters from setup_params
    let m_d = setup_params.m_D; // Total number of wires
    let s_d = setup_params.s_D; // Number of subcircuits
    let n = setup_params.n; // Number of constraints per subcircuit
    let s_max = setup_params.s_max; // The maximum number of placements.
                                    // Additional wire-related parameters
    let l = setup_params.l; // Number of public I/O wires
    let l_pub = setup_params.l_pub_in + setup_params.l_pub_out;
    let l_prv = setup_params.l_prv_in + setup_params.l_prv_out;
    let l_d = setup_params.l_D; // Number of interface wires
                                // The last wire-related parameter
    let m_i = l_d - l;
    println!(
        "Setup parameters: \n n = {:?}, \n s_max = {:?}, \n l = {:?}, \n m_I = {:?}, \n m_D = {:?}",
        n, s_max, l, m_i, m_d
    );

    // Verify n is a power of two
    if !n.is_power_of_two() {
        panic!("n is not a power of two.");
    }

    if !(l_pub.is_power_of_two() || l_pub == 0) {
        panic!("l_pub is not a power of two.");
    }

    if !(l_prv.is_power_of_two()) {
        panic!("l_prv is not a power of two.");
    }

    // Verify s_max is a power of two
    if !s_max.is_power_of_two() {
        panic!("s_max is not a power of two.");
    }

    // Verify m_I is a power of two
    if !m_i.is_power_of_two() {
        panic!("m_I is not a power of two.");
    }

    // Load subcircuit information
    let subcircuit_file_name = "subcircuitInfo.json";
    let subcircuit_infos = SubcircuitInfo::from_path(subcircuit_file_name).unwrap();

    /* let qap = QAP::gen_from_R1CS(&subcircuit_infos, &setup_params);
    qap.save_to_json("setup/mpc-setup/output/qap_all.bin").expect("cannot qap save to a json file");
    let qap2 = QAP::load_from_json("setup/mpc-setup/output/qap_all.bin").unwrap();
    println!("{}", qap.compare(&qap2));*/

    let start1 = Instant::now();
    println!("start loading...");

    let qap = QAP::load_from_json("setup/mpc-setup/output/qap.bin").unwrap();
    let lap = start1.elapsed();
    println!("The qap load time: {:.6} seconds", lap.as_secs_f64());
    //cargo run --release --bin prepare_phase2
}

/*
{
  "l": 288,
  "l_pub_in": 16,
  "l_pub_out": 16,
  "l_prv_in": 240,
  "l_prv_out": 16,
  "l_D": 544,
  "m_D": 5116,
  "n": 2048,
  "s_D": 15,
  "s_max": 64
}

    pub gamma_inv_o_inst: Box<[G1serde]>, // {γ^(-1)(L_t(y)o_j(x) + M_j(x))}_{t=0,j=0}^{1,l-1} where t=0 for j∈[0,l_in-1] and t=1 for j∈[l_in,l-1]
    pub eta_inv_li_o_inter_alpha4_kj: Box<[Box<[G1serde]>]>, // {η^(-1)L_i(y)(o_{j+l}(x) + α^4 K_j(x))}_{i=0,j=0}^{s_max-1,m_I-1}
    pub delta_inv_li_o_prv: Box<[Box<[G1serde]>]>, // {δ^(-1)L_i(y)o_j(x)}_{i=0,j=l+m_I}^{s_max-1,m_I-1}
    pub delta_inv_alphak_xh_tx: Box<[Box<[G1serde]>]>, // {δ^(-1)α^k x^h t_n(x)}_{h=0,k=1}^{2,3}
    pub delta_inv_alpha4_xj_tx: Box<[G1serde]>, // {δ^(-1)α^4 x^j t_{m_I}(x)}_{j=0}^{1}
    pub delta_inv_alphak_yi_ty: Box<[Box<[G1serde]>]>, // {δ^(-1)α^k y^i t_{s_max}(y)}_{i=0,k=1}^{2,4}
*/

fn main() {
    let acc = Accumulator::load_from_json("setup/mpc-setup/output/new_challenge.json")
        .expect("cannot accumulator read from file");
    let mut tau = Tau::gen();
    tau.x = ScalarField::from_u32(3);
    tau.y = ScalarField::from_u32(5);
    tau.alpha = ScalarField::from_u32(7);
    tau.gamma = ScalarField::from_u32(1);
    tau.delta = ScalarField::from_u32(1);
    tau.eta = ScalarField::from_u32(1);
    
    let xG1 = icicle_g1_generator().mul(tau.x);
    let yG1 = icicle_g1_generator().mul(tau.y);
    let alphaG1 = icicle_g1_generator().mul(tau.alpha);

    assert_eq!(xG1, acc.get_x_g1(1));
    assert_eq!(yG1, acc.get_y_g1(1));
    assert_eq!(alphaG1, acc.get_alpha_g1(1));

    println!("params are good for testing");

    let setup_file_name = "setupParams.json";
    let setup_params = SetupParams::from_path(setup_file_name).unwrap();
    let n = setup_params.n;     // Number of constraints per subcircuit

    let s_max = setup_params.s_max;
    let m_d = setup_params.m_D; // Total number of wires
    let l_pub_out = setup_params.l_pub_out;
    let l = setup_params.l; // Number of public I/O wires
    let l_pub = setup_params.l_pub_in + setup_params.l_pub_out;
    let l_prv = setup_params.l_prv_in + setup_params.l_prv_out;
    let l_prv_out = setup_params.l_prv_out;
    let l_prv_in = setup_params.l_prv_in;
    let l_d = setup_params.l_D; // Number of interface wires
    let m_i = l_d - l;
    println!("Setup parameters: \n n = {:?}, \n s_max = {:?}, \n l = {:?}, \n m_I = {:?}, \n m_D = {:?}", n, s_max, l, m_i, m_d);


    let start1 = Instant::now();
    println!(
        "start loading... with smax  {} and l_pub_out {}",
        s_max, l_pub_out
    );

    let qap = QAP::load_from_json("setup/mpc-setup/output/qap_all.bin").unwrap();
    let lap = start1.elapsed();
    println!("The qap load time: {:.6} seconds", lap.as_secs_f64());
    let mut li_y_vec: Vec<DensePolynomialExt> = vec![];
    for i in 0..s_max {
        let li_y = thread_safe_compute_langrange_i_poly(i, 1, s_max);
        li_y_vec.push(li_y);
    }
   
    // {γ^(-1)(L_t(y)o_j(x) + M_j(x))}_{t=0,j=0}^{1,l-1} where t=0 for j∈[0,l_in-1] and t=1 for j∈[l_in,l-1]
    let mut gamma_inv_o_inst = vec![G1serde::zero(); l];
    let l1_y = thread_safe_compute_langrange_i_poly(1, 1, s_max);
    let l1_y_y_size = l1_y.y_size; // Cached to avoid borrow checker issues
    let xiG1s = acc.get_x_g1_range(0, l_pub - 1);

    for j in 0..l_pub_out {
        //lookup with alpha x^i y^j
        if !qap.u_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * l1_y_y_size];
            poly_mult(&qap.u_j_X[j], &l1_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, l1_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^2 x^i y^j
        if !qap.v_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * l1_y_y_size];
            poly_mult(&qap.v_j_X[j], &l1_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, l1_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^3 x^i y^j
        if !qap.w_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * l1_y_y_size];
            poly_mult(&qap.w_j_X[j], &l1_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, l1_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }

        let mut m_j_x_coeffs = vec![ScalarField::zero(); l_pub];
        thread_safe_compute_langrange_i_coeffs(j, l_pub, 1, &mut m_j_x_coeffs);

        gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&m_j_x_coeffs, &xiG1s);
        println!("result[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
    }

    let l0_y =thread_safe_compute_langrange_i_poly(0, 1, s_max);
    let l0_y_y_size = l0_y.y_size; // Cached to avoid borrow checker issues

    for j in l_pub_out..l_pub {
        //lookup with alpha x^i y^j
        if !qap.u_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * l0_y_y_size];
            poly_mult(&qap.u_j_X[j], &l0_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, l0_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^2 x^i y^j
        if !qap.v_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * l0_y_y_size];
            poly_mult(&qap.v_j_X[j], &l0_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, l0_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^3 x^i y^j
        if !qap.w_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * l0_y_y_size];
            poly_mult(&qap.w_j_X[j], &l0_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, l0_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }

        let mut m_j_x_coeffs = vec![ScalarField::zero(); l_pub];
        thread_safe_compute_langrange_i_coeffs(j, l_pub, 1, &mut m_j_x_coeffs);

        gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&m_j_x_coeffs, &xiG1s);
        println!("result[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
    }

    let l3_y =thread_safe_compute_langrange_i_poly(3, 1, s_max);
    let l3_y_y_size = l3_y.y_size; // Cached to avoid borrow checker issues

    for j in l_pub..(l_pub+l_prv_out) {
        //lookup with alpha x^i y^j
        if !qap.u_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * l3_y_y_size];
            poly_mult(&qap.u_j_X[j], &l3_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, l3_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^2 x^i y^j
        if !qap.v_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * l3_y_y_size];
            poly_mult(&qap.v_j_X[j], &l3_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, l3_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^3 x^i y^j
        if !qap.w_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * l3_y_y_size];
            poly_mult(&qap.w_j_X[j], &l3_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, l3_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        println!("result[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
    }

    let l2_y =thread_safe_compute_langrange_i_poly(2, 1, s_max);
    let l2_y_y_size = l2_y.y_size; // Cached to avoid borrow checker issues

    for j in (l_pub+l_prv_out)..l {
        //lookup with alpha x^i y^j
        if !qap.u_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * l2_y_y_size];
            poly_mult(&qap.u_j_X[j], &l2_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, l2_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^2 x^i y^j
        if !qap.v_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * l2_y_y_size];
            poly_mult(&qap.v_j_X[j], &l2_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, l2_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        //lookup with alpha^3 x^i y^j
        if !qap.w_j_X[j].is_zero() {
            let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * l2_y_y_size];
            poly_mult(&qap.w_j_X[j], &l2_y, &mut multpxy_coeffs);
            let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, l2_y_y_size);
            gamma_inv_o_inst[j] = gamma_inv_o_inst[j] + sum_vector_dot_product(&multpxy_coeffs, &alphaxyG1s);
        }
        println!("result[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
    }

    // {δ^(-1)L_i(y)o_j(x)}_{i=0,j=l+m_I}^{s_max-1,m_I-1}
    let mut delta_inv_li_o_prv = vec![G1serde::zero(); (m_d-(l+m_i))*s_max];
    //for private wires,
    let mut idx = 0;
    let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[0].x_size * li_y_vec[0].y_size];

    let alpha1xyG1s = acc.get_alphaxy_g1_range(1, 2048, li_y_vec[0].y_size);
    let alpha2xyG1s = acc.get_alphaxy_g1_range(2, 2048, li_y_vec[0].y_size);
    let alpha3xyG1s = acc.get_alphaxy_g1_range(3, 2048, li_y_vec[0].y_size);

    /*for j in (l+m_i)..m_d {
       for i in 0..s_max {
            let li_y =&li_y_vec[i];
            //Li(y)oj(x)
            //lookup with alpha x^i y^j
            if !qap.u_j_X[j].is_zero() {
              //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.u_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, li_y.y_size);
                delta_inv_li_o_prv[idx] = delta_inv_li_o_prv[idx] + sum_vector_dot_product(&multpxy_coeffs, &alpha1xyG1s);
            }
            //lookup with alpha^2 x^i y^j
            if !qap.v_j_X[j].is_zero() {
               // let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.v_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, li_y.y_size);
                delta_inv_li_o_prv[idx] = delta_inv_li_o_prv[idx] + sum_vector_dot_product(&multpxy_coeffs, &alpha2xyG1s);
            }
            //lookup with alpha^3 x^i y^j
            if !qap.w_j_X[j].is_zero() {
              //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.w_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, li_y.y_size);
                delta_inv_li_o_prv[idx] = delta_inv_li_o_prv[idx] + sum_vector_dot_product(&multpxy_coeffs, &alpha3xyG1s);
            }
           idx = idx + 1;
        }

    }*/

    // {δ^(-1)α^k x^h t_n(x)}_{h=0,k=1}^{2,3}
   /* let mut delta_inv_alphak_xh_tx = vec![G1serde::zero(); 3 * 3];
     for k in 1..=3 {
        for h in 0..=2 {
            //let alpha^k x^{n+h}
            let alphak_xnh = acc.get_alphax_g1(k,n + h);
            //let alpha^k x^{h}
            let alphak_xh = acc.get_alphax_g1(k, h);

            let idx = (k-1) * 3 + h;
            delta_inv_alphak_xh_tx[idx] = alphak_xnh.sub(alphak_xh);
            println!("result[{}] = {:?} k={} h={} n={}", idx,delta_inv_alphak_xh_tx[idx].0.x, k, h,n );
        }
    }*/
    
     
    /*
    pub delta_inv_alphak_xh_tx: Box<[Box<[G1serde]>]>, // {δ^(-1)α^k x^h t_n(x)}_{h=0,k=1}^{2,3}
    pub delta_inv_alpha4_xj_tx: Box<[G1serde]>, // {δ^(-1)α^4 x^j t_{m_I}(x)}_{j=0}^{1}
    pub delta_inv_alphak_yi_ty: Box<[Box<[G1serde]>]>, // {δ^(-1)α^k y^i t_{s_max}(y)}_{i=0,k=1}^{2,4}
    */
    let lap = start1.elapsed();
    println!("The total time: {:.6} seconds", lap.as_secs_f64());
}

/// Computes the multi-scalar multiplication, or MSM: s1*P1 + s2*P2 + ... + sn*Pn, or a batch of several MSMs.
fn sum_vector_dot_product(scalars: &Vec<ScalarField>, commit: &[G1Affine]) -> G1serde {
    let mut msm_res = vec![G1Projective::zero(); 1];
    msm::msm(
        HostSlice::from_slice(&scalars),
        HostSlice::from_slice(&commit),
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut msm_res)
    ).unwrap();

    G1serde(G1Affine::from(msm_res[0]))
}
