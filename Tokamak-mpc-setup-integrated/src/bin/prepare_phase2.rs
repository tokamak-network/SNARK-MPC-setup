use ark_ff::Zero;
use icicle_bls12_381::curve::{G1Affine, G1Projective, ScalarField};
use icicle_bls12_381::polynomials::DensePolynomial;
use icicle_core::msm;
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::{Arithmetic, FieldImpl};
use icicle_runtime::memory::{HostOrDeviceSlice, HostSlice};
use libs::bivariate_polynomial::{BivariatePolynomial, DensePolynomialExt};
use libs::field_structures::Tau;
use libs::group_structures::{G1serde, Sigma, Sigma1, Sigma2};
use libs::iotools::{SetupParams, SubcircuitInfo};
use mpc_setup::accumulator::Accumulator;
use mpc_setup::conversions::{icicle_g1_generator, icicle_g2_generator};
use mpc_setup::mpc_utils::{
    poly_mult, thread_safe_compute_langrange_i_coeffs, thread_safe_compute_langrange_i_poly,
};
pub use mpc_setup::prepare::QAP;
use mpc_setup::utils::next_random;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use std::ops::{Mul, Sub};
use std::time::Instant;

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
// Calculate max(2n-2, 3m_I-3) for h upper bound
        let h_max = std::cmp::max(2*n, 2*m_i);

*/

fn main() {
    let acc = Accumulator::load_from_json("setup/mpc-setup/output/new_challenge.json")
        .expect("cannot accumulator read from file");
    
    let sigma = Sigma::read_from_json("setup/mpc-setup/output/combined_sigma_o.json")
        .expect("cannot read sigma from file");
    let xypowers = acc.get_boxed_xypower();
    assert_eq!(sigma.sigma_1.xy_powers, xypowers);

    
    let mut tau = Tau::gen();
    tau.x = ScalarField::from_u32(3);
    tau.y = ScalarField::from_u32(5);
    tau.alpha = ScalarField::from_u32(7);
    tau.gamma = ScalarField::from_u32(1);
    tau.delta = ScalarField::from_u32(1);
    tau.eta = ScalarField::from_u32(1);

    println!("x {}", acc.x.len());
    println!("y {}", acc.y.len_g1());

    let g1 = icicle_g1_generator();
    let g2 = icicle_g2_generator();

    let xG1 = g1.mul(tau.x);
    let yG1 = g1.mul(tau.y);
    let alphaG1 = g1.mul(tau.alpha);

    assert_eq!(xG1, acc.get_x_g1(1));
    assert_eq!(yG1, acc.get_y_g1(1));
    assert_eq!(alphaG1, acc.get_alpha_g1(1));

    println!("params are good for testing");

    let setup_file_name = "setupParams.json";
    let setup_params = SetupParams::from_path(setup_file_name).unwrap();
    let n = setup_params.n; // Number of constraints per subcircuit

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
    println!(
        "Setup parameters: \n n = {:?}, \n s_max = {:?}, \n l = {:?}, \n m_I = {:?}, \n m_D = {:?}",
        n, s_max, l, m_i, m_d
    );

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
        println!("gamma_inv_o_inst[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
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
        println!("gamma_inv_o_inst[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
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
        println!("gamma_inv_o_inst[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
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
        println!("gamma_inv_o_inst[{}] = {:?}", j, gamma_inv_o_inst[j].0.x);
    }

    // {δ^(-1)L_i(y)o_j(x)}_{i=0,j=l+m_I}^{s_max-1,m_I-1}
    let mut delta_inv_li_o_prv = vec![vec![G1serde::zero(); s_max].into_boxed_slice(); (m_d - (l + m_i))].into_boxed_slice();
    //for private wires,
     let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[0].x_size * li_y_vec[0].y_size];

    let alpha1xyG1s = acc.get_alphaxy_g1_range(1, 2048, li_y_vec[0].y_size);
    let alpha2xyG1s = acc.get_alphaxy_g1_range(2, 2048, li_y_vec[0].y_size);
    let alpha3xyG1s = acc.get_alphaxy_g1_range(3, 2048, li_y_vec[0].y_size);

    for j in (l+m_i)..m_d {
       for i in 0..s_max {
            let li_y =&li_y_vec[i];
           let mut out = G1serde::zero();
            //Li(y)oj(x)
            //lookup with alpha x^i y^j
            if !qap.u_j_X[j].is_zero() {
              //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.u_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha1xyG1s);
            }
            //lookup with alpha^2 x^i y^j
            if !qap.v_j_X[j].is_zero() {
               // let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.v_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha2xyG1s);
            }
            //lookup with alpha^3 x^i y^j
            if !qap.w_j_X[j].is_zero() {
              //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.w_j_X[j], &li_y, &mut multpxy_coeffs);
              //  let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha3xyG1s);
            }
           delta_inv_li_o_prv[j-(l+m_i)][i] = out;
            println!("delta_inv_li_o_prv[{}][{}] = {:?}", j-(l+m_i),i,out.0.x);
        }

    }

    // {δ^(-1)α^k x^h t_n(x)}_{h=0,k=1}^{2,3}
     let mut delta_inv_alphak_xh_tx =
        vec![vec![G1serde::zero(); 3].into_boxed_slice(); 3].into_boxed_slice();
    for k in 1..=3 {
        for h in 0..=2 {
            //let alpha^k x^{n+h}
            let alphak_xnh = acc.get_alphax_g1(k, n + h);
            //let alpha^k x^{h}
            let alphak_xh = acc.get_alphax_g1(k, h);

             delta_inv_alphak_xh_tx[k-1][h] = alphak_xnh.sub(alphak_xh);
            println!(
                "delta_inv_alphak_xh_tx[{}][{}] = {:?}",
                k-1,h, delta_inv_alphak_xh_tx[k-1][h].0.x
            );
        }
    }

    // {δ^(-1)α^4 x^j t_{m_I}(x)}_{j=0}^{1}
    let mut delta_inv_alpha4_xj_tx = vec![G1serde::zero(); 2];
    for j in 0..=1 {
        delta_inv_alpha4_xj_tx[j] = acc.get_alphax_g1(4, m_i + j);
        delta_inv_alpha4_xj_tx[j] = delta_inv_alpha4_xj_tx[j].sub(acc.get_alphax_g1(4, j));
        println!(
            "delta_inv_alpha4_xj_tx[{}] = {:?}",
            j, delta_inv_alpha4_xj_tx[j].0.x
        );
    }

    // {δ^(-1)α^k y^i t_{s_max}(y)}_{i=0,k=1}^{2,4}
    let mut delta_inv_alphak_yi_ty =
        vec![vec![G1serde::zero(); 3].into_boxed_slice(); 4].into_boxed_slice();
     for k in 1..=4 {
        for i in 0..=2 {
            delta_inv_alphak_yi_ty[k-1][i] = acc.get_alphay_g1(k, s_max + i).sub(acc.get_alphay_g1(k, i));

            println!(
                "delta_inv_alphak_yi_ty[{}][{}] = {:?}",
                k-1, i, delta_inv_alphak_yi_ty[k-1][i].0.x
            );
         }
    }

    // {η^(-1)L_i(y)(o_{j+l}(x) + α^4 K_j(x))}_{i=0,j=0}^{s_max-1,m_I-1}
    let mut eta_inv_li_o_inter_alpha4_kj = vec![vec![G1serde::zero(); m_i].into_boxed_slice(); s_max].into_boxed_slice();
    let mut kj_x_vec: Vec<DensePolynomialExt> = vec![];
    for i in 0..m_i {
        let kj_x = thread_safe_compute_langrange_i_poly(i, m_i, 1);
        kj_x_vec.push(kj_x);
    }
    let mut xy_coeffs = vec![ScalarField::zero(); li_y_vec[0].y_size * kj_x_vec[0].x_size];
    let alpha4xyG1s = acc.get_alphaxy_g1_range(4, kj_x_vec[0].x_size, li_y_vec[0].y_size);

    for j in 0..m_i {
        for i in 0..s_max {
            let li_y =&li_y_vec[i];
            let mut out = G1serde::zero();
            //Li(y)oj(x)
            //lookup with alpha x^i y^j
            if !qap.u_j_X[j+l].is_zero() {
                //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.u_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.u_j_X[j+l], &li_y, &mut multpxy_coeffs);
                //  let alphaxyG1s = acc.get_alphaxy_g1_range(1, qap.u_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha1xyG1s);
            }
            //lookup with alpha^2 x^i y^j
            if !qap.v_j_X[j+l].is_zero() {
                // let mut multpxy_coeffs = vec![ScalarField::zero(); qap.v_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.v_j_X[j+l], &li_y, &mut multpxy_coeffs);
                //  let alphaxyG1s = acc.get_alphaxy_g1_range(2, qap.v_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha2xyG1s);
            }
            //lookup with alpha^3 x^i y^j
            if !qap.w_j_X[j+l].is_zero() {
                //  let mut multpxy_coeffs = vec![ScalarField::zero(); qap.w_j_X[j].x_size * li_y.y_size];
                poly_mult(&qap.w_j_X[j+l], &li_y, &mut multpxy_coeffs);
                //  let alphaxyG1s = acc.get_alphaxy_g1_range(3, qap.w_j_X[j].x_size, li_y.y_size);
                out = out + sum_vector_dot_product(&multpxy_coeffs, &alpha3xyG1s);
            }

            poly_mult(li_y, &kj_x_vec[j], &mut xy_coeffs);
            out = out + sum_vector_dot_product(&xy_coeffs, &alpha4xyG1s);

            eta_inv_li_o_inter_alpha4_kj[j][i] = out;
            println!("eta_inv_li_o_inter_alpha4_kj[{}][{}] = {:?}", j,i,out.0.x);
        }
    }


    let sigma = Sigma{
        G: g1,
        H: g2,
        sigma_1:Sigma1{
            xy_powers: acc.get_boxed_xypower(),
            x: acc.get_x_g1(1),
            y: acc.get_y_g1(1),
            delta: g1,
            eta: g1,
            gamma_inv_o_inst: gamma_inv_o_inst.into_boxed_slice(),
            eta_inv_li_o_inter_alpha4_kj,
            delta_inv_li_o_prv,
            delta_inv_alphak_xh_tx,
            delta_inv_alpha4_xj_tx: delta_inv_alpha4_xj_tx.into_boxed_slice(),
            delta_inv_alphak_yi_ty,
        },
        sigma_2: Sigma2 {
            alpha: acc.alpha[0].g2,
            alpha2: acc.alpha[1].g2,
            alpha3: acc.alpha[2].g2,
            alpha4: acc.alpha[3].g2,
            gamma: g2,
            delta: g2,
            eta: g2,
            x: acc.x[0].g2,
            y: acc.y.g2,
        },
    };

    sigma.write_into_json("setup/mpc-setup/output/combined_sigma.json").expect("cannot write sigma into json");

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
        HostSlice::from_mut_slice(&mut msm_res),
    )
    .unwrap();

    G1serde(G1Affine::from(msm_res[0]))
}
