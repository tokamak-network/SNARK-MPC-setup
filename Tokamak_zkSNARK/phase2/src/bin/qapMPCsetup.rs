use bellman_ce::pairing::ff::Field as BellmanField;
use icicle_bls12_381::curve::ScalarField;
use phase2::tools::{gen_cached_pows, MixedSubcircuitQAPEvaled, SetupParams, SubcircuitInfo, Tau};
use phase2::MPCParameters;
use serde::{Deserialize, Serialize};
use std::fs::File;

fn main() {
    let mut rng = rand::thread_rng();
    let tau = Tau::gen();

    let setup_params =
        SetupParams::from_path("setup/trusted-setup/inputs/setupParams.json").unwrap();
    let subcircuit_infos =
        SubcircuitInfo::from_path("setup/trusted-setup/inputs/subcircuitInfo.json").unwrap();

    let mut cached_x_pows_vec = vec![ScalarField::zero(); setup_params.n].into_boxed_slice();
    gen_cached_pows(&tau.x, setup_params.n, &mut cached_x_pows_vec);

    let mut combined_qap = MixedSubcircuitQAPEvaled {
        a_evals: vec![ScalarField::zero(); setup_params.n].into_boxed_slice(),
        b_evals: vec![ScalarField::zero(); setup_params.n].into_boxed_slice(),
        c_evals: vec![ScalarField::zero(); setup_params.n].into_boxed_slice(),
        z: ScalarField::zero(),
    };

    for (i, sub_info) in subcircuit_infos.iter().enumerate() {
        let path = format!("setup/trusted-setup/inputs/json/subcircuit{}.json", i);
        let sub_qap = MixedSubcircuitQAPEvaled::from_r1cs_to_evaled_qap(
            &subcircuit_path,
            &setup_params,
            sub_info,
            &tau,
            &cached_x_pows_vec,
        );
        // Logic to combine subcircuit QAPs
    }

    let evaled_circuit = PreEvaledQAPCircuit { evaled_qap: combined_qap };

    let rng = &mut rand::thread_rng();
    let mut mpc_params = MPCParameters::new(evaled_circuit).unwrap();
    mpc_params.contribute(rng);

    let mut file = File::create("groth16.params").unwrap();
    evaled_circuit.get_params().write(&mut f).unwrap();

    println!("Groth16 setup parameters successfully generated.");
}
