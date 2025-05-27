mod tokamak_integration;

use bellman_ce::pairing::bn256::Bn256;
use rand::thread_rng;
use tokamak_integration::{generate_mpc_parameters, save_parameters};

fn main() {
    let mut rng = thread_rng();

    let subcircuits = ["subcircuit0.json", "subcircuit1.json", "subcircuit2.json"];
    let global_wire_list = "globalWireList.json";
    let setup_params = "setupParams.json";
    let subcircuit_info = "subcircuitInfo.json";

    let params = generate_mpc_parameters::<Bn256, _>(
        &subcircuits,
        global_wire_list,
        setup_params,
        subcircuit_info,
        &mut rng
    ).expect("MPC parameter generation failed");

    save_parameters(&params, "mpc_tokamak.params");

    println!("Tokamak zkEVM MPC parameters successfully generated!");
}
