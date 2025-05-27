use std::fs::File;
use std::io::Write;
use bellman_ce::groth16::Parameters;
use bellman_ce::pairing::Engine;

pub fn save_parameters<E: Engine>(params: &Parameters<E>, path: &str) {
    let mut file = File::create(path).expect("Cannot create file");
    params.write(&mut file).expect("Cannot write parameters");
}