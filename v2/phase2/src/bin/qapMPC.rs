extern crate bellman_ce;
extern crate pairing_ce;
extern crate rand;
extern crate serde;
extern crate serde_json;

use bellman_ce::{
    groth16::{generate_random_parameters, Parameters},
    Circuit, ConstraintSystem, SynthesisError,
};
use pairing_ce::bn256::{Bn256, Fr};
use pairing_ce::ff::Field;
use serde::Deserialize;
use rand::thread_rng;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct QAPPolynomials {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at_inputs: Vec<Vec<(String, usize)>>,
    bt_inputs: Vec<Vec<(String, usize)>>,
    ct_inputs: Vec<Vec<(String, usize)>>,
    at_aux: Vec<Vec<(String, usize)>>,
    bt_aux: Vec<Vec<(String, usize)>>,
    ct_aux: Vec<Vec<(String, usize)>>
}

struct MyCircuit {
    qap: QAPPolynomials,
}

// Convert BLS
impl Circuit<Bn256> for MyCircuit {
    fn synthesize<CS: ConstraintSystem<Bn256>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        for constraint in self.qap.at_inputs.iter().chain(self.qap.at_aux.iter()) {
            for (_val, idx) in constraint.iter() {
                let var = if *idx < self.qap.num_inputs {
                    CS::alloc_input(cs, || "input variable", || Ok(Fr::one()))?
                } else {
                    CS::alloc(cs, || "aux variable", || Ok(Fr::one()))?
                };

                cs.enforce(
                    || "dummy constraint",
                    |lc| lc + var,
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
            }
        }
        Ok(())
    }
}

fn main() {
    let file = File::open("full_circuit_qap.json").expect("File open error");
    let reader = BufReader::new(file);
    let qap: QAPPolynomials = serde_json::from_reader(reader).expect("Reading error");

    let rng = &mut thread_rng();

    let params: Parameters<Bn256> = generate_random_parameters(MyCircuit { qap }, rng)
        .expect("parameters cannot be generated");

    let params_file = File::create("qap.params").expect("File cannot be generated");
    params.write(params_file).expect("File cannot be written");

    println!("Succesfully generated: qap.params");
}
