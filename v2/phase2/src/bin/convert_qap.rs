extern crate pairing_ce;
extern crate serde;
extern crate serde_json;

use pairing_ce::bls12_381::Fr;
use pairing_ce::ff::PrimeField;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::str::FromStr;

#[derive(Debug, Clone)]
struct SerdeFr(pub Fr);

impl Serialize for SerdeFr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for SerdeFr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let fr = Fr::from_str(&s).ok_or_else(|| serde::de::Error::custom("Invalid Fr value"))?;
        Ok(SerdeFr(fr))
    }
}

#[derive(Serialize, Deserialize)]
struct EvaledQAP {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at: Vec<Vec<(String, usize)>>,
    bt: Vec<Vec<(String, usize)>>,
    ct: Vec<Vec<(String, usize)>>,
}

#[derive(Serialize, Deserialize)]
struct BellmanQAP {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at_inputs: Vec<Vec<(SerdeFr, usize)>>,
    bt_inputs: Vec<Vec<(SerdeFr, usize)>>,
    ct_inputs: Vec<Vec<(SerdeFr, usize)>>,
    at_aux: Vec<Vec<(SerdeFr, usize)>>,
    bt_aux: Vec<Vec<(SerdeFr, usize)>>,
    ct_aux: Vec<Vec<(SerdeFr, usize)>>,
}

fn convert_poly(poly: &[Vec<(String, usize)>]) -> Vec<Vec<(SerdeFr, usize)>> {
    poly.iter()
        .map(|terms| {
            terms
                .iter()
                .map(|(coeff, idx)| {
                    let fr = Fr::from_str(coeff).expect("Invalid Fr string");
                    (SerdeFr(fr), *idx)
                })
                .collect()
        })
        .collect()
}

fn main() {
    let file = File::open("full_circuit_qap.json").expect("Dosya açılamadı!");
    let reader = BufReader::new(file);
    let evaled_qap: EvaledQAP = serde_json::from_reader(reader).expect("JSON parse hatası!");

    let num_inputs = evaled_qap.num_inputs;

    let bellman_qap = BellmanQAP {
        num_inputs,
        num_aux: evaled_qap.num_aux,
        num_constraints: evaled_qap.num_constraints,
        at_inputs: convert_poly(&evaled_qap.at[..num_inputs]),
        bt_inputs: convert_poly(&evaled_qap.bt[..num_inputs]),
        ct_inputs: convert_poly(&evaled_qap.ct[..num_inputs]),
        at_aux: convert_poly(&evaled_qap.at[num_inputs..]),
        bt_aux: convert_poly(&evaled_qap.bt[num_inputs..]),
        ct_aux: convert_poly(&evaled_qap.ct[num_inputs..]),
    };

    let output_file = File::create("bellman_qap.json").expect("Çıktı dosyası açılamadı!");
    let writer = BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &bellman_qap).expect("Yazma hatası!");
    println!("Dönüşüm tamamlandı ve 'bellman_qap.json' olarak kaydedildi.");
}
