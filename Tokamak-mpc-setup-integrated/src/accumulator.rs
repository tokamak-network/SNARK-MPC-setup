use crate::utils::{compute5, icicle_g1_generator, icicle_g2_generator, verify5, PairSerde, Proof5, SerialSerde};
use blake2::{Blake2b, Digest};
use libs::group_structures::{G1serde, G2serde};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Accumulator {
    pub contributor_count: usize,
    pub alpha: Vec<PairSerde>,
    pub x: Vec<PairSerde>,
    pub y: SerialSerde,
    pub alpha_x: Vec<G1serde>,
    pub alpha_y: Vec<G1serde>,
    pub xy: Vec<G1serde>,
    pub alpha_xy: Vec<G1serde>,
    /// Keep parameters here as a marker
    #[serde(skip)]
    marker: std::marker::PhantomData<G2serde>,
}

impl Accumulator {
    pub fn new(power_alpha_length: usize, power_x_length: usize, power_y_length: usize) -> Self {
        let g1 = icicle_g1_generator();
        let g2 = icicle_g2_generator();
        let mut acc = Accumulator {
            alpha: vec![PairSerde::new(g1, g2); power_alpha_length],
            x: vec![PairSerde::new(g1, g2); power_x_length],
            y: SerialSerde::new(power_y_length),
            alpha_x: vec![g1; power_alpha_length * power_x_length], 
            alpha_y: vec![g1; power_alpha_length * power_y_length],
            xy: vec![g1; power_x_length * power_y_length],
            alpha_xy: vec![g1; power_alpha_length * power_x_length * power_y_length],
            contributor_count: 0,
            marker: Default::default(),
        };
        acc
    }

    pub fn compute(&self) -> (Accumulator, Proof5) {
        let (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, proof_5) =
            compute5(&self.alpha_xy, &self.xy, &self.alpha_x, &self.alpha_y, &self.alpha, &self.x, &self.y, &self.hash());

        let mut acc = Accumulator {
            alpha: cur_alpha,
            x: cur_x,
            y: cur_y,
            alpha_x: cur_alphax,
            alpha_y: cur_alphay,
            xy: cur_xy,
            alpha_xy: cur_alphaxy,
            marker: Default::default(),
            contributor_count: self.contributor_count + 1,
        };
        (acc, proof_5)
    }
    pub fn verify(&self, cur: &Accumulator, cur_proof: &Proof5) -> bool {
        verify5(&self.alpha, &self.x, &self.y, &cur.alpha_xy, &cur.xy, &cur.alpha_x, &cur.alpha_y, &cur.alpha, &cur.x, &cur.y, &cur_proof)
    }
    pub fn hash(&self) -> [u8; 32] {
        // Serialize without the hash field
        let serialized = bincode::serialize(&self)
            .expect("Serialization failed for Accumulator");

        // Calculate Blake2b hash (32-byte output)
        let hash = Blake2b::digest(&serialized);

        // Convert GenericArray into [u8; 32]
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash[..32]);
        result
    }
    /// Save the Accumulator to a JSON file
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
        let acc: Accumulator = serde_json::from_str(&json_str)
            .expect("JSON deserialization failed");
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_accumulator_serialization_roundtrip() {
        // Construct dummy data for the Accumulator struct
        let accumulator = Accumulator {
            alpha: vec![PairSerde::new(icicle_g1_generator(), icicle_g2_generator())],
            x: vec![PairSerde::new(icicle_g1_generator(), icicle_g2_generator())],
            y: SerialSerde::new(2),
            alpha_x: vec![icicle_g1_generator(); 2],
            alpha_y: vec![icicle_g1_generator(); 2],
            xy: vec![icicle_g1_generator(); 4],
            alpha_xy: vec![icicle_g1_generator(); 8],
            marker: std::marker::PhantomData,
            contributor_count: 0,
        };

        // Serialize the Accumulator to JSON
        let serialized = serde_json::to_string_pretty(&accumulator)
            .expect("Serialization of Accumulator failed");

        println!("Serialized Accumulator:\n{}", serialized);

        // Deserialize JSON back into Accumulator
        let deserialized: Accumulator = serde_json::from_str(&serialized)
            .expect("Deserialization of Accumulator failed");

        // Verify integrity (assuming Accumulator implements PartialEq)
        assert_eq!(accumulator.alpha, deserialized.alpha, "alpha mismatch");
        assert_eq!(accumulator.x, deserialized.x, "x mismatch");
        assert_eq!(accumulator.y, deserialized.y, "y mismatch");
        assert_eq!(accumulator.alpha_x, deserialized.alpha_x, "alpha_x mismatch");
        assert_eq!(accumulator.alpha_y, deserialized.alpha_y, "alpha_y mismatch");
        assert_eq!(accumulator.xy, deserialized.xy, "xy mismatch");
        assert_eq!(accumulator.alpha_xy, deserialized.alpha_xy, "alpha_xy mismatch");
        assert_eq!(accumulator.hash(), deserialized.hash(), "hash mismatch");
    }
    #[test]
    fn test_save_load_accumulator() {
        let accumulator = Accumulator::new(2, 4, 8);
        accumulator.save_to_json("accumulator.json").expect("Failed to save");

        let loaded_accumulator = Accumulator::load_from_json("accumulator.json").expect("Failed to load");
        println!("Loaded Accumulator: {:?}", loaded_accumulator);
        assert_eq!(accumulator, loaded_accumulator);
    }
}






