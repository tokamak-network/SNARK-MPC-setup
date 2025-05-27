use serde_json::Value;
use std::fs::File;
use std::io::BufReader;

pub fn load_json(file_path: &str) -> Value {
    let file = File::open(file_path).expect("Unable to open file");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("Unable to parse JSON")
}

pub fn load_all_subcircuits(paths: &[&str]) -> Vec<Value> {
    paths.iter().map(|path| load_json(path)).collect()
}