use ark_serialize::CanonicalSerialize;
use phase1::*;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write, Cursor};

fn main() {
    // Original functionality: Create and write to the challenge file
    let writer = OpenOptions::new()
        .read(false)
        .write(true)
        .create_new(true)
        .open("challenge")
        .expect("unable to create `./challenge`");

    let mut writer = BufWriter::new(writer);

    // Write a blank BLAKE2b hash
    let blank_hash = blank_hash();
    writer
        .write_all(blank_hash.as_slice())
        .expect("unable to write blank hash to `./challenge`");

    // Create a new accumulator and serialize it
    let acc = Accumulator::new();
    acc.serialize_uncompressed(&mut writer)
        .expect("unable to write fresh accumulator to `./challenge`");
    writer.flush().expect("unable to flush accumulator to disk");

    println!("Wrote a fresh accumulator to `./challenge`");

    // Added functionality: Print the data to be written in a readable way
    // Simulate the data being written to inspect it
    let mut buffer = Vec::new();

    // Write blank hash to the buffer
    buffer.extend_from_slice(blank_hash.as_slice());

    // Serialize the accumulator to the buffer
    let mut cursor = Cursor::new(&mut buffer);
    acc.serialize_uncompressed(&mut cursor)
        .expect("unable to serialize accumulator");

    // Print the data to the console
    println!("Blank hash (hex): {}", hex::encode(&buffer[..64])); // Assuming the hash is 64 bytes
    //println!(
    //    "Accumulator data (hex): {}",
    //    hex::encode(&buffer[64..]) // Rest of the buffer is the accumulator
    //);
}

/*use ark_serialize::CanonicalSerialize;
use phase1::*;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

fn main() {
    let writer = OpenOptions::new()
        .read(false)
        .write(true)
        .create_new(true)
        .open("challenge")
        .expect("unable to create `./challenge`");

    let mut writer = BufWriter::new(writer);

    // Write a blank BLAKE2b hash:
    writer
        .write_all(blank_hash().as_slice())
        .expect("unable to write blank hash to `./challenge`");

    let acc = Accumulator::new();
    acc.serialize_uncompressed(&mut writer)
        .expect("unable to write fresh accumulator to `./challenge`");
    writer.flush().expect("unable to flush accumulator to disk");

    println!("Wrote a fresh accumulator to `./challenge`");
}*/


/*use ark_serialize::CanonicalSerialize;
use phase1::*;

use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Write, Read};

fn main() {
    // Create and write to the challenge file
    let writer = OpenOptions::new()
        .read(false)
        .write(true)
        .create_new(true)
        .open("challenge")
        .expect("unable to create `./challenge`");

    let mut writer = BufWriter::new(writer);

    // Write a blank BLAKE2b hash:
    writer
        .write_all(blank_hash().as_slice())
        .expect("unable to write blank hash to `./challenge`");

    let acc = Accumulator::new();
    acc.serialize_uncompressed(&mut writer)
        .expect("unable to write fresh accumulator to `./challenge`");
    writer.flush().expect("unable to flush accumulator to disk");

    println!("Wrote a fresh accumulator to `./challenge`");

    // Read and print the content of the challenge file
    let mut file = File::open("challenge").expect("unable to open `./challenge`");
    let mut content = Vec::new();
    file.read_to_end(&mut content).expect("unable to read `./challenge`");
    println!("Content of `./challenge` (hex): {}", hex::encode(content));
}*/
