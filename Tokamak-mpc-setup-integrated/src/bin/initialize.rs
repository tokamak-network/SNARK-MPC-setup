extern crate memmap;

use mpc_setup::accumulator::Accumulator;
use mpc_setup::conversions::{icicle_g1_generator, icicle_g2_generator};
use mpc_setup::utils::{compute5, verify5, PairSerde, SerialSerde};
use std::path::Path;
use std::time::Instant;
use std::{env, fs, io};

fn main() {
    let _ = clear_folder(Path::new("output"));

    let args: Vec<String> = env::args().collect();
    // Check if the correct number of arguments is provided
    if args.len() != 2 {
        eprintln!("Usage: {} <smax1>", args[0]);
        std::process::exit(1);
    }

    // Parse the argument into usize
    let power_x_length: usize = args[1].parse().expect("Invalid number provided for power_x_length");

    println!("Got power_x_length = {}", power_x_length);

    let start = Instant::now();

    let power_alpha_length: usize = 4;
    let power_y_length: usize = 2*power_x_length;

    let acc = Accumulator::new(power_alpha_length, power_x_length, power_y_length);

    acc.save_to_json("output/new_challenge.json").expect("cannot write to file");

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration.as_secs());
}
fn clear_folder(path: &Path) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        // Check if it's a file and delete it
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
fn test_compute5() {
    let start = Instant::now();
    //initialize
    let g1 = icicle_g1_generator();
    let g2 = icicle_g2_generator();


    let s_max0: usize = 4;  //alpha
    let s_max1: usize = 64; //x^i
    let s_max2: usize = 128; //y^k

    let v = [34u8; 32];
    let mut prev_alpha = vec![PairSerde::new(g1.clone(), g2.clone()); s_max0];
    let mut prev_x = vec![PairSerde::new(g1.clone(), g2.clone()); s_max1];
    let mut prev_y = SerialSerde::new(s_max2);
    let mut prev_xy = vec![g1; s_max1 * s_max2];
    let mut prev_alphax = vec![g1; s_max0 * s_max1];
    let mut prev_alphay = vec![g1; s_max0 * s_max2];

    let mut prev_alphaxy = vec![g1; s_max0 * s_max1 * s_max2];

    // first participant
    let (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, proof5) =
        compute5(&prev_alphaxy, &prev_xy, &prev_alphax, &prev_alphay, &prev_alpha, &prev_x, &prev_y, &v);

    assert_eq!(verify5(&prev_alpha,&prev_x,&prev_y,
               &cur_alphaxy,&cur_xy,&cur_alphax,&cur_alphay,&cur_alpha,&cur_x,&cur_y,&proof5), true,);

    prev_alpha = cur_alpha;
    prev_x = cur_x;
    prev_y = cur_y;
    prev_xy = cur_xy;
    prev_alphaxy = cur_alphaxy;
    prev_alphax = cur_alphax;
    prev_alphay = cur_alphay;

    // second participant
    let (cur_alphaxy, cur_xy, cur_alphax, cur_alphay, cur_alpha, cur_x, cur_y, proof5) =
        compute5(&prev_alphaxy, &prev_xy, &prev_alphax, &prev_alphay, &prev_alpha, &prev_x, &prev_y, &v);


    let duration = start.elapsed();

    println!("Time elapsed: {:?}", duration.as_secs());
}
