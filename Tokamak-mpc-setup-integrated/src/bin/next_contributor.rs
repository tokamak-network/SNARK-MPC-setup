use mpc_setup::accumulator::Accumulator;
use mpc_setup::utils::Proof5;
use std::fs;
use std::time::Instant;
use mpc_setup::conversions::{icicle_g1_generator, icicle_g2_generator};

fn main() {
    let start = Instant::now();
    let cur_acc = Accumulator::load_from_json("setup/mpc-setup/output/new_challenge.json").expect("cannot accumulator read from file");
    println!("loading old challenge and proof...");

    if cur_acc.contributor_count > 0 {
        println!("previous contributor count: {}", cur_acc.contributor_count);
        let acc_old = Accumulator::load_from_json("setup/mpc-setup/output/old_challenge.json").expect("cannot accumulator read from file");
        let proof = Proof5::load_from_json("setup/mpc-setup/output/new_proof.json").expect("cannot proof read from file");

        assert_eq!(acc_old.verify(&cur_acc, &proof), true, "verification failed");
        let hash = hex::encode(acc_old.hash());
        fs::rename("setup/mpc-setup/output/old_challenge.json", format!("setup/mpc-setup/output/acc-{}.json", hash));
        fs::rename("setup/mpc-setup/output/old_proof.json", format!("setup/mpc-setup/output/proof-{}.json", hash));
        fs::rename("setup/mpc-setup/output/new_proof.json", "setup/mpc-setup/output/old_proof.json").expect("cannot rename new_proof.json");
    } else {
        //first contributor
        println!("previous contributor is genesis");
        let g1 = icicle_g1_generator();
        let g2 = icicle_g2_generator();
        let acc_check = Accumulator::new(g1,g2,cur_acc.alpha.len(), cur_acc.x.len(), cur_acc.y.len_g1());
        assert_eq!(cur_acc.hash(), acc_check.hash(), "genesis hash is not correct");
    }
    fs::rename("setup/mpc-setup/output/new_challenge.json", "setup/mpc-setup/output/old_challenge.json").expect("cannot rename new_challenge.json");
    println!("current contributor count: {}", cur_acc.contributor_count + 1);
    println!("computing new challenge and proof...");
    let (new_acc, new_proof) = cur_acc.compute();

    new_acc.save_to_json("setup/mpc-setup/output/new_challenge.json").expect("cannot write new_challenge to file");
    new_proof.save_to_json("setup/mpc-setup/output/new_proof.json").expect("cannot write new_proof to file");

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration.as_secs());
}
