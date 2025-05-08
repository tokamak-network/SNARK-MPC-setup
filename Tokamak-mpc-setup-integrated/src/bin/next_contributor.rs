use mpc_setup::accumulator::Accumulator;
use mpc_setup::utils::Proof5;
use std::fs;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let cur_acc = Accumulator::load_from_json("output/new_challenge.json").expect("cannot accumulator read from file");
    println!("loading old challenge and proof...");

    if cur_acc.contributor_count > 0 {
        println!("previous contributor count: {}", cur_acc.contributor_count);
        let acc_old = Accumulator::load_from_json("output/old_challenge.json").expect("cannot accumulator read from file");
        let proof = Proof5::load_from_json("output/new_proof.json").expect("cannot proof read from file");

        assert_eq!(acc_old.verify(&cur_acc, &proof), true, "verification failed");
        let hash = hex::encode(acc_old.hash());
        fs::rename("output/old_challenge.json", format!("output/acc-{}.json", hash));
        fs::rename("output/old_proof.json", format!("output/proof-{}.json", hash));
        fs::rename("output/new_proof.json", "output/old_proof.json").expect("cannot rename new_proof.json");
    } else {
        //first contributor
        println!("previous contributor is genesis");
        let acc_check = Accumulator::new(cur_acc.alpha.len(), cur_acc.x.len(), cur_acc.y.len_g1());
        assert_eq!(cur_acc.hash(), acc_check.hash(), "genesis hash is not correct");
    }
    fs::rename("output/new_challenge.json", "output/old_challenge.json").expect("cannot rename new_challenge.json");
    println!("current contributor count: {}", cur_acc.contributor_count + 1);
    println!("computing new challenge and proof...");
    let (new_acc, new_proof) = cur_acc.compute();

    new_acc.save_to_json("output/new_challenge.json").expect("cannot write new_challenge to file");
    new_proof.save_to_json("output/new_proof.json").expect("cannot write new_proof to file");

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration.as_secs());
}
