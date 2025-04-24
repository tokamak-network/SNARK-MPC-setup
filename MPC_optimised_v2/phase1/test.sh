#!/bin/sh

rm challenge
rm response
rm new_challenge
rm challenge_old
rm response_old
rm phase1radix*
rm transcript

cargo clean
rm Cargo.lock
cargo build
#for initialization
cargo run --release --bin initialize 

# for each participant
cargo run --release --bin compute
cargo run --release --bin verify_transform

mv challenge challenge_old
mv response response_old
mv new_challenge challenge

#after participants
cargo run --release --bin beacon
cargo run --release --bin verify_transform

# make transcript file
# cat response_old response > transcript
Get-Content response_old, response | Add-Content transcript
# cat response_old response >> transcript


cargo run --release --bin prepare_phase2



# go to phase2 
rm pk.json
rm vk.json
rm qap.params
rm circom.params
rm mimc.params

cargo run --release --bin qapMPC

cargo run --release --bin mimc
cargo run --release --bin circom
