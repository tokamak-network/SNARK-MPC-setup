#!/bin/sh

rm challenge
rm response
rm new_challenge
rm challenge_old
rm response_old
rm phase1radix*
rm transcript

cargo run --release --bin initialize #for initialization

# for each participant
cargo run --release --bin compute_constrained
cargo run --release --bin verify_transform_constrained

mv challenge challenge_old
mv response response_old

mv new_challenge challenge
cargo run --release --bin beacon_constrained
cargo run --release --bin verify_transform_constrained

# cat response_old response > transcript
Get-Content response_old, response | Add-Content transcript

cargo run --release --bin prepare_phase2

cargo run --release --bin verify
