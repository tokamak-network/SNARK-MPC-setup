
extern crate rand;
extern crate crossbeam;
extern crate num_cpus;
extern crate blake2;
extern crate generic_array;
extern crate typenum;
extern crate byteorder;

extern crate pairing_ce;


use pairing_ce::{
    CurveAffine, CurveProjective, Engine, EncodedPoint,
    GroupDecodingError, Wnaf,
};
use pairing_ce::bls12_381::{Bls12, Fr, G1, G2, G1Affine, G2Affine};
use ff_ce::{Field, PrimeField};

use crate::parameters::{UseCompression, CheckForCorrectness};



use pairing_ce::bls12_381::*;
use pairing_ce::*;
use byteorder::{ReadBytesExt, BigEndian};
use rand::{SeedableRng, Rng, Rand};
use rand::chacha::ChaChaRng;



use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use generic_array::GenericArray;
use typenum::consts::U64;
use blake2::{Blake2b, Digest};
use std::fmt;

use crate::parameters::PowersOfTauParameters;

#[derive(Clone)]
pub struct Bls12CeremonyParameters;

impl PowersOfTauParameters for Bls12CeremonyParameters {
    const REQUIRED_POWER: usize = 7;

    const G1_UNCOMPRESSED_BYTE_SIZE: usize = 96;
    const G2_UNCOMPRESSED_BYTE_SIZE: usize = 192;
    const G1_COMPRESSED_BYTE_SIZE: usize = 48;
    const G2_COMPRESSED_BYTE_SIZE: usize = 96;
    
    const COMPRESSED_CHALLENGE_LENGTH: usize = 64;
    const NEW_CHALLENGE_LENGTH: usize = 64;

}
