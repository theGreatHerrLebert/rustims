use std::env;
use rustdf::data::meta::{read_dda_precursor_meta};


fn main() {
    let _args: Vec<String> = env::args().collect();

    let data_path = "/media/hd01/CCSPred/M210115_001_Slot1-1_1_850.d";

    let result = read_dda_precursor_meta(data_path);

    match result {
        Ok(precursors) => {
            println!("Precursors: {:?}", precursors);
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }

}
