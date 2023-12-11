use std::env;
use rustdf::data::simulation::{TimsTofSynthetics};
use std::path::Path;
fn main() {
    let _args: Vec<String> = env::args().collect();

    let db_path_str = "/home/administrator/Documents/promotion/rust/notebook/simulation/sim-data/EXP-MIDIDA-SMALL/experiment_data.db";
    let path = Path::new(db_path_str);
    let experiment = TimsTofSynthetics::new(path).unwrap();
    let first_frames = experiment.precursor_frames.iter().map(|x| x.frame_id.clone()).take(500).collect::<Vec<_>>();

    for frame in first_frames {
        println!("frame_id: {}", frame);
        let build_frame = experiment.build_frame(frame);
        println!("frame: {}", build_frame);
    }
}
