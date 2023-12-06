use std::env;
use rustdf::data::simulation::{TimsTofSyntheticsDDA};
use std::path::Path;
fn main() {
    let _args: Vec<String> = env::args().collect();

    let db_path_str = "/home/administrator/Documents/promotion/rust/notebook/simulation/EXP-SHORT/experiment_data.db";
    let path = Path::new(db_path_str);
    let experiment = TimsTofSyntheticsDDA::new(path).unwrap();
    let first_five_precursor_frames = experiment.precursor_frames.iter().map(|x| x.frame_id.clone()).take(150).collect::<Vec<_>>();
    let frames = experiment.build_frames(first_five_precursor_frames, 16);

    for frame in frames {
        println!("frame_id: {}", frame.frame_id);
        println!("mz: {:?}", frame.ims_frame.mz);
    }
}
