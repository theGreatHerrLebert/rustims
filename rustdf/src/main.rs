use std::env;
use rustdf::data::simulation::{TimsTofSyntheticsDIA};
use std::path::Path;

fn main() {
    let _args: Vec<String> = env::args().collect();

    let db_path_str = "/media/hd02/data/SIM/TEST-SIM-MONSTER-2/synthetic_data.db";
    let path = Path::new(db_path_str);
    let experiment = TimsTofSyntheticsDIA::new(path).unwrap();
    let first_frames = experiment.synthetics.frames.iter().map(|x| x.frame_id.clone()).take(4096).collect::<Vec<_>>();

    // go over the frames in batches of 256
    for frame_batch in first_frames.chunks(32) {
        let frames = experiment.build_frames(frame_batch.to_vec(), false, 64);
        for frame in frames {
            println!("frame_id: {}", frame.frame_id);
            println!("frame: {}", frame);
        }
    }
}
