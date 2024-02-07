use rustdf::data::simulation::{TimsTofSyntheticsDIA};
use std::path::Path;
use clap::Parser;

/// Create synthetic DIA proteomics experiment data
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to synthetic_data.db
    #[arg(short, long)]
    path: String,

    /// Number of threads to use
    #[arg(short, long, default_value_t = 64)]
    num_threads: usize,

    /// Fragment the precursors into product ions
    #[arg(short, long, default_value_t = false)]
    fragment: bool,

    /// Batch size
    #[arg(short, long, default_value_t = 256)]
    batch_size: usize,

    /// Number of frames to process
    #[arg(short, long, default_value_t = 4096)]
    num_frames: usize,
}


fn main() {

    let args = Args::parse();

    let db_path_str = args.path;
    let path = Path::new(&db_path_str);
    let num_threads = args.num_threads;
    let fragment = args.fragment;

    let experiment = TimsTofSyntheticsDIA::new(path).unwrap();
    let first_frames = experiment.synthetics.frames.iter().map(|x| x.frame_id.clone()).take(args.num_frames).collect::<Vec<_>>();

    /*
    for frame in first_frames.iter() {
        let build_frame = experiment.build_frame(*frame, fragment);
        println!("frame_id: {}", build_frame.frame_id);
    }
    */

    // go over the frames in batches of 256
    for frame_batch in first_frames.chunks(args.batch_size) {
        let frames = experiment.build_frames(frame_batch.to_vec(), fragment, num_threads);

        for frame in frames {
            println!("frame_id: {}", frame.frame_id);
            println!("frame: {}", frame);
        }
    }
}
