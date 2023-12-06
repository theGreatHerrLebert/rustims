use std::env;
use rustdf::data::simulation::{SyntheticsDataHandle};
use std::path::Path;

fn main() {
    let _args: Vec<String> = env::args().collect();

    let db_path_str = "/home/administrator/Documents/promotion/rust/notebook/simulation/EXP-SHORT/experiment_data.db";
    let path = Path::new(db_path_str);
    let handle = SyntheticsDataHandle::new(path).unwrap();
    let ions = handle.read_ions().unwrap();
    for ion in ions {
        println!("{:?}", ion);
    }
}
