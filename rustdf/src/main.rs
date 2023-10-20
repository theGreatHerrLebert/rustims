use rustdf::data::handle::TimsDataHandle;

fn main() {
    let args: Vec<String> = env::args().collect();

    // args[0] is always the path to the program itself
    if args.len() <= 1 {
        eprintln!("Please provide a frame id.");
        return;
    }

    let frame_id: u32 = match args[1].parse() {
        Ok(id) => id,
        Err(_) => {
            eprintln!("Invalid frame id provided.");
            return;
        }
    };

    println!("Frame ID: {}", frame_id);

    let data_path = "/media/hd01/CCSPred/M210115_001_Slot1-1_1_850.d";
    let bruker_lib_path = "/home/administrator/Documents/promotion/ENV/lib/python3.8/site-packages/opentims_bruker_bridge/libtimsdata.so";
    let tims_data = TimsDataHandle::new(bruker_lib_path, data_path);
    match tims_data {
        Ok(tims_data) => {
            let frame = tims_data.get_frame(frame_id);
        }

        Err(e) => println!("error: {}", e),
    };
}
