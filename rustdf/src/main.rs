use rust_tdf::data::handle::TimsDataset;

fn main() {
    let data_path = "/media/hd01/CCSPred/M210115_001_Slot1-1_1_850.d";
    let bruker_lib_path = "/home/administrator/Documents/promotion/rust/rust_tdf/libs/libtimsdata.so";
    let tims_data = TimsDataset::new(bruker_lib_path, data_path);
    match tims_data {
        Ok(tims_data) => {
            let frame = tims_data.get_frame(66073);
            match frame {
                Ok(frame) => println!("frame: {:?}", frame.0),
                Err(e) => println!("error: {}", e),
            };
        },
        Err(e) => println!("error: {}", e),
    };
}