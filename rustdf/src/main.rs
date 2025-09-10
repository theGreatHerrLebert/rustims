use rustdf::data::dia::TimsDatasetDIA;

fn main() {
    let data_handle = TimsDatasetDIA::new("", "/media/hd02/data/raw/dia/O24026/O240206_027_S1-B6_1_15503.d", false, false);
    // print first frame meta
    let first_frame = &data_handle.meta_data[0];
    println!("{:?}", first_frame);
}