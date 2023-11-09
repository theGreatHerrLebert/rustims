#[derive(Clone)]
pub struct TimsDDAPrecursor {
    pub id: u32,
    pub parent_id: u32,
    pub scan_num: u32,
    pub mz_average: f32,
    pub mz_most_intense: f32,
    pub intensity: f32,
    pub mz_mono_isotopic: Option<f32>,
    pub charge: Option<u32>,
}