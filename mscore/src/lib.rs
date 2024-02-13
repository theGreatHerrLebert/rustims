// chemistry module
pub mod chemistry {
    pub mod atoms;
    pub mod amino_acids;
    pub mod unimod;
    pub mod aa_sequence;
    pub mod constants;
    pub mod formulas;
}

// algorithm module
pub mod algorithm {
    pub mod quadrupole;
    pub mod isotope_distributions;
    pub mod aa_sequence;
}

// data module
pub mod data {
    pub mod mz_spectrum;
    pub mod tims_frame;
    pub mod tims_slice;
}
