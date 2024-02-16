use mscore::data::mz_spectrum::{MzSpectrum, MsType};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FragmentIon {
    pub mz: f64,
    pub kind: String,
    pub intensity: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FragmentIonSeries {
    pub charge: i32,
    pub b_ions: Vec<FragmentIon>,
    pub y_ions: Vec<FragmentIon>,
}

impl FragmentIonSeries {
    pub fn to_mz_spectrum(&self) -> MzSpectrum {

        // create a tuple vector from the fragment spectra
        let mut tuples = Vec::new();

        for ion in &self.b_ions {
            tuples.push((ion.mz, ion.intensity));
        }
        for ion in &self.y_ions {
            tuples.push((ion.mz, ion.intensity));
        }

        // sort the tuples by mz
        tuples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // create the mz and intensity vectors
        let mz = tuples.iter().map(|(m, _)| *m).collect();
        let intensity = tuples.iter().map(|(_, i)| *i).collect();

        MzSpectrum {
            mz,
            intensity,
        }

    }

}

#[derive(Debug, Clone)]
pub struct PeptidesSim {
    pub peptide_id: u32,
    pub sequence: String,
    pub proteins: String,
    pub decoy: bool,
    pub missed_cleavages: i8,
    pub n_term : Option<bool>,
    pub c_term : Option<bool>,
    pub mono_isotopic_mass: f32,
    pub retention_time: f32,
    pub events: f32,
    pub frame_occurrence: Vec<u32>,
    pub frame_abundance: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WindowGroupSettingsSim {
    pub window_group: u32,
    pub scan_start: u32,
    pub scan_end: u32,
    pub isolation_mz: f32,
    pub isolation_width: f32,
    pub collision_energy: f32,
}

impl WindowGroupSettingsSim {
    pub fn new(
        window_group: u32,
        scan_start: u32,
        scan_end: u32,
        isolation_mz: f32,
        isolation_width: f32,
        collision_energy: f32,
    ) -> Self {
        WindowGroupSettingsSim {
            window_group,
            scan_start,
            scan_end,
            isolation_mz,
            isolation_width,
            collision_energy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameToWindowGroupSim {
    pub frame_id: u32,
    pub window_group:u32,
}

impl FrameToWindowGroupSim {
    pub fn new(frame_id: u32, window_group: u32) -> Self {
        FrameToWindowGroupSim {
            frame_id,
            window_group,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IonsSim {
    pub peptide_id: u32,
    pub mono_isotopic_mass: f32,
    pub mz: f32,
    pub charge: i8,
    pub relative_abundance: f32,
    pub mobility: f32,
    pub simulated_spectrum: MzSpectrum,
    pub scan_occurrence: Vec<u32>,
    pub scan_abundance: Vec<f32>,
}

impl IonsSim {
    pub fn new(
        peptide_id: u32,
        mz: f32,
        mono_isotopic_mass: f32,
        charge: i8,
        relative_abundance: f32,
        mobility: f32,
        simulated_spectrum: MzSpectrum,
        scan_occurrence: Vec<u32>,
        scan_abundance: Vec<f32>,
    ) -> Self {
        IonsSim {
            peptide_id,
            mono_isotopic_mass,
            mz,
            charge,
            relative_abundance,
            mobility,
            simulated_spectrum,
            scan_occurrence,
            scan_abundance,
        }
    }
}


#[derive(Debug, Clone)]
pub struct ScansSim {
    pub scan: u32,
    pub mobility: f32,
}

impl ScansSim {
    pub fn new(scan: u32, mobility: f32) -> Self {
        ScansSim { scan, mobility }
    }
}

#[derive(Debug, Clone)]
pub struct FramesSim {
    pub frame_id: u32,
    pub time: f32,
    pub ms_type: i64,
}

impl FramesSim {
    pub fn new(frame_id: u32, time: f32, ms_type: i64) -> Self {
        FramesSim {
            frame_id,
            time,
            ms_type,
        }
    }
    pub fn parse_ms_type(&self) -> MsType {
        match self.ms_type {
            0 => MsType::Precursor,
            8 => MsType::FragmentDda,
            9 => MsType::FragmentDia,
            _ => MsType::Unknown,
        }

    }
}

pub struct FragmentIonSim {
    pub peptide_id: u32,
    pub collision_energy: f64,
    pub charge: i8,
    pub fragment_intensities: Vec<FragmentIonSeries>,
}

impl FragmentIonSim {
    pub fn new(
        peptide_id: u32,
        collision_energy: f64,
        charge: i8,
        fragment_intensities: Vec<FragmentIonSeries>,
    ) -> Self {
        FragmentIonSim {
            peptide_id,
            charge,
            collision_energy,
            fragment_intensities,
        }
    }
}