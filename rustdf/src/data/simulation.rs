use std::collections::BTreeMap;
use mscore::{IndexedMzSpectrum, MsType, MzSpectrum, TimsFrame, TimsSpectrum};
use rusqlite::{Connection, Result};
use std::path::Path;
use serde_json;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;


pub struct TimsTofSynthetics {
    pub ions: Vec<IonsSim>,
    pub peptides: Vec<PeptidesSim>,
    pub scans: Vec<ScansSim>,
    pub frames: Vec<FramesSim>,
    pub precursor_frames: Vec<FramesSim>,
    pub frame_to_abundances: BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
    pub peptide_to_ions: BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<MzSpectrum>)>,
    pub frame_to_rt: BTreeMap<u32, f32>,
    pub scan_to_mobility: BTreeMap<u32, f32>,
    pub peptide_to_events: BTreeMap<u32, f32>,
}

impl TimsTofSynthetics {
    pub fn new(path: &Path) -> Result<Self> {
        let handle = SyntheticsDataHandle::new(path)?;
        let ions = handle.read_ions()?;
        let peptides = handle.read_peptides()?;
        let scans = handle.read_scans()?;
        let frames = handle.read_frames()?;
        Ok(Self {
            ions: ions.clone(),
            peptides: peptides.clone(),
            scans: scans.clone(),
            frames: frames.clone(),
            precursor_frames: Self::build_precursor_frames(frames.clone()),
            frame_to_abundances: Self::build_frame_to_abundances(peptides.clone()),
            peptide_to_ions: Self::build_peptide_to_ions(ions.clone()),
            frame_to_rt: Self::build_frame_to_rt(frames.clone()),
            scan_to_mobility: Self::build_scan_to_mobility(scans.clone()),
            peptide_to_events: Self::build_peptide_to_events(peptides.clone()),
        })
    }
    pub fn build_precursor_frames(frames: Vec<FramesSim>) -> Vec<FramesSim> {
        frames.iter().filter(|frame| frame.parse_ms_type() == MsType::Precursor)
            .cloned()
            .collect()
    }

    pub fn build_peptide_to_events(peptides: Vec<PeptidesSim>) -> BTreeMap<u32, f32> {
        let mut peptide_to_events = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_to_events.insert(peptide.peptide_id, peptide.events);
        }
        peptide_to_events
    }

    pub fn build_frame_to_rt(frames: Vec<FramesSim>) -> BTreeMap<u32, f32> {
        let mut frame_to_rt = BTreeMap::new();
        for frame in frames.iter() {
            frame_to_rt.insert(frame.frame_id, frame.time);
        }
        frame_to_rt
    }

    pub fn build_scan_to_mobility(scans: Vec<ScansSim>) -> BTreeMap<u32, f32> {
        let mut scan_to_mobility = BTreeMap::new();
        for scan in scans.iter() {
            scan_to_mobility.insert(scan.scan, scan.mobility);
        }
        scan_to_mobility
    }

    pub fn build_frame(&self, frame_id: u32) -> TimsFrame {
        // TODO: This is a temporary hack to get the ms_type, need to make this faster and more robust
        let ms_type = self.frames.iter().find(|frame| frame.frame_id == frame_id).unwrap().parse_ms_type();

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        // TODO: make sure that the frame_id is in the frame_to_abundances map
        if !self.frame_to_abundances.contains_key(&frame_id) {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                vec![100],
                vec![1.0],
                vec![0],
                vec![1000.0],
                vec![1.0],
            );
        }


        let (peptide_ids, abundances) = self.frame_to_abundances.get(&frame_id).unwrap();
        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {

            // check if the peptide_id is in the peptide_to_ions map
            if !self.peptide_to_ions.contains_key(&peptide_id) {
                return TimsFrame::new(
                    frame_id as i32,
                    ms_type.clone(),
                    *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                    vec![100],
                    vec![1.0],
                    vec![0],
                    vec![1000.0],
                    vec![1.0],
                );
            }

            // jump to next peptide if the peptide_id is not in the peptide_to_ions map
            if !self.peptide_to_ions.contains_key(&peptide_id) {
                continue;
            }

            let (ion_abundances, scan_occurrences, scan_abundances, spectra) = self.peptide_to_ions.get(&peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = scan_occurrences.get(index).unwrap();
                let scan_abundance = scan_abundances.get(index).unwrap();
                let spectrum = spectra.get(index).unwrap();

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * self.peptide_to_events.get(&peptide_id).unwrap();
                    let scan_id = *scan;
                    let scaled_spec: MzSpectrum = spectrum.clone() * abundance_factor as f64;
                    let index = vec![0; scaled_spec.mz.len()];
                    let tims_spec = TimsSpectrum::new(
                        frame_id as i32,
                        *scan as i32,
                        *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                        *self.scan_to_mobility.get(&scan_id).unwrap() as f64,
                        ms_type.clone(),
                        IndexedMzSpectrum::new(index, scaled_spec.mz, scaled_spec.intensity),
                    );
                    tims_spectra.push(tims_spec);
                }
            }
        }

        let tims_frame = TimsFrame::from_tims_spectra(tims_spectra);

        tims_frame.filter_ranged(
            0.0,
            10000.0,
            0,
            2000,
            0.0,
            10.0,
            1.0,
            1e9,
        )
    }

    // Method to build multiple frames in parallel
    pub fn build_frames(&self, frame_ids: Vec<u32>, num_threads: usize) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids.par_iter().map(|frame_id| self.build_frame(*frame_id)).collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }

    pub fn build_frame_to_abundances(peptides: Vec<PeptidesSim>) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
        let mut frame_to_abundances = BTreeMap::new();

        for peptide in peptides.iter() {
            let peptide_id = peptide.peptide_id;
            let frame_occurrence = peptide.frame_occurrence.clone();
            let frame_abundance = peptide.frame_abundance.clone();

            for (frame_id, abundance) in frame_occurrence.iter().zip(frame_abundance.iter()) {
                let (occurrences, abundances) = frame_to_abundances.entry(*frame_id).or_insert((vec![], vec![]));
                occurrences.push(peptide_id);
                abundances.push(*abundance);
            }
        }

        frame_to_abundances
    }
    pub fn build_peptide_to_ions(ions: Vec<IonsSim>) -> BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<MzSpectrum>)> {
        let mut peptide_to_ions = BTreeMap::new();

        for ion in ions.iter() {
            let peptide_id = ion.peptide_id;
            let abundance = ion.relative_abundance;
            let scan_occurrence = ion.scan_occurrence.clone();
            let scan_abundance = ion.scan_abundance.clone();
            let spectrum = ion.simulated_spectrum.clone();

            let (abundances, scan_occurrences, scan_abundances, spectra) = peptide_to_ions.entry(peptide_id).or_insert((vec![], vec![], vec![], vec![]));
            abundances.push(abundance);
            scan_occurrences.push(scan_occurrence);
            scan_abundances.push(scan_abundance);
            spectra.push(spectrum);
        }

        peptide_to_ions
    }
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

#[derive(Debug)]
pub struct SyntheticsDataHandle {
    pub connection: Connection,
}

impl SyntheticsDataHandle {
    pub fn new(path: &Path) -> Result<Self> {
        let connection = Connection::open(path)?;
        Ok(Self { connection })
    }

    pub fn read_frames(&self) -> Result<Vec<FramesSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM frames")?;
        let frames_iter = stmt.query_map([], |row| {
            Ok(FramesSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
            ))
        })?;
        let mut frames = Vec::new();
        for frame in frames_iter {
            frames.push(frame?);
        }
        Ok(frames)
    }

    pub fn read_scans(&self) -> Result<Vec<ScansSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM scans")?;
        let scans_iter = stmt.query_map([], |row| {
            Ok(ScansSim::new(
                row.get(0)?,
                row.get(1)?,
            ))
        })?;
        let mut scans = Vec::new();
        for scan in scans_iter {
            scans.push(scan?);
        }
        Ok(scans)
    }
    pub fn read_peptides(&self) -> Result<Vec<PeptidesSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM peptides")?;
        let peptides_iter = stmt.query_map([], |row| {
            let frame_occurrence_str: String = row.get(10)?;
            let frame_abundance_str: String = row.get(11)?;

            let frame_occurrence: Vec<u32> = match serde_json::from_str(&frame_occurrence_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    10,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let frame_abundance: Vec<f32> = match serde_json::from_str(&frame_abundance_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    11,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            Ok(PeptidesSim {
                peptide_id: row.get(0)?,
                sequence: row.get(1)?,
                proteins: row.get(2)?,
                decoy: row.get(3)?,
                missed_cleavages: row.get(4)?,
                n_term: row.get(5)?,
                c_term: row.get(6)?,
                mono_isotopic_mass: row.get(7)?,
                retention_time: row.get(8)?,
                events: row.get(9)?,
                frame_occurrence,
                frame_abundance,
            })
        })?;
        let mut peptides = Vec::new();
        for peptide in peptides_iter {
            peptides.push(peptide?);
        }
        Ok(peptides)
    }

    pub fn read_ions(&self) -> Result<Vec<IonsSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM ions")?;
        let ions_iter = stmt.query_map([], |row| {
            let simulated_spectrum_str: String = row.get(6)?;
            let scan_occurrence_str: String = row.get(7)?;
            let scan_abundance_str: String = row.get(8)?;

            let simulated_spectrum: MzSpectrum = match serde_json::from_str(&simulated_spectrum_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    6,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let scan_occurrence: Vec<u32> = match serde_json::from_str(&scan_occurrence_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let scan_abundance: Vec<f32> = match serde_json::from_str(&scan_abundance_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    8,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            Ok(IonsSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                simulated_spectrum,
                scan_occurrence,
                scan_abundance,
            ))
        })?;
        let mut ions = Vec::new();
        for ion in ions_iter {
            ions.push(ion?);
        }
        Ok(ions)
    }

    pub fn read_window_group_settings(&self) -> Result<Vec<WindowGroupSettingsSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
        let window_group_settings_iter = stmt.query_map([], |row| {
            Ok(WindowGroupSettingsSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
            ))
        })?;
        let mut window_group_settings = Vec::new();
        for window_group_setting in window_group_settings_iter {
            window_group_settings.push(window_group_setting?);
        }
        Ok(window_group_settings)
    }

    pub fn read_frame_to_window_group(&self) -> Result<Vec<FrameToWindowGroupSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM frame_to_window_group")?;
        let frame_to_window_group_iter = stmt.query_map([], |row| {
            Ok(FrameToWindowGroupSim::new(
                row.get(0)?,
                row.get(1)?,
            ))
        })?;

        let mut frame_to_window_groups: Vec<FrameToWindowGroupSim> = Vec::new();
        for frame_to_window_group in frame_to_window_group_iter {
            frame_to_window_groups.push(frame_to_window_group?);
        }

        Ok(frame_to_window_groups)
    }
}