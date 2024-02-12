use std::collections::{BTreeMap, HashSet};
use mscore::{IndexedMzSpectrum, IonTransmission, MsType, MzSpectrum, TimsFrame, TimsSpectrum, TimsTransmissionDIA};
use rusqlite::{Connection, Result};
use std::path::Path;
use serde_json;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::sim::containers::{FramesSim, FrameToWindowGroupSim, IonsSim, PeptidesSim, ScansSim, WindowGroupSettingsSim, FragmentIonSeriesSim};

pub struct TimsTofSyntheticsDIA {
    pub synthetics: TimsTofSynthetics,
    pub transmission_settings: TimsTransmissionDIA,
}

impl TimsTofSyntheticsDIA {
    pub fn new(path: &Path) -> Result<Self> {
        let synthetics = TimsTofSynthetics::new(path)?;
        let frame_to_window_group = SyntheticsDataHandle::new(path)?.read_frame_to_window_group()?;
        let window_group_settings = SyntheticsDataHandle::new(path)?.read_window_group_settings()?;
        let transmission_settings = TimsTransmissionDIA::new(
            frame_to_window_group.iter().map(|x| x.frame_id as i32).collect(),
            frame_to_window_group.iter().map(|x| x.window_group as i32).collect(),
            window_group_settings.iter().map(|x| x.window_group as i32).collect(),
            window_group_settings.iter().map(|x| x.scan_start as i32).collect(),
            window_group_settings.iter().map(|x| x.scan_end as i32).collect(),
            window_group_settings.iter().map(|x| x.isolation_mz as f64).collect(),
            window_group_settings.iter().map(|x| x.isolation_width as f64).collect(),
            None,
        );
        Ok(Self {
            synthetics,
            transmission_settings,
        })
    }

    pub fn build_frame(&self, frame_id: u32, fragmentation: bool) -> TimsFrame {
        match self.synthetics.precursor_frame_id_set.contains(&frame_id) {
            true => self.build_ms1_frame(frame_id),
            false => self.build_ms2_frame(frame_id, fragmentation),
        }
    }

    pub fn build_frames(&self, frame_ids: Vec<u32>, fragmentation: bool, num_threads: usize) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids.par_iter().map(|frame_id| self.build_frame(*frame_id, fragmentation)).collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
    }

    fn build_ms1_frame(&self, frame_id: u32) -> TimsFrame {
        let tims_frame = self.synthetics.build_precursor_frame(frame_id);
        tims_frame
    }
    fn build_ms2_frame(&self, frame_id: u32, fragmentation: bool) -> TimsFrame {
        match fragmentation {
            false => {
                let mut frame = self.transmission_settings.transmit_tims_frame(&self.build_ms1_frame(frame_id), None);
                frame.ms_type = MsType::FragmentDia;
                frame
            },
            true => self.build_fragment_frame(frame_id),
        }
    }
    fn build_fragment_frame(&self, _frame_id: u32) -> TimsFrame {
        todo!("implement the method to build a fragment frame")
    }
}

pub struct TimsTofSynthetics {
    pub ions: Vec<IonsSim>,
    pub peptides: Vec<PeptidesSim>,
    pub scans: Vec<ScansSim>,
    pub frames: Vec<FramesSim>,
    pub precursor_frame_id_set: HashSet<u32>,
    pub frame_to_abundances: BTreeMap<u32, (Vec<u32>, Vec<f32>)>,
    pub peptide_to_ions: BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<MzSpectrum>)>,
    pub frame_to_rt: BTreeMap<u32, f32>,
    pub scan_to_mobility: BTreeMap<u32, f32>,
    pub peptide_to_events: BTreeMap<u32, f32>,
}

impl TimsTofSynthetics {
    /// Create a new instance of TimsTofSynthetics
    ///
    /// # Arguments
    ///
    /// * `path` - A reference to a Path
    ///
    /// # Returns
    ///
    /// * A Result containing the TimsTofSynthetics instance
    ///
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
            precursor_frame_id_set: Self::build_precursor_frame_id_set(frames.clone()),
            frame_to_abundances: Self::build_frame_to_abundances(peptides.clone()),
            peptide_to_ions: Self::build_peptide_to_ions(ions.clone()),
            frame_to_rt: Self::build_frame_to_rt(frames.clone()),
            scan_to_mobility: Self::build_scan_to_mobility(scans.clone()),
            peptide_to_events: Self::build_peptide_to_events(peptides.clone()),
        })
    }

    // Method to build a set of precursor frame ids, can be used to check if a frame is a precursor frame
    fn build_precursor_frame_id_set(frames: Vec<FramesSim>) -> HashSet<u32> {
        frames.iter().filter(|frame| frame.parse_ms_type() == MsType::Precursor)
            .map(|frame| frame.frame_id)
            .collect()
    }

    // Method to build a map from peptide id to events (absolute number of events in the simulation)
     fn build_peptide_to_events(peptides: Vec<PeptidesSim>) -> BTreeMap<u32, f32> {
        let mut peptide_to_events = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_to_events.insert(peptide.peptide_id, peptide.events);
        }
        peptide_to_events
    }

    // Method to build a map from frame id to retention time
    fn build_frame_to_rt(frames: Vec<FramesSim>) -> BTreeMap<u32, f32> {
        let mut frame_to_rt = BTreeMap::new();
        for frame in frames.iter() {
            frame_to_rt.insert(frame.frame_id, frame.time);
        }
        frame_to_rt
    }

    // Method to build a map from scan id to mobility
    fn build_scan_to_mobility(scans: Vec<ScansSim>) -> BTreeMap<u32, f32> {
        let mut scan_to_mobility = BTreeMap::new();
        for scan in scans.iter() {
            scan_to_mobility.insert(scan.scan, scan.mobility);
        }
        scan_to_mobility
    }
    fn build_frame_to_abundances(peptides: Vec<PeptidesSim>) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
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
    fn build_peptide_to_ions(ions: Vec<IonsSim>) -> BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<MzSpectrum>)> {
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
    pub fn build_precursor_frame(&self, frame_id: u32) -> TimsFrame {

        let ms_type = match self.precursor_frame_id_set.contains(&frame_id) {
            true => MsType::Precursor,
            false => MsType::Unknown,
        };

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        // TODO: make sure that the frame_id is in the frame_to_abundances map
        if !self.frame_to_abundances.contains_key(&frame_id) {
            return TimsFrame::new(
                frame_id as i32,
                ms_type.clone(),
                *self.frame_to_rt.get(&frame_id).unwrap() as f64,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
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
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
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
    pub fn build_precursor_frames(&self, frame_ids: Vec<u32>, num_threads: usize) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids.par_iter().map(|frame_id| self.build_precursor_frame(*frame_id)).collect();
        });

        tims_frames.sort_by(|a, b| a.frame_id.cmp(&b.frame_id));

        tims_frames
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
            let fragment_ion_list_str: String = row.get(12)?;

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

            let fragment_ion_sim: Vec<FragmentIonSeriesSim> = match serde_json::from_str(&fragment_ion_list_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    12,
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
                fragments: fragment_ion_sim,
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
        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_info")?;
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