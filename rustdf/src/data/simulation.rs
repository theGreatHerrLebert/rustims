use std::collections::BTreeMap;
use mscore::{IndexedMzSpectrum, MsType, MzSpectrum, TimsFrame, TimsSpectrum};
use rusqlite::{Connection, Result};
use std::path::Path;
use serde_json;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

pub struct TimsTofSyntheticsDDA {
    pub ions: Vec<IonsSim>,
    pub peptides: Vec<PeptidesSim>,
    pub scans: Vec<ScansSim>,
    pub frames: Vec<FramesSim>,
    pub precursor_frames: Vec<FramesSim>,
    pub frame_to_abundances: BTreeMap<i64, (Vec<i64>, Vec<f64>)>,
    pub peptide_to_ions: BTreeMap<i64, (Vec<f64>, Vec<Vec<i64>>, Vec<Vec<f64>>, Vec<MzSpectrum>)>,
    pub frame_to_rt: BTreeMap<i64, f64>,
    pub scan_to_mobility: BTreeMap<i64, f64>,
    pub peptide_to_events: BTreeMap<i64, f64>,
}

impl TimsTofSyntheticsDDA {
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
        frames
            .iter()
            .filter(|frame| frame.parse_ms_type() == MsType::Precursor)
            .cloned()
            .collect()
    }

    pub fn build_peptide_to_events(peptides: Vec<PeptidesSim>) -> BTreeMap<i64, f64> {
        let mut peptide_to_events = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_to_events.insert(peptide.peptide_id, peptide.events);
        }
        peptide_to_events
    }

    pub fn build_frame_to_rt(frames: Vec<FramesSim>) -> BTreeMap<i64, f64> {
        let mut frame_to_rt = BTreeMap::new();
        for frame in frames.iter() {
            frame_to_rt.insert(frame.frame_id, frame.time);
        }
        frame_to_rt
    }

    pub fn build_scan_to_mobility(scans: Vec<ScansSim>) -> BTreeMap<i64, f64> {
        let mut scan_to_mobility = BTreeMap::new();
        for scan in scans.iter() {
            scan_to_mobility.insert(scan.scan, scan.mobility);
        }
        scan_to_mobility
    }

    pub fn build_frame(&self, frame_id: i64) -> TimsFrame {

        let mut tims_spectra: Vec<TimsSpectrum> = Vec::new();

        let (peptide_ids, abundances) = self.frame_to_abundances.get(&frame_id).unwrap();
        for (peptide_id, abundance) in peptide_ids.iter().zip(abundances.iter()) {
            let (ion_abundances, scan_occurrences, scan_abundances, spectra) = self.peptide_to_ions.get(&peptide_id).unwrap();

            for (index, ion_abundance) in ion_abundances.iter().enumerate() {
                let scan_occurrence = scan_occurrences.get(index).unwrap();
                let scan_abundance = scan_abundances.get(index).unwrap();
                let spectrum = spectra.get(index).unwrap();

                for (scan, scan_abu) in scan_occurrence.iter().zip(scan_abundance.iter()) {
                    let abundance_factor = abundance * ion_abundance * scan_abu * self.peptide_to_events.get(&peptide_id).unwrap();
                    let scaled_spec: MzSpectrum = spectrum.clone() * abundance_factor;
                    let index = vec![0; scaled_spec.mz.len()];
                    let tims_spec = TimsSpectrum::new(
                        frame_id as i32,
                        *scan as i32,
                        *self.frame_to_rt.get(&frame_id).unwrap(),
                        *self.scan_to_mobility.get(&(*&scan - 1)).unwrap(),
                        MsType::Precursor,
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
            15.0,
            1e9,
        )
    }

    // Method to build multiple frames in parallel
    pub fn build_frames(&self, frame_ids: Vec<i64>, num_threads: usize) -> Vec<TimsFrame> {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let mut tims_frames: Vec<TimsFrame> = Vec::new();

        thread_pool.install(|| {
            tims_frames = frame_ids.par_iter().map(|frame_id| self.build_frame(*frame_id)).collect();
        });

        tims_frames
    }

    pub fn build_frame_to_abundances(peptides: Vec<PeptidesSim>) -> BTreeMap<i64, (Vec<i64>, Vec<f64>)> {
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
    pub fn build_peptide_to_ions(ions: Vec<IonsSim>) -> BTreeMap<i64, (Vec<f64>, Vec<Vec<i64>>, Vec<Vec<f64>>, Vec<MzSpectrum>)> {
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
pub struct IonsSim {
    pub peptide_id: i64,
    pub mz: f64,
    pub charge: i64,
    pub relative_abundance: f64,
    pub mobility: f64,
    pub simulated_spectrum: MzSpectrum,
    pub scan_occurrence: Vec<i64>,
    pub scan_abundance: Vec<f64>,
}

impl IonsSim {
    pub fn new(
        peptide_id: i64,
        mz: f64,
        charge: i64,
        relative_abundance: f64,
        mobility: f64,
        simulated_spectrum: MzSpectrum,
        scan_occurrence: Vec<i64>,
        scan_abundance: Vec<f64>,
    ) -> Self {
        IonsSim {
            peptide_id,
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
    pub peptide_id: i64,
    pub sequence: String,
    pub proteins: String,
    pub decoy: bool,
    pub missed_cleavages: i64,
    pub n_term : Option<bool>,
    pub c_term : Option<bool>,
    pub mono_isotopic_mass: f64,
    pub retention_time: f64,
    pub events: f64,
    pub frame_occurrence: Vec<i64>,
    pub frame_abundance: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ScansSim {
    pub scan: i64,
    pub mobility: f64,
}

impl ScansSim {
    pub fn new(scan: i64, mobility: f64) -> Self {
        ScansSim { scan, mobility }
    }
}

#[derive(Debug, Clone)]
pub struct FramesSim {
    pub frame_id: i64,
    pub time: f64,
    pub ms_type: i64,
}

impl FramesSim {
    pub fn new(frame_id: i64, time: f64, ms_type: i64) -> Self {
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

            let frame_occurrence: Vec<i64> = match serde_json::from_str(&frame_occurrence_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    10,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let frame_abundance: Vec<f64> = match serde_json::from_str(&frame_abundance_str) {
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
            let simulated_spectrum_str: String = row.get(5)?;
            let scan_occurrence_str: String = row.get(6)?;
            let scan_abundance_str: String = row.get(7)?;

            let simulated_spectrum: MzSpectrum = match serde_json::from_str(&simulated_spectrum_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let scan_occurrence: Vec<i64> = match serde_json::from_str(&scan_occurrence_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    6,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            let scan_abundance: Vec<f64> = match serde_json::from_str(&scan_abundance_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    7,
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
}