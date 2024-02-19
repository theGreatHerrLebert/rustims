use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use mscore::algorithm::fragmentation::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::algorithm::quadrupole::{IonTransmission, TimsTransmissionDIA};
use mscore::data::mz_spectrum::{MsType, MzSpectrum};
use rusqlite::Connection;
use crate::sim::containers::{FragmentIonSeries, FragmentIonSim, FramesSim, FrameToWindowGroupSim, IonsSim, PeptidesSim, ScansSim, WindowGroupSettingsSim};

#[derive(Debug)]
pub struct TimsTofSyntheticsDataHandle {
    pub connection: Connection,
}

impl TimsTofSyntheticsDataHandle {
    pub fn new(path: &Path) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;
        Ok(Self { connection })
    }

    pub fn read_frames(&self) -> rusqlite::Result<Vec<FramesSim>> {
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

    pub fn read_scans(&self) -> rusqlite::Result<Vec<ScansSim>> {
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
    pub fn read_peptides(&self) -> rusqlite::Result<Vec<PeptidesSim>> {
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

    pub fn read_ions(&self) -> rusqlite::Result<Vec<IonsSim>> {
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

    pub fn read_window_group_settings(&self) -> rusqlite::Result<Vec<WindowGroupSettingsSim>> {
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

    pub fn read_frame_to_window_group(&self) -> rusqlite::Result<Vec<FrameToWindowGroupSim>> {
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

    pub fn read_fragment_ions(&self) -> rusqlite::Result<Vec<FragmentIonSim>> {

        let mut stmt = self.connection.prepare("SELECT * FROM fragment_ions")?;

        let fragment_ion_sim_iter = stmt.query_map([], |row| {
            // get json string from database
            let fragment_ion_list_str: String = row.get(3)?;

            // convert json string to FragmentIonSeries
            let fragment_ion_sim: Vec<FragmentIonSeries> = match serde_json::from_str(&fragment_ion_list_str) {
                Ok(value) => value,
                Err(e) => return Err(rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )),
            };

            Ok(FragmentIonSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                fragment_ion_sim,
            ))
        })?;

        let mut fragment_ion_sim = Vec::new();
        for fragment_ion in fragment_ion_sim_iter {
            fragment_ion_sim.push(fragment_ion?);
        }

        Ok(fragment_ion_sim)
    }

    pub fn build_peptide_to_ion_map(ions: &Vec<IonsSim>) -> BTreeMap<u32, Vec<IonsSim>> {
        let mut ion_map = BTreeMap::new();
        for ion in ions.iter() {
            let ions = ion_map.entry(ion.peptide_id).or_insert_with(Vec::new);
            ions.push(ion.clone());
        }
        ion_map
    }

    pub fn build_peptide_map(peptides: &Vec<PeptidesSim>) -> BTreeMap<u32, PeptidesSim> {
        let mut peptide_map = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_map.insert(peptide.peptide_id, peptide.clone());
        }
        peptide_map
    }


    // Method to build a set of precursor frame ids, can be used to check if a frame is a precursor frame
    pub fn build_precursor_frame_id_set(frames: &Vec<FramesSim>) -> HashSet<u32> {
        frames.iter().filter(|frame| frame.parse_ms_type() == MsType::Precursor)
            .map(|frame| frame.frame_id)
            .collect()
    }

    // Method to build a map from peptide id to events (absolute number of events in the simulation)
    pub fn build_peptide_to_events(peptides: &Vec<PeptidesSim>) -> BTreeMap<u32, f32> {
        let mut peptide_to_events = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_to_events.insert(peptide.peptide_id, peptide.events);
        }
        peptide_to_events
    }

    // Method to build a map from frame id to retention time
    pub fn build_frame_to_rt(frames: &Vec<FramesSim>) -> BTreeMap<u32, f32> {
        let mut frame_to_rt = BTreeMap::new();
        for frame in frames.iter() {
            frame_to_rt.insert(frame.frame_id, frame.time);
        }
        frame_to_rt
    }

    // Method to build a map from scan id to mobility
    pub fn build_scan_to_mobility(scans: &Vec<ScansSim>) -> BTreeMap<u32, f32> {
        let mut scan_to_mobility = BTreeMap::new();
        for scan in scans.iter() {
            scan_to_mobility.insert(scan.scan, scan.mobility);
        }
        scan_to_mobility
    }
    pub fn build_frame_to_abundances(peptides: &Vec<PeptidesSim>) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
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
    pub fn build_peptide_to_ions(ions: &Vec<IonsSim>) -> BTreeMap<u32, (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<i8>, Vec<MzSpectrum>)> {
        let mut peptide_to_ions = BTreeMap::new();

        for ion in ions.iter() {
            let peptide_id = ion.peptide_id;
            let abundance = ion.relative_abundance;
            let scan_occurrence = ion.scan_occurrence.clone();
            let scan_abundance = ion.scan_abundance.clone();
            let charge = ion.charge;
            let spectrum = ion.simulated_spectrum.clone();

            let (abundances, scan_occurrences, scan_abundances, charges, spectra) = peptide_to_ions.entry(peptide_id).or_insert((vec![], vec![], vec![], vec![], vec![]));
            abundances.push(abundance);
            scan_occurrences.push(scan_occurrence);
            scan_abundances.push(scan_abundance);
            charges.push(charge);
            spectra.push(spectrum);
        }

        peptide_to_ions
    }

    pub fn build_fragment_ions(fragment_ions: &Vec<FragmentIonSim>) -> BTreeMap<(u32, i8, i8), Vec<FragmentIonSeries>> {
        let mut fragment_ion_map: BTreeMap<(u32, i8, i8), Vec<FragmentIonSeries>> = BTreeMap::new();
        for fragment_ion in fragment_ions.iter() {
            let key = (fragment_ion.peptide_id, fragment_ion.charge, (fragment_ion.collision_energy * 1e3).round() as i8);
            let value = fragment_ion.fragment_intensities.clone();
            fragment_ion_map.entry(key).or_insert_with(Vec::new).extend(value);
        }
        fragment_ion_map
    }

    pub fn build_dia_transmitted_fragment_ions_with_collision_energies(&self) -> (Vec<u32>, Vec<String>, Vec<i8>, Vec<f64>) {

        let ions = self.read_ions().unwrap();
        let ion_map = TimsTofSyntheticsDataHandle::build_peptide_to_ion_map(&ions);
        let peptides = self.read_peptides().unwrap();

        // quadrupole and collision energy
        let frame_to_window_group = self.read_frame_to_window_group().unwrap();
        let window_group_settings = self.read_window_group_settings().unwrap();

        // TODO: avoid code duplication with DIA handle
        // get collision energy settings per window group
        let fragmentation_settings = TimsTofCollisionEnergyDIA::new(
            frame_to_window_group.iter().map(|x| x.frame_id as i32).collect(),
            frame_to_window_group.iter().map(|x| x.window_group as i32).collect(),
            window_group_settings.iter().map(|x| x.window_group as i32).collect(),
            window_group_settings.iter().map(|x| x.scan_start as i32).collect(),
            window_group_settings.iter().map(|x| x.scan_end as i32).collect(),
            window_group_settings.iter().map(|x| x.collision_energy as f64).collect(),
        );

        // get ion transmission settings per window group
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

        let mut peptide_ids: Vec<i32> = Vec::new();
        let mut sequences: Vec<String> = Vec::new();
        let mut charges: Vec<i8> = Vec::new();
        let mut collision_energies: Vec<f32> = Vec::new();

        /*
        // go over all peptides
        for peptide in peptides.iter() {

            // get frame occurrence and abundance for peptide
            let frame_occurrence = &peptide.frame_occurrence;

            // get peptide id, key for ion_map
            let peptide_id = peptide.peptide_id;

            // get all ions for peptide
            let ion_list = ion_map.get(&peptide_id).unwrap();

            // go over all ions for a given peptide
            for ion in ion_list.iter() {
                // TODO: calculate full transmission for precursor ions and deal with isotopes
                let mz_mono = &ion.mz;
                let scan_occurrences = &ion.scan_occurrence;

                for frame in frame_occurrence.iter() {
                    for scan in scan_occurrences.iter() {
                        if transmission_settings.is_transmitted(*frame as i32, *scan as i32, *mz_mono as f64, None) {
                            let collision_energy = fragmentation_settings.get_collision_energy(*frame as i32, *scan as i32);
                            peptide_ids.push(peptide_id);
                            sequences.push(peptide.sequence.clone());
                            charges.push(ion.charge);
                            collision_energies.push(collision_energy);
                        }
                    }
                }
            }
        }

        (peptide_ids, sequences, charges, collision_energies)
         */

        (vec![], vec![], vec![], vec![])
    }
}