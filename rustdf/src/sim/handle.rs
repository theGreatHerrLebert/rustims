use crate::sim::containers::{
    FragmentIonSim, FrameToWindowGroupSim, FramesSim, IonSim, PeptidesSim, ScansSim,
    SignalDistribution, WindowGroupSettingsSim,
};
use mscore::data::peptide::{FragmentType, PeptideProductIonSeriesCollection, PeptideSequence};
use mscore::data::spectrum::{MsType, MzSpectrum};
use mscore::simulation::annotation::MzSpectrumAnnotated;
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::timstof::quadrupole::{IonTransmission, PASEFMeta, TimsTransmissionDDA, TimsTransmissionDIA};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

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
            Ok(FramesSim::new(row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        let mut frames = Vec::new();
        for frame in frames_iter {
            frames.push(frame?);
        }
        Ok(frames)
    }

    pub fn read_scans(&self) -> rusqlite::Result<Vec<ScansSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM scans")?;
        let scans_iter = stmt.query_map([], |row| Ok(ScansSim::new(row.get(0)?, row.get(1)?)))?;
        let mut scans = Vec::new();
        for scan in scans_iter {
            scans.push(scan?);
        }
        Ok(scans)
    }
    pub fn read_peptides(&self) -> rusqlite::Result<Vec<PeptidesSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM peptides")?;
        let peptides_iter = stmt.query_map([], |row| {
            let frame_occurrence_str: String = row.get(15)?;
            let frame_abundance_str: String = row.get(16)?;

            let frame_occurrence: Vec<u32> = match serde_json::from_str(&frame_occurrence_str) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        15,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            // if the frame abundance is not available, set it to 0
            let frame_abundance: Vec<f32> = match serde_json::from_str(&frame_abundance_str) {
                Ok(value) => value,
                Err(_e) => vec![0.0; frame_occurrence.len()],
            };

            let frame_distribution =
                SignalDistribution::new(0.0, 0.0, 0.0, frame_occurrence, frame_abundance);

            Ok(PeptidesSim {
                protein_id: row.get(0)?,
                peptide_id: row.get(1)?,
                sequence: PeptideSequence::new(row.get(2)?, row.get(1)?),
                proteins: row.get(3)?,
                decoy: row.get(4)?,
                missed_cleavages: row.get(5)?,
                n_term: row.get(6)?,
                c_term: row.get(7)?,
                mono_isotopic_mass: row.get(8)?,
                retention_time: row.get(9)?,
                events: row.get(10)?,
                frame_start: row.get(13)?,
                frame_end: row.get(14)?,
                frame_distribution,
            })
        })?;
        let mut peptides = Vec::new();
        for peptide in peptides_iter {
            peptides.push(peptide?);
        }
        Ok(peptides)
    }

    pub fn read_ions(&self) -> rusqlite::Result<Vec<IonSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM ions")?;
        let ions_iter = stmt.query_map([], |row| {
            let simulated_spectrum_str: String = row.get(8)?;
            let scan_occurrence_str: String = row.get(9)?;
            let scan_abundance_str: String = row.get(10)?;

            let simulated_spectrum: MzSpectrum = match serde_json::from_str(&simulated_spectrum_str)
            {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        8,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            let scan_occurrence: Vec<u32> = match serde_json::from_str(&scan_occurrence_str) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        9,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            let scan_abundance: Vec<f32> = match serde_json::from_str(&scan_abundance_str) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        10,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            Ok(IonSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(5)?,
                row.get(6)?,
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
            Ok(FrameToWindowGroupSim::new(row.get(0)?, row.get(1)?))
        })?;

        let mut frame_to_window_groups: Vec<FrameToWindowGroupSim> = Vec::new();
        for frame_to_window_group in frame_to_window_group_iter {
            frame_to_window_groups.push(frame_to_window_group?);
        }

        Ok(frame_to_window_groups)
    }

    pub fn read_pasef_meta(&self) -> rusqlite::Result<Vec<PASEFMeta>> {
        let mut stmt = self.connection.prepare("SELECT * FROM pasef_meta")?;
        let pasef_meta_iter = stmt.query_map([], |row| {
            Ok(PASEFMeta::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?))
        })?;

        let mut pasef_meta: Vec<PASEFMeta> = Vec::new();

        for pasef_meta_entry in pasef_meta_iter {
            pasef_meta.push(pasef_meta_entry?);
        }

        Ok(pasef_meta)
    }

    /// Read peptides that are present in the given frame range.
    /// A peptide is included if its frame range overlaps with [frame_min, frame_max].
    pub fn read_peptides_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
    ) -> rusqlite::Result<Vec<PeptidesSim>> {
        let mut stmt = self.connection.prepare(
            "SELECT * FROM peptides WHERE frame_occurrence_start <= ?1 AND frame_occurrence_end >= ?2"
        )?;

        let peptides_iter = stmt.query_map([frame_max, frame_min], |row| {
            let frame_occurrence_str: String = row.get(15)?;
            let frame_abundance_str: String = row.get(16)?;

            let frame_occurrence: Vec<u32> = match serde_json::from_str(&frame_occurrence_str) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        15,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            let frame_abundance: Vec<f32> = match serde_json::from_str(&frame_abundance_str) {
                Ok(value) => value,
                Err(_e) => vec![0.0; frame_occurrence.len()],
            };

            let frame_distribution =
                SignalDistribution::new(0.0, 0.0, 0.0, frame_occurrence, frame_abundance);

            Ok(PeptidesSim {
                protein_id: row.get(0)?,
                peptide_id: row.get(1)?,
                sequence: PeptideSequence::new(row.get(2)?, row.get(1)?),
                proteins: row.get(3)?,
                decoy: row.get(4)?,
                missed_cleavages: row.get(5)?,
                n_term: row.get(6)?,
                c_term: row.get(7)?,
                mono_isotopic_mass: row.get(8)?,
                retention_time: row.get(9)?,
                events: row.get(10)?,
                frame_start: row.get(13)?,
                frame_end: row.get(14)?,
                frame_distribution,
            })
        })?;

        let mut peptides = Vec::new();
        for peptide in peptides_iter {
            peptides.push(peptide?);
        }
        Ok(peptides)
    }

    /// Read ions for specific peptide IDs.
    /// Uses batched queries for efficiency with large peptide ID lists.
    pub fn read_ions_for_peptides(&self, peptide_ids: &[u32]) -> rusqlite::Result<Vec<IonSim>> {
        if peptide_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_ions = Vec::new();

        // Process in chunks to avoid SQLite parameter limits
        const CHUNK_SIZE: usize = 500;

        for chunk in peptide_ids.chunks(CHUNK_SIZE) {
            let placeholders: String = chunk.iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(",");

            let sql = format!(
                "SELECT * FROM ions WHERE peptide_id IN ({})",
                placeholders
            );

            let mut stmt = self.connection.prepare(&sql)?;

            let ions_iter = stmt.query_map(
                rusqlite::params_from_iter(chunk.iter()),
                |row| {
                    let simulated_spectrum_str: String = row.get(8)?;
                    let scan_occurrence_str: String = row.get(9)?;
                    let scan_abundance_str: String = row.get(10)?;

                    let simulated_spectrum: MzSpectrum = match serde_json::from_str(&simulated_spectrum_str) {
                        Ok(value) => value,
                        Err(e) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                8,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))
                        }
                    };

                    let scan_occurrence: Vec<u32> = match serde_json::from_str(&scan_occurrence_str) {
                        Ok(value) => value,
                        Err(e) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                9,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))
                        }
                    };

                    let scan_abundance: Vec<f32> = match serde_json::from_str(&scan_abundance_str) {
                        Ok(value) => value,
                        Err(e) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                10,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))
                        }
                    };

                    Ok(IonSim::new(
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(5)?,
                        row.get(6)?,
                        simulated_spectrum,
                        scan_occurrence,
                        scan_abundance,
                    ))
                },
            )?;

            for ion in ions_iter {
                all_ions.push(ion?);
            }
        }

        Ok(all_ions)
    }

    /// Read fragment ions for specific peptide IDs.
    /// Uses batched queries for efficiency with large peptide ID lists.
    pub fn read_fragment_ions_for_peptides(&self, peptide_ids: &[u32]) -> rusqlite::Result<Vec<FragmentIonSim>> {
        if peptide_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_fragment_ions = Vec::new();

        // Process in chunks to avoid SQLite parameter limits
        const CHUNK_SIZE: usize = 500;

        for chunk in peptide_ids.chunks(CHUNK_SIZE) {
            let placeholders: String = chunk.iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(",");

            let sql = format!(
                "SELECT * FROM fragment_ions WHERE peptide_id IN ({})",
                placeholders
            );

            let mut stmt = self.connection.prepare(&sql)?;

            let fragment_ion_iter = stmt.query_map(
                rusqlite::params_from_iter(chunk.iter()),
                |row| {
                    let indices_string: String = row.get(4)?;
                    let values_string: String = row.get(5)?;

                    let indices: Vec<u32> = match serde_json::from_str(&indices_string) {
                        Ok(value) => value,
                        Err(e) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                4,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))
                        }
                    };

                    let values: Vec<f64> = match serde_json::from_str(&values_string) {
                        Ok(value) => value,
                        Err(e) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                5,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))
                        }
                    };

                    Ok(FragmentIonSim::new(
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        indices,
                        values,
                    ))
                },
            )?;

            for fragment_ion in fragment_ion_iter {
                all_fragment_ions.push(fragment_ion?);
            }
        }

        Ok(all_fragment_ions)
    }

    pub fn read_fragment_ions(&self) -> rusqlite::Result<Vec<FragmentIonSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM fragment_ions")?;

        let fragment_ion_sim_iter = stmt.query_map([], |row| {
            let indices_string: String = row.get(4)?;
            let values_string: String = row.get(5)?;

            let indices: Vec<u32> = match serde_json::from_str(&indices_string) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            let values: Vec<f64> = match serde_json::from_str(&values_string) {
                Ok(value) => value,
                Err(e) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                }
            };

            Ok(FragmentIonSim::new(
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                indices,
                values,
            ))
        })?;

        let mut fragment_ion_sim = Vec::new();
        for fragment_ion in fragment_ion_sim_iter {
            fragment_ion_sim.push(fragment_ion?);
        }

        Ok(fragment_ion_sim)
    }

    pub fn get_transmission_dia(&self) -> TimsTransmissionDIA {
        let frame_to_window_group = self.read_frame_to_window_group().unwrap();
        let window_group_settings = self.read_window_group_settings().unwrap();

        TimsTransmissionDIA::new(
            frame_to_window_group
                .iter()
                .map(|x| x.frame_id as i32)
                .collect(),
            frame_to_window_group
                .iter()
                .map(|x| x.window_group as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.window_group as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.scan_start as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.scan_end as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.isolation_mz as f64)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.isolation_width as f64)
                .collect(),
            None,
        )
    }

    pub fn get_transmission_dda(&self) -> TimsTransmissionDDA {
        let pasef_meta = self.read_pasef_meta().unwrap();
        TimsTransmissionDDA::new(
            pasef_meta,
            None,
        )
    }

    pub fn get_collision_energy_dia(&self) -> TimsTofCollisionEnergyDIA {
        let frame_to_window_group = self.read_frame_to_window_group().unwrap();
        let window_group_settings = self.read_window_group_settings().unwrap();

        TimsTofCollisionEnergyDIA::new(
            frame_to_window_group
                .iter()
                .map(|x| x.frame_id as i32)
                .collect(),
            frame_to_window_group
                .iter()
                .map(|x| x.window_group as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.window_group as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.scan_start as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.scan_end as i32)
                .collect(),
            window_group_settings
                .iter()
                .map(|x| x.collision_energy as f64)
                .collect(),
        )
    }

    fn ion_map_fn_dda(
        ion: IonSim,
        peptide_map: &BTreeMap<u32, PeptidesSim>,
        precursor_frames: &HashSet<u32>,
        transmission: &TimsTransmissionDDA,
    ) -> BTreeSet<(u32, u32, String, i8, i32)> {
        let peptide = peptide_map.get(&ion.peptide_id).unwrap();
        let mut ret_tree: BTreeSet<(u32, u32, String, i8, i32)> = BTreeSet::new();

        // go over all frames the ion occurs in
        for frame in peptide.frame_distribution.occurrence.iter() {
            // only consider fragment frames
            if !precursor_frames.contains(frame) {
                // go over all scans the ion occurs in
                for scan in &ion.scan_distribution.occurrence {
                    // check transmission for all precursor ion peaks of the isotopic envelope

                    let precursor_spec = &ion.simulated_spectrum;

                    if transmission.any_transmitted(
                        *frame as i32,
                        *scan as i32,
                        &precursor_spec.mz,
                        Some(0.5),
                    ) {
                        let collision_energy =
                            transmission.get_collision_energy(*frame as i32, *scan as i32).unwrap_or(0.0);

                        let quantized_energy = (collision_energy * 100.0).round() as i32;

                        ret_tree.insert((
                            ion.peptide_id,
                            ion.ion_id,
                            peptide.sequence.sequence.clone(),
                            ion.charge,
                            quantized_energy,
                        ));
                    }
                }
            }
        }
        ret_tree
    }

    fn ion_map_fn_dia(
        ion: IonSim,
        peptide_map: &BTreeMap<u32, PeptidesSim>,
        precursor_frames: &HashSet<u32>,
        transmission: &TimsTransmissionDIA,
        collision_energy: &TimsTofCollisionEnergyDIA,
    ) -> BTreeSet<(u32, u32, String, i8, i32)> {
        let peptide = peptide_map.get(&ion.peptide_id).unwrap();
        let mut ret_tree: BTreeSet<(u32, u32, String, i8, i32)> = BTreeSet::new();

        // go over all frames the ion occurs in
        for frame in peptide.frame_distribution.occurrence.iter() {
            // only consider fragment frames
            if !precursor_frames.contains(frame) {
                // go over all scans the ion occurs in
                for scan in &ion.scan_distribution.occurrence {
                    // check transmission for all precursor ion peaks of the isotopic envelope

                    let precursor_spec = &ion.simulated_spectrum;

                    if transmission.any_transmitted(
                        *frame as i32,
                        *scan as i32,
                        &precursor_spec.mz,
                        Some(0.5),
                    ) {
                        let collision_energy =
                            collision_energy.get_collision_energy(*frame as i32, *scan as i32);
                        let quantized_energy = (collision_energy * 100.0).round() as i32;

                        ret_tree.insert((
                            ion.peptide_id,
                            ion.ion_id,
                            peptide.sequence.sequence.clone(),
                            ion.charge,
                            quantized_energy,
                        ));
                    }
                }
            }
        }
        ret_tree
    }

    // TODO: take isotopic envelope into account
    pub fn get_transmitted_ions(
        &self,
        num_threads: usize,
        dda_mode: bool,
    ) -> (Vec<i32>, Vec<i32>, Vec<String>, Vec<i8>, Vec<f32>) {

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let peptides = self.read_peptides().unwrap();

        let peptide_map = TimsTofSyntheticsDataHandle::build_peptide_map(&peptides);

        let precursor_frames =
            TimsTofSyntheticsDataHandle::build_precursor_frame_id_set(&self.read_frames().unwrap());

        let ions = self.read_ions().unwrap();

        let trees = match dda_mode {
            true => {
                let transmission = self.get_transmission_dda();
                thread_pool.install(|| {
                    ions.par_iter()
                        .map(|ion| {
                            TimsTofSyntheticsDataHandle::ion_map_fn_dda(
                                ion.clone(),
                                &peptide_map,
                                &precursor_frames,
                                &transmission,
                            )
                        })
                        .collect::<Vec<_>>()
            })
        },
            false => {
                let transmission = self.get_transmission_dia();
                let collision_energy = self.get_collision_energy_dia();
                thread_pool.install(|| {
                    ions.par_iter()
                        .map(|ion| {
                            TimsTofSyntheticsDataHandle::ion_map_fn_dia(
                                ion.clone(),
                                &peptide_map,
                                &precursor_frames,
                                &transmission,
                                &collision_energy,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            },
        };

        let mut ret_tree: BTreeSet<(u32, u32, String, i8, i32)> = BTreeSet::new();
        for tree in trees {
            ret_tree.extend(tree);
        }

        let mut ret_peptide_id = Vec::new();
        let mut ret_ion_id = Vec::new();
        let mut ret_sequence = Vec::new();
        let mut ret_charge = Vec::new();
        let mut ret_energy = Vec::new();

        for (peptide_id, ion_id, sequence, charge, energy) in ret_tree {
            ret_peptide_id.push(peptide_id as i32);
            ret_ion_id.push(ion_id as i32);
            ret_sequence.push(sequence);
            ret_charge.push(charge);
            ret_energy.push(energy as f32 / 100.0);
        }

        (
            ret_peptide_id,
            ret_ion_id,
            ret_sequence,
            ret_charge,
            ret_energy,
        )
    }

    /// Lazy version of get_transmitted_ions that only loads data for a specific frame range.
    /// This reduces memory usage by only loading peptides and ions that are relevant to the
    /// specified frame range instead of all data from the database.
    ///
    /// # Arguments
    ///
    /// * `frame_min` - Minimum frame ID to include (inclusive)
    /// * `frame_max` - Maximum frame ID to include (inclusive)
    /// * `num_threads` - Number of threads to use for parallel processing
    /// * `dda_mode` - If true, use DDA transmission; if false, use DIA transmission
    ///
    /// # Returns
    ///
    /// Tuple of (peptide_ids, ion_ids, sequences, charges, collision_energies) for transmitted ions
    pub fn get_transmitted_ions_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
        num_threads: usize,
        dda_mode: bool,
    ) -> (Vec<i32>, Vec<i32>, Vec<String>, Vec<i8>, Vec<f32>) {

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // Only load peptides for the specified frame range
        let peptides = self.read_peptides_for_frame_range(frame_min, frame_max).unwrap();

        if peptides.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        let peptide_ids: Vec<u32> = peptides.iter().map(|p| p.peptide_id).collect();
        let peptide_map = TimsTofSyntheticsDataHandle::build_peptide_map(&peptides);

        let precursor_frames =
            TimsTofSyntheticsDataHandle::build_precursor_frame_id_set(&self.read_frames().unwrap());

        // Only load ions for the peptides in our frame range
        let ions = self.read_ions_for_peptides(&peptide_ids).unwrap();

        if ions.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        let trees = match dda_mode {
            true => {
                let transmission = self.get_transmission_dda();
                thread_pool.install(|| {
                    ions.par_iter()
                        .map(|ion| {
                            TimsTofSyntheticsDataHandle::ion_map_fn_dda(
                                ion.clone(),
                                &peptide_map,
                                &precursor_frames,
                                &transmission,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            },
            false => {
                let transmission = self.get_transmission_dia();
                let collision_energy = self.get_collision_energy_dia();
                thread_pool.install(|| {
                    ions.par_iter()
                        .map(|ion| {
                            TimsTofSyntheticsDataHandle::ion_map_fn_dia(
                                ion.clone(),
                                &peptide_map,
                                &precursor_frames,
                                &transmission,
                                &collision_energy,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            },
        };

        let mut ret_tree: BTreeSet<(u32, u32, String, i8, i32)> = BTreeSet::new();
        for tree in trees {
            ret_tree.extend(tree);
        }

        let mut ret_peptide_id = Vec::new();
        let mut ret_ion_id = Vec::new();
        let mut ret_sequence = Vec::new();
        let mut ret_charge = Vec::new();
        let mut ret_energy = Vec::new();

        for (peptide_id, ion_id, sequence, charge, energy) in ret_tree {
            ret_peptide_id.push(peptide_id as i32);
            ret_ion_id.push(ion_id as i32);
            ret_sequence.push(sequence);
            ret_charge.push(charge);
            ret_energy.push(energy as f32 / 100.0);
        }

        (
            ret_peptide_id,
            ret_ion_id,
            ret_sequence,
            ret_charge,
            ret_energy,
        )
    }

    /// Method to build a map from peptide id to ions
    pub fn build_peptide_to_ion_map(ions: &Vec<IonSim>) -> BTreeMap<u32, Vec<IonSim>> {
        let mut ion_map = BTreeMap::new();
        for ion in ions.iter() {
            let ions = ion_map.entry(ion.peptide_id).or_insert_with(Vec::new);
            ions.push(ion.clone());
        }
        ion_map
    }

    /// Method to build a map from peptide id to events (absolute number of events in the simulation)
    pub fn build_peptide_map(peptides: &Vec<PeptidesSim>) -> BTreeMap<u32, PeptidesSim> {
        let mut peptide_map = BTreeMap::new();
        for peptide in peptides.iter() {
            peptide_map.insert(peptide.peptide_id, peptide.clone());
        }
        peptide_map
    }

    /// Method to build a set of precursor frame ids, can be used to check if a frame is a precursor frame
    pub fn build_precursor_frame_id_set(frames: &Vec<FramesSim>) -> HashSet<u32> {
        frames
            .iter()
            .filter(|frame| frame.parse_ms_type() == MsType::Precursor)
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
    pub fn build_frame_to_abundances(
        peptides: &Vec<PeptidesSim>,
    ) -> BTreeMap<u32, (Vec<u32>, Vec<f32>)> {
        let mut frame_to_abundances = BTreeMap::new();

        for peptide in peptides.iter() {
            let peptide_id = peptide.peptide_id;
            let frame_occurrence = peptide.frame_distribution.occurrence.clone();
            let frame_abundance = peptide.frame_distribution.abundance.clone();

            for (frame_id, abundance) in frame_occurrence.iter().zip(frame_abundance.iter()) {
                // only insert if the abundance is greater than 1e-6

                if *abundance > 1e-6 {
                    let (occurrences, abundances) = frame_to_abundances
                        .entry(*frame_id)
                        .or_insert((vec![], vec![]));
                    occurrences.push(peptide_id);
                    abundances.push(*abundance);
                }
            }
        }

        frame_to_abundances
    }
    pub fn build_peptide_to_ions(
        ions: &Vec<IonSim>,
    ) -> BTreeMap<
        u32,
        (
            Vec<f32>,
            Vec<Vec<u32>>,
            Vec<Vec<f32>>,
            Vec<i8>,
            Vec<MzSpectrum>,
        ),
    > {
        let mut peptide_to_ions = BTreeMap::new();

        for ion in ions.iter() {
            let peptide_id = ion.peptide_id;
            let abundance = ion.relative_abundance;
            let scan_occurrence = ion.scan_distribution.occurrence.clone();
            let scan_abundance = ion.scan_distribution.abundance.clone();
            let charge = ion.charge;
            let spectrum = ion.simulated_spectrum.clone();

            let (abundances, scan_occurrences, scan_abundances, charges, spectra) = peptide_to_ions
                .entry(peptide_id)
                .or_insert((vec![], vec![], vec![], vec![], vec![]));
            abundances.push(abundance);
            scan_occurrences.push(scan_occurrence);
            scan_abundances.push(scan_abundance);
            charges.push(charge);
            spectra.push(spectrum);
        }

        peptide_to_ions
    }
    pub fn build_fragment_ions(
        peptides_sim: &BTreeMap<u32, PeptidesSim>,
        fragment_ions: &Vec<FragmentIonSim>,
        num_threads: usize,
    ) -> BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrum>)> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let fragment_ion_map = thread_pool.install(|| {
            fragment_ions
                .par_iter()
                .map(|fragment_ion| {
                    let key = (
                        fragment_ion.peptide_id,
                        fragment_ion.charge,
                        (fragment_ion.collision_energy * 1e3).round() as i32,
                    );

                    let value = peptides_sim
                        .get(&fragment_ion.peptide_id)
                        .unwrap()
                        .sequence
                        .associate_with_predicted_intensities(
                            fragment_ion.charge as i32,
                            FragmentType::B,
                            fragment_ion.to_dense(174),
                            true,
                            true,
                        );

                    let fragment_ions: Vec<MzSpectrum> = value
                        .peptide_ions
                        .par_iter()
                        .map(|ion_series| {
                            ion_series.generate_isotopic_spectrum(1e-2, 1e-3, 100, 1e-5)
                        })
                        .collect();
                    (key, (value, fragment_ions))
                })
                .collect::<BTreeMap<_, _>>() // Collect the results into a BTreeMap
        });

        fragment_ion_map
    }

    pub fn build_fragment_ions_annotated(
        peptides_sim: &BTreeMap<u32, PeptidesSim>,
        fragment_ions: &Vec<FragmentIonSim>,
        num_threads: usize,
    ) -> BTreeMap<(u32, i8, i32), (PeptideProductIonSeriesCollection, Vec<MzSpectrumAnnotated>)>
    {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let fragment_ion_map = thread_pool.install(|| {
            fragment_ions
                .par_iter()
                .map(|fragment_ion| {
                    let key = (
                        fragment_ion.peptide_id,
                        fragment_ion.charge,
                        (fragment_ion.collision_energy * 1e3).round() as i32,
                    );

                    let value = peptides_sim
                        .get(&fragment_ion.peptide_id)
                        .unwrap()
                        .sequence
                        .associate_with_predicted_intensities(
                            fragment_ion.charge as i32,
                            FragmentType::B,
                            fragment_ion.to_dense(174),
                            true,
                            true,
                        );

                    let fragment_ions: Vec<MzSpectrumAnnotated> = value
                        .peptide_ions
                        .par_iter()
                        .map(|ion_series| {
                            ion_series.generate_isotopic_spectrum_annotated(1e-2, 1e-3, 100, 1e-5)
                        })
                        .collect();
                    (key, (value, fragment_ions))
                })
                .collect::<BTreeMap<_, _>>() // Collect the results into a BTreeMap
        });

        fragment_ion_map
    }

    /// Build fragment ions with complementary isotope distribution data.
    ///
    /// This variant calculates both the fragment isotope distribution and
    /// the complementary fragment isotope distribution, which are needed
    /// for quad-selection dependent isotope transmission calculations.
    ///
    /// # Arguments
    ///
    /// * `peptides_sim` - Map of peptide_id to PeptidesSim
    /// * `fragment_ions` - Vector of FragmentIonSim
    /// * `num_threads` - Number of threads for parallel processing
    ///
    /// # Returns
    ///
    /// * `BTreeMap` mapping (peptide_id, charge, collision_energy) to
    ///   (PeptideProductIonSeriesCollection, fragment spectra, fragment distributions, complementary distributions)
    /// Build fragment ions with transmission data for both precursor scaling and per-fragment modes.
    ///
    /// This function calculates:
    /// - Precursor isotope distribution (for PrecursorScaling mode)
    /// - Per-fragment isotope distributions with their complementary distributions (for PerFragment mode)
    pub fn build_fragment_ions_with_transmission_data(
        peptides_sim: &BTreeMap<u32, PeptidesSim>,
        fragment_ions: &Vec<FragmentIonSim>,
        num_threads: usize,
    ) -> BTreeMap<(u32, i8, i32), FragmentIonsWithComplementary> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let fragment_ion_map = thread_pool.install(|| {
            fragment_ions
                .par_iter()
                .map(|fragment_ion| {
                    let key = (
                        fragment_ion.peptide_id,
                        fragment_ion.charge,
                        (fragment_ion.collision_energy * 1e3).round() as i32,
                    );

                    let peptide_sim = peptides_sim.get(&fragment_ion.peptide_id).unwrap();

                    // Get precursor atomic composition for complementary calculations
                    let precursor_composition = peptide_sim.sequence.atomic_composition();

                    // Calculate precursor isotope distribution for PrecursorScaling mode
                    let precursor_composition_owned: std::collections::HashMap<String, i32> =
                        precursor_composition.iter().map(|(k, v)| (k.to_string(), *v)).collect();
                    let precursor_isotope_distribution = mscore::algorithm::isotope::generate_isotope_distribution(
                        &precursor_composition_owned,
                        1e-3,
                        1e-8,
                        100,
                    ).into_iter().filter(|&(_, abundance)| abundance > 1e-10).collect();

                    let ion_series_collection = peptide_sim
                        .sequence
                        .associate_with_predicted_intensities(
                            fragment_ion.charge as i32,
                            FragmentType::B,
                            fragment_ion.to_dense(174),
                            true,
                            true,
                        );

                    // Calculate fragment spectra and per-fragment transmission data
                    let mut fragment_spectra: Vec<MzSpectrum> = Vec::new();
                    let mut per_fragment_data: Vec<Vec<FragmentIonTransmissionData>> = Vec::new();

                    for ion_series in &ion_series_collection.peptide_ions {
                        // Generate the full isotopic spectrum for this ion series
                        let spectrum = ion_series.generate_isotopic_spectrum(1e-2, 1e-3, 100, 1e-5);
                        fragment_spectra.push(spectrum);

                        // Build per-fragment data for this series
                        let mut series_fragment_data: Vec<FragmentIonTransmissionData> = Vec::new();

                        // Process n-terminal ions (b-ions)
                        for n_ion in &ion_series.n_ions {
                            let frag_dist = n_ion.isotope_distribution(1e-3, 1e-8, 100, 1e-10);
                            let comp_dist = n_ion.complementary_isotope_distribution(
                                &precursor_composition,
                                1e-3,
                                1e-8,
                                100,
                            );

                            series_fragment_data.push(FragmentIonTransmissionData {
                                fragment_distribution: frag_dist,
                                complementary_distribution: comp_dist,
                                predicted_intensity: n_ion.ion.intensity,
                            });
                        }

                        // Process c-terminal ions (y-ions)
                        for c_ion in &ion_series.c_ions {
                            let frag_dist = c_ion.isotope_distribution(1e-3, 1e-8, 100, 1e-10);
                            let comp_dist = c_ion.complementary_isotope_distribution(
                                &precursor_composition,
                                1e-3,
                                1e-8,
                                100,
                            );

                            series_fragment_data.push(FragmentIonTransmissionData {
                                fragment_distribution: frag_dist,
                                complementary_distribution: comp_dist,
                                predicted_intensity: c_ion.ion.intensity,
                            });
                        }

                        per_fragment_data.push(series_fragment_data);
                    }

                    let data = FragmentIonsWithComplementary {
                        ion_series_collection,
                        fragment_spectra,
                        precursor_isotope_distribution,
                        per_fragment_data,
                    };

                    (key, data)
                })
                .collect::<BTreeMap<_, _>>()
        });

        fragment_ion_map
    }
}

/// Data for a single fragment ion with its complementary distribution.
#[derive(Debug, Clone)]
pub struct FragmentIonTransmissionData {
    /// Fragment isotope distribution as (m/z, abundance) pairs
    pub fragment_distribution: Vec<(f64, f64)>,
    /// Complementary fragment isotope distribution as (mass, abundance) pairs
    pub complementary_distribution: Vec<(f64, f64)>,
    /// Predicted intensity of this fragment ion
    pub predicted_intensity: f64,
}

/// Struct holding fragment ion data along with transmission calculation data.
///
/// This is used for quad-selection dependent isotope transmission calculations,
/// where fragment isotope patterns are adjusted based on which precursor isotopes
/// were transmitted through the quadrupole.
#[derive(Debug, Clone)]
pub struct FragmentIonsWithComplementary {
    /// The original ion series collection with intensity predictions
    pub ion_series_collection: PeptideProductIonSeriesCollection,
    /// Pre-calculated fragment spectra (standard isotope patterns)
    pub fragment_spectra: Vec<MzSpectrum>,
    /// Precursor isotope distribution for scaling mode (m/z, abundance)
    pub precursor_isotope_distribution: Vec<(f64, f64)>,
    /// Per-fragment transmission data for per-fragment mode
    /// Outer Vec: one per ion_series, Inner Vec: one per fragment ion in that series
    pub per_fragment_data: Vec<Vec<FragmentIonTransmissionData>>,
}
