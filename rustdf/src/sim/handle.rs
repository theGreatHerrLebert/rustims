use crate::sim::containers::{
    FragmentIonSim, FrameToWindowGroupSim, FramesSim, IonScalar, IonSim, MobilityEnv,
    PeptideScalar, PeptidesSim, ScansSim, SignalDistribution, WindowGroupSettingsSim,
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

/// Resolve the fragment-map collision-energy key for an applied CE (eV),
/// tolerant to ~0.1 eV quantization noise.
///
/// The map is keyed `round(stored_ce * 1e3)`, where `stored_ce` is the CE the
/// predictor saw — quantized to 2 decimals (`ion_map_fn_*`: `round(ce*100)`) and
/// persisted normalized as **f32**. The renderer only has the *applied* CE
/// (DDA: full-precision `ce_bias + ce_slope*mobility`; DIA: the window CE). At
/// ~0.1 eV key boundaries the stored f32 lands on the opposite side of the
/// rounding boundary from any value reconstructed from the applied CE, so an
/// exact key cannot be reproduced — the renderer used to silently drop those
/// fragment series (DDA; DIA window CE is pre-rounded so it was unaffected).
///
/// Probe the natural key and its ±1 neighbours (±0.1 eV = the quantization-noise
/// magnitude) and return the first present. A genuinely different CE (≫0.1 eV
/// away — e.g. a prediction set built for another instrument) still resolves to
/// `None`, which callers treat as a real miss. Returns `None` when no fragments
/// exist near this CE for `(peptide, charge)`.
pub fn resolve_fragment_ce_key<V>(
    map: &BTreeMap<(u32, i8, i32), V>,
    peptide_id: u32,
    charge: i8,
    applied_ce_ev: f64,
) -> Option<i32> {
    let base = (applied_ce_ev * 1e1).round() as i32;
    [base, base - 1, base + 1]
        .into_iter()
        .find(|&k| map.contains_key(&(peptide_id, charge, k)))
}

/// Whether ANY collision-energy entry exists for `(peptide, charge)`. Used at
/// render time to distinguish a legitimate "this precursor has no predicted
/// fragments" skip (prefix absent) from a real "fragments exist but none near
/// the applied CE" mismatch (prefix present) — the latter means the prediction
/// set does not cover the instrument's applied CE and is a hard error.
pub fn fragment_prefix_exists<V>(
    map: &std::collections::BTreeMap<(u32, i8, i32), V>,
    peptide_id: u32,
    charge: i8,
) -> bool {
    map.range((peptide_id, charge, i32::MIN)..=(peptide_id, charge, i32::MAX))
        .next()
        .is_some()
}

/// Mean of consecutive differences of `xs` (the frame `rt_cycle_length` /
/// `im_cycle_length` the legacy jobs derive via `np.mean(np.diff(...))`).
fn mean_consecutive_diff(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let total: f64 = xs.windows(2).map(|w| w[1] - w[0]).sum();
    total / (xs.len() - 1) as f64
}

/// How the stored `fragment_ions` were produced (P5 prediction set). The
/// renderer reads this to verify the stored fragments are compatible with the
/// instrument it is rendering for, instead of silently rendering against a set
/// built for a different instrument / collision-energy encoding.
#[derive(Debug, Clone)]
pub struct PredictionSet {
    pub prediction_set_id: i64,
    pub predictor_model: Option<String>,
    pub instrument: String,
    pub acquisition_type: String,
    pub activation_method: String,
    pub energy_unit: String,
    /// How collision energy is encoded in `fragment_ions` — the render-time CE
    /// keying (`resolve_fragment_ce_key`) only matches the legacy
    /// `normalized_div100` encoding.
    pub collision_energy_encoding: String,
}

impl PredictionSet {
    /// The implicit set for pre-P5 DBs (no `prediction_sets` table): timsTOF
    /// collisional activation, CE stored normalized (raw/100), eV.
    pub fn legacy_bruker() -> Self {
        PredictionSet {
            prediction_set_id: 0,
            predictor_model: None,
            instrument: "bruker_timstof".to_string(),
            acquisition_type: "unknown".to_string(),
            activation_method: "hcd".to_string(),
            energy_unit: "ev".to_string(),
            collision_energy_encoding: "normalized_div100".to_string(),
        }
    }

    /// Fail unless this set's collision-energy encoding is the one the render-time
    /// CE keying assumes. Guards against rendering fragments that were stored with
    /// an encoding the current keying cannot resolve (e.g. a future Thermo set).
    pub fn assert_render_compatible(&self) -> Result<(), String> {
        if self.collision_energy_encoding != "normalized_div100" {
            return Err(format!(
                "prediction set {} has collision_energy_encoding='{}' but the \
                 renderer expects 'normalized_div100'; the stored fragments were \
                 built for a different instrument/encoding (model={:?}, instrument={})",
                self.prediction_set_id,
                self.collision_energy_encoding,
                self.predictor_model,
                self.instrument,
            ));
        }
        Ok(())
    }
}

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
        let mut stmt = self.connection.prepare(
            "SELECT frame_id, time, ms_type FROM frames"
        )?;
        let frames_iter = stmt.query_map([], |row| {
            Ok(FramesSim::new(
                row.get("frame_id")?,
                row.get("time")?,
                row.get("ms_type")?,
            ))
        })?;
        let mut frames = Vec::new();
        for frame in frames_iter {
            frames.push(frame?);
        }
        Ok(frames)
    }

    pub fn read_scans(&self) -> rusqlite::Result<Vec<ScansSim>> {
        let mut stmt = self.connection.prepare(
            "SELECT scan, mobility FROM scans ORDER BY scan"
        )?;
        let scans_iter = stmt.query_map([], |row| {
            Ok(ScansSim::new(
                row.get("scan")?,
                row.get("mobility")?,
            ))
        })?;
        let mut scans = Vec::new();
        for scan in scans_iter {
            scans.push(scan?);
        }
        Ok(scans)
    }

    pub fn read_peptides(&self) -> rusqlite::Result<Vec<PeptidesSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM peptides ORDER BY peptide_id")?;
        let peptides_iter = stmt.query_map([], |row| {
            Self::peptide_from_row(row)
        })?;
        let mut peptides = Vec::new();
        for peptide in peptides_iter {
            peptides.push(peptide?);
        }
        Ok(peptides)
    }

    /// Parse a single peptide row by column name (order-independent).
    fn peptide_from_row(row: &rusqlite::Row) -> rusqlite::Result<PeptidesSim> {
        let frame_occurrence_str: String = row.get("frame_occurrence")?;
        let frame_abundance_str: String = row.get("frame_abundance")?;

        let frame_occurrence: Vec<u32> = serde_json::from_str(&frame_occurrence_str)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        let frame_abundance: Vec<f32> = match serde_json::from_str(&frame_abundance_str) {
            Ok(value) => value,
            Err(_) => vec![0.0; frame_occurrence.len()],
        };

        let frame_distribution =
            SignalDistribution::new(0.0, 0.0, 0.0, frame_occurrence, frame_abundance);

        let peptide_id: u32 = row.get("peptide_id")?;

        Ok(PeptidesSim {
            protein_id: row.get("protein_id")?,
            peptide_id,
            sequence: PeptideSequence::new(row.get("sequence")?, Some(peptide_id as i32)),
            proteins: row.get("protein")?,
            decoy: row.get("decoy")?,
            missed_cleavages: row.get("missed_cleavages")?,
            n_term: row.get("n_term")?,
            c_term: row.get("c_term")?,
            mono_isotopic_mass: row.get("monoisotopic-mass")?,
            retention_time: row.get("retention_time_gru_predictor")?,
            events: row.get("events")?,
            frame_start: row.get("frame_occurrence_start")?,
            frame_end: row.get("frame_occurrence_end")?,
            frame_distribution,
        })
    }

    /// Parse a single ion row by column name (order-independent).
    fn ion_from_row(row: &rusqlite::Row) -> rusqlite::Result<IonSim> {
        let simulated_spectrum_str: String = row.get("simulated_spectrum")?;
        let scan_occurrence_str: String = row.get("scan_occurrence")?;
        let scan_abundance_str: String = row.get("scan_abundance")?;

        let simulated_spectrum: MzSpectrum = serde_json::from_str(&simulated_spectrum_str)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        let scan_occurrence: Vec<u32> = serde_json::from_str(&scan_occurrence_str)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        let scan_abundance: Vec<f32> = serde_json::from_str(&scan_abundance_str)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        Ok(IonSim::new(
            row.get("ion_id")?,
            row.get("peptide_id")?,
            row.get("sequence")?,
            row.get("charge")?,
            row.get("relative_abundance")?,
            row.get("inv_mobility_gru_predictor")?,
            simulated_spectrum,
            scan_occurrence,
            scan_abundance,
        ))
    }

    /// Parse a single fragment ion row by column name (order-independent).
    fn fragment_ion_from_row(row: &rusqlite::Row) -> rusqlite::Result<FragmentIonSim> {
        let indices_string: String = row.get("indices")?;
        let values_string: String = row.get("values")?;

        let indices: Vec<u32> = serde_json::from_str(&indices_string)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        let values: Vec<f64> = serde_json::from_str(&values_string)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                0, rusqlite::types::Type::Text, Box::new(e),
            ))?;

        Ok(FragmentIonSim::new(
            row.get("peptide_id")?,
            row.get("ion_id")?,
            row.get("collision_energy")?,
            row.get("charge")?,
            indices,
            values,
        ))
    }

    pub fn read_ions(&self) -> rusqlite::Result<Vec<IonSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM ions ORDER BY ion_id")?;
        let ions_iter = stmt.query_map([], |row| Self::ion_from_row(row))?;
        let mut ions = Vec::new();
        for ion in ions_iter {
            ions.push(ion?);
        }
        Ok(ions)
    }

    // ----------------------------------------------------------------------- //
    // Instrument-dispatch P1: parallel scalar-native readers.
    //
    // These read ONLY the trunk scalar physics — no JSON occurrence/abundance
    // vectors — and are additive (the legacy read_peptides/read_ions above are
    // untouched). For a legacy DB that persisted 1/K0 (and no `ccs` column),
    // CCS is derived under the supplied `MobilityEnv`.
    // ----------------------------------------------------------------------- //

    /// Reject anything that isn't a bare SQL identifier (PRAGMA can't be
    /// parameterised, so the table name is interpolated — guard it even though
    /// all internal callers pass literals).
    fn assert_ident(name: &str) -> rusqlite::Result<()> {
        let ok = !name.is_empty()
            && name.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if ok {
            Ok(())
        } else {
            Err(rusqlite::Error::InvalidParameterName(format!(
                "unsafe SQL identifier: {name:?}"
            )))
        }
    }

    /// True if `table` exists and has `column` (used to stay forward/backward
    /// compatible across schema versions without per-row failures).
    fn table_has_column(&self, table: &str, column: &str) -> rusqlite::Result<bool> {
        Self::assert_ident(table)?;
        let mut stmt = self
            .connection
            .prepare(&format!("PRAGMA table_info({})", table))?;
        let mut found = false;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for name in rows {
            if name? == column {
                found = true;
            }
        }
        Ok(found)
    }

    /// Read the run's mobility environment from `experiment_conditions`, falling
    /// back to timsTOF defaults when the table/columns are absent (pre-P1 DB).
    /// Requires all three env columns; a partially-migrated table falls back to
    /// defaults rather than erroring on a missing column.
    pub fn read_mobility_env(&self) -> rusqlite::Result<MobilityEnv> {
        let complete = self.table_has_column("experiment_conditions", "drift_gas_mass")?
            && self.table_has_column("experiment_conditions", "temperature_c")?
            && self.table_has_column("experiment_conditions", "t_diff")?;
        if !complete {
            return Ok(MobilityEnv::default());
        }
        let mut stmt = self.connection.prepare(
            "SELECT drift_gas_mass, temperature_c, t_diff FROM experiment_conditions LIMIT 1",
        )?;
        let mut rows = stmt.query_map([], |row| {
            Ok(MobilityEnv {
                gas_mass: row.get(0)?,
                temp_c: row.get(1)?,
                t_diff: row.get(2)?,
            })
        })?;
        match rows.next() {
            Some(env) => env,
            None => Ok(MobilityEnv::default()),
        }
    }

    /// Read peptides as scalar-native trunk entities (no frame-occurrence vectors).
    fn peptide_scalar_from_row(
        row: &rusqlite::Row,
        has_condition: bool,
        has_rt_mu: bool,
    ) -> rusqlite::Result<PeptideScalar> {
        let peptide_id: u32 = row.get("peptide_id")?;
        let condition_id = if has_condition {
            row.get::<_, Option<i64>>("condition_id")?
        } else {
            None
        };
        let retention_time: f32 = row.get("retention_time_gru_predictor")?;
        // The EMG location is rt_mu (derived from the apex); fall back to the
        // apex itself only when the column is absent (pre-distribution DB). Read
        // as f64 (DB REAL) for byte-faithful projection.
        let rt_mu: f64 = if has_rt_mu { row.get("rt_mu")? } else { retention_time as f64 };
        Ok(PeptideScalar {
            protein_id: row.get("protein_id")?,
            peptide_id,
            sequence: PeptideSequence::new(row.get("sequence")?, Some(peptide_id as i32)),
            proteins: row.get("protein")?,
            decoy: row.get("decoy")?,
            missed_cleavages: row.get("missed_cleavages")?,
            n_term: row.get("n_term")?,
            c_term: row.get("c_term")?,
            mono_isotopic_mass: row.get("monoisotopic-mass")?,
            retention_time,
            rt_mu,
            rt_sigma: row.get("rt_sigma")?,
            rt_lambda: row.get("rt_lambda")?,
            events: row.get("events")?,
            condition_id,
        })
    }

    fn ion_scalar_from_row(
        row: &rusqlite::Row,
        env: &MobilityEnv,
        has_ccs: bool,
        has_condition: bool,
        has_inv_std: bool,
    ) -> rusqlite::Result<IonScalar> {
        let simulated_spectrum_str: String = row.get("simulated_spectrum")?;
        let simulated_spectrum: MzSpectrum = serde_json::from_str(&simulated_spectrum_str)
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
        let charge: i8 = row.get("charge")?;
        let mz: f64 = row.get("mz")?;
        let ccs: f64 = if has_ccs {
            row.get("ccs")?
        } else {
            let one_over_k0: f64 = row.get("inv_mobility_gru_predictor")?;
            env.ccs_from_inv_mobility(one_over_k0, mz, charge)
        };
        // Probe presence explicitly; propagate real conversion errors
        // (only a genuinely absent column defaults to 0.0).
        let inv_mobility_std: f64 = if has_inv_std {
            row.get("inv_mobility_gru_predictor_std")?
        } else {
            0.0
        };
        let condition_id = if has_condition {
            row.get::<_, Option<i64>>("condition_id")?
        } else {
            None
        };
        Ok(IonScalar {
            ion_id: row.get("ion_id")?,
            peptide_id: row.get("peptide_id")?,
            sequence: row.get("sequence")?,
            charge,
            relative_abundance: row.get("relative_abundance")?,
            mz,
            ccs,
            inv_mobility_std,
            simulated_spectrum,
            condition_id,
        })
    }

    pub fn read_peptides_scalar(&self) -> rusqlite::Result<Vec<PeptideScalar>> {
        let has_condition = self.table_has_column("peptides", "condition_id")?;
        let has_rt_mu = self.table_has_column("peptides", "rt_mu")?;
        let mut stmt = self.connection.prepare("SELECT * FROM peptides ORDER BY peptide_id")?;
        let iter = stmt.query_map([], |row| {
            Self::peptide_scalar_from_row(row, has_condition, has_rt_mu)
        })?;
        let mut out = Vec::new();
        for p in iter {
            out.push(p?);
        }
        Ok(out)
    }

    /// Read scalar peptides for a specific set of ids (chunked `IN` query), so
    /// the lazy projector path loads only the batch's candidates instead of the
    /// whole table. Order matches `read_peptides_scalar` (ORDER BY peptide_id).
    pub fn read_peptides_scalar_for_ids(
        &self,
        peptide_ids: &[u32],
    ) -> rusqlite::Result<Vec<PeptideScalar>> {
        if peptide_ids.is_empty() {
            return Ok(Vec::new());
        }
        let has_condition = self.table_has_column("peptides", "condition_id")?;
        let has_rt_mu = self.table_has_column("peptides", "rt_mu")?;
        const CHUNK_SIZE: usize = 500;
        let mut out = Vec::new();
        for chunk in peptide_ids.chunks(CHUNK_SIZE) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT * FROM peptides WHERE peptide_id IN ({}) ORDER BY peptide_id",
                placeholders
            );
            let mut stmt = self.connection.prepare(&sql)?;
            let iter = stmt.query_map(rusqlite::params_from_iter(chunk.iter()), |row| {
                Self::peptide_scalar_from_row(row, has_condition, has_rt_mu)
            })?;
            for p in iter {
                out.push(p?);
            }
        }
        Ok(out)
    }

    /// Read ions as scalar-native trunk entities (no scan-occurrence vectors).
    /// CCS is read directly if a `ccs` column exists, otherwise derived from the
    /// legacy `inv_mobility_gru_predictor` (1/K0) under `env`.
    pub fn read_ions_scalar(&self, env: &MobilityEnv) -> rusqlite::Result<Vec<IonScalar>> {
        let has_ccs = self.table_has_column("ions", "ccs")?;
        let has_condition = self.table_has_column("ions", "condition_id")?;
        let has_inv_std = self.table_has_column("ions", "inv_mobility_gru_predictor_std")?;
        let mut stmt = self.connection.prepare("SELECT * FROM ions ORDER BY ion_id")?;
        let env = *env;
        let iter = stmt.query_map([], move |row| {
            Self::ion_scalar_from_row(row, &env, has_ccs, has_condition, has_inv_std)
        })?;
        let mut out = Vec::new();
        for i in iter {
            out.push(i?);
        }
        Ok(out)
    }

    /// Read scalar ions for a specific set of peptide ids (chunked `IN` query).
    /// The lazy projector path uses this so it does NOT load + JSON-deserialize
    /// every ion's simulated spectrum per batch (that would restore eager-scale
    /// memory). Within a peptide, ions are ordered by ion_id (chunking is by
    /// peptide_id, so a peptide's ions all live in one chunk).
    pub fn read_ions_scalar_for_peptides(
        &self,
        peptide_ids: &[u32],
        env: &MobilityEnv,
    ) -> rusqlite::Result<Vec<IonScalar>> {
        if peptide_ids.is_empty() {
            return Ok(Vec::new());
        }
        let has_ccs = self.table_has_column("ions", "ccs")?;
        let has_condition = self.table_has_column("ions", "condition_id")?;
        let has_inv_std = self.table_has_column("ions", "inv_mobility_gru_predictor_std")?;
        let env = *env;
        const CHUNK_SIZE: usize = 500;
        let mut out = Vec::new();
        for chunk in peptide_ids.chunks(CHUNK_SIZE) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT * FROM ions WHERE peptide_id IN ({}) ORDER BY peptide_id, ion_id",
                placeholders
            );
            let mut stmt = self.connection.prepare(&sql)?;
            let iter = stmt.query_map(rusqlite::params_from_iter(chunk.iter()), move |row| {
                Self::ion_scalar_from_row(row, &env, has_ccs, has_condition, has_inv_std)
            })?;
            for i in iter {
                out.push(i?);
            }
        }
        Ok(out)
    }

    // ----------------------------------------------------------------------- //
    // Instrument-dispatch P4 (canonical builder state): source-aware entity
    // reads. Produce the SAME PeptidesSim/IonSim the column path produces, but
    // with the occurrence/abundance distributions taken from either the legacy
    // JSON columns (`Columns`) or the render-time projector (`Projector`). Every
    // downstream consumer reads the same entity fields, so nothing else changes.
    // ----------------------------------------------------------------------- //

    /// Read peptides with their frame distribution from `source`.
    pub fn read_peptides_with_source(
        &self,
        source: &crate::sim::projector::DistributionSource,
    ) -> rusqlite::Result<Vec<PeptidesSim>> {
        use crate::sim::projector::DistributionSource;
        let (mode, params) = match source {
            DistributionSource::Columns => return self.read_peptides(),
            DistributionSource::Projector { mode, params, .. } => (*mode, *params),
        };
        let scalars = self.read_peptides_scalar()?;
        self.project_peptide_scalars(scalars, mode, params)
    }

    /// Convert a projected abundance to the stored `f32`, applying LegacyCompat's
    /// bit-compatibility rounding. The legacy / `project_distributions` writer
    /// stores abundances as `float(np.round(x, num_decimals))`
    /// (utility.python_list_to_json_string), AFTER the remove_epsilon threshold.
    /// LegacyCompat reproduces that exact stored value (configurable precision +
    /// NumPy's round-half-to-even) so projector-fed rendering byte-matches
    /// column-fed rendering; Accurate keeps full precision.
    fn abundance_for_mode(
        a: f64,
        mode: crate::sim::projector::ProjectionMode,
        num_decimals: u32,
    ) -> f32 {
        match mode {
            crate::sim::projector::ProjectionMode::LegacyCompat => {
                let scale = 10f64.powi(num_decimals as i32);
                // round_ties_even matches np.round (banker's rounding); f64::round
                // rounds half away from zero and would break byte-compat on exact
                // half-way abundances.
                ((a * scale).round_ties_even() / scale) as f32
            }
            crate::sim::projector::ProjectionMode::Accurate => a as f32,
        }
    }

    /// Project a (possibly filtered) set of scalar peptides into `PeptidesSim`
    /// with projector-filled frame distributions. Shared by the eager
    /// (full-table) reader and the lazy per-batch reader so there is one
    /// projection implementation. Projection is per-peptide independent, so a
    /// filtered subset yields identical entities to projecting all then
    /// filtering.
    fn project_peptide_scalars(
        &self,
        scalars: Vec<PeptideScalar>,
        mode: crate::sim::projector::ProjectionMode,
        params: crate::sim::projector::ProjectionParams,
    ) -> rusqlite::Result<Vec<PeptidesSim>> {
        use crate::sim::projector::ProjectionMode;
        // Frames table, ascending by id, as the projector expects.
        let mut frames = self.read_frames()?;
        frames.sort_by_key(|f| f.frame_id);
        let frame_ids: Vec<u32> = frames.iter().map(|f| f.frame_id).collect();
        let frame_times: Vec<f64> = frames.iter().map(|f| f.time as f64).collect();
        if frame_times.len() < 2 {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let rt_cycle = mean_consecutive_diff(&frame_times);
        let mus: Vec<f64> = scalars.iter().map(|p| p.rt_mu).collect();
        let sigmas: Vec<f64> = scalars.iter().map(|p| p.rt_sigma).collect();
        let lambdas: Vec<f64> = scalars.iter().map(|p| p.rt_lambda).collect();

        let projected: Vec<Vec<(u32, f64)>> = match mode {
            ProjectionMode::LegacyCompat => crate::sim::projector::project_time_legacy(
                &mus, &sigmas, &lambdas, &frame_ids, &frame_times, rt_cycle, params.target_p,
                params.frame_step_size, params.n_steps, params.remove_epsilon, params.num_threads,
            ),
            ProjectionMode::Accurate => {
                // Accurate: integrate each frame's true [prev, this] exposure
                // interval; map event index -> frame id; apply remove_epsilon.
                let mut starts = Vec::with_capacity(frame_times.len());
                let mut ends = Vec::with_capacity(frame_times.len());
                for (i, &t) in frame_times.iter().enumerate() {
                    let prev = if i > 0 { frame_times[i - 1] } else { t - (frame_times[1] - frame_times[0]) };
                    starts.push(prev);
                    ends.push(t);
                }
                let intervals: Vec<(f64, f64)> = starts.into_iter().zip(ends).collect();
                let proj = mscore::algorithm::utility::project_emg_over_events_par(
                    &intervals, mus.clone(), sigmas.clone(), lambdas.clone(), params.target_p,
                    params.frame_step_size, params.num_threads.max(1), params.n_steps,
                );
                proj.into_iter()
                    .map(|pairs| {
                        pairs.into_iter()
                            .filter(|(_, a)| *a > params.remove_epsilon)
                            .map(|(i, a)| (frame_ids[i], a))
                            .collect()
                    })
                    .collect()
            }
        };

        Ok(scalars
            .into_iter()
            .zip(projected)
            .map(|(p, pairs)| {
                let occ: Vec<u32> = pairs.iter().map(|(f, _)| *f).collect();
                let ab: Vec<f32> = pairs.iter().map(|(_, a)| Self::abundance_for_mode(*a, mode, params.num_decimals)).collect();
                let frame_start = occ.first().copied().unwrap_or(0);
                let frame_end = occ.last().copied().unwrap_or(0);
                PeptidesSim::new(
                    p.protein_id, p.peptide_id, p.sequence.sequence.clone(), p.proteins, p.decoy,
                    p.missed_cleavages, p.n_term, p.c_term, p.mono_isotopic_mass, p.retention_time,
                    p.events, frame_start, frame_end, occ, ab,
                )
            })
            .collect())
    }

    /// Read ions with their scan distribution from `source`.
    pub fn read_ions_with_source(
        &self,
        source: &crate::sim::projector::DistributionSource,
    ) -> rusqlite::Result<Vec<IonSim>> {
        use crate::sim::projector::DistributionSource;
        let (mode, env, params) = match source {
            DistributionSource::Columns => return self.read_ions(),
            DistributionSource::Projector { mode, env, params } => (*mode, *env, *params),
        };
        let scalars = self.read_ions_scalar(&env)?;
        self.project_ion_scalars(scalars, mode, env, params)
    }

    /// Project a (possibly filtered) set of scalar ions into `IonSim` with
    /// projector-filled scan distributions. Shared by the eager (full-table)
    /// reader and the lazy per-batch reader. Per-ion independent, so a filtered
    /// subset yields identical entities.
    fn project_ion_scalars(
        &self,
        scalars: Vec<IonScalar>,
        mode: crate::sim::projector::ProjectionMode,
        env: MobilityEnv,
        params: crate::sim::projector::ProjectionParams,
    ) -> rusqlite::Result<Vec<IonSim>> {
        use crate::sim::projector::ProjectionMode;
        // Scans ascending by mobility (im_cycle_length > 0), as the projector expects.
        let mut scans = self.read_scans()?;
        scans.sort_by(|a, b| a.mobility.partial_cmp(&b.mobility).unwrap_or(std::cmp::Ordering::Equal));
        let scan_ids: Vec<u32> = scans.iter().map(|s| s.scan).collect();
        let scan_mob: Vec<f64> = scans.iter().map(|s| s.mobility as f64).collect();
        if scan_mob.len() < 2 {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let im_cycle = mean_consecutive_diff(&scan_mob);
        let means: Vec<f64> = scalars.iter().map(|i| i.inv_mobility(&env)).collect();
        let sigmas: Vec<f64> = scalars.iter().map(|i| i.inv_mobility_std).collect();

        let projected: Vec<Vec<(i32, f64)>> = match mode {
            ProjectionMode::LegacyCompat => crate::sim::projector::project_mobility_legacy_par(
                &means, &sigmas, &scan_ids, &scan_mob, im_cycle, params.target_p,
                params.scan_step_size, params.num_threads,
            ),
            ProjectionMode::Accurate => {
                // Accurate per-scan midpoint bins; ascending-grid index -> scan id.
                let acc = crate::sim::projector::project_mobility_accurate_par(
                    &means, &sigmas, &scan_mob, params.target_p, params.scan_step_size,
                    params.num_threads,
                );
                acc.into_iter()
                    .map(|pairs| pairs.into_iter().map(|(idx, a)| (scan_ids[idx as usize] as i32, a)).collect())
                    .collect()
            }
        };

        Ok(scalars
            .into_iter()
            .zip(projected)
            .map(|(ion, pairs)| {
                let occ: Vec<u32> = pairs.iter().map(|(s, _)| *s as u32).collect();
                let ab: Vec<f32> = pairs.iter().map(|(_, a)| Self::abundance_for_mode(*a, mode, params.num_decimals)).collect();
                let mobility = ion.inv_mobility(&env) as f32;
                IonSim::new(
                    ion.ion_id, ion.peptide_id, ion.sequence, ion.charge, ion.relative_abundance,
                    mobility, ion.simulated_spectrum, occ, ab,
                )
            })
            .collect())
    }

    /// Read the fragment prediction set (P5). Returns the single registered set,
    /// or `PredictionSet::legacy_bruker()` for pre-P5 DBs that have no
    /// `prediction_sets` table (so existing data keeps rendering as the implicit
    /// Bruker set). Distinguishes "verified provenance" from "assumed legacy" by
    /// virtue of the table's presence.
    pub fn read_prediction_set(&self) -> rusqlite::Result<PredictionSet> {
        let has_table: bool = self
            .connection
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_sets'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !has_table {
            return Ok(PredictionSet::legacy_bruker());
        }
        self.connection.query_row(
            "SELECT prediction_set_id, predictor_model, instrument, acquisition_type, \
             activation_method, energy_unit, collision_energy_encoding \
             FROM prediction_sets ORDER BY prediction_set_id LIMIT 1",
            [],
            |row| {
                Ok(PredictionSet {
                    prediction_set_id: row.get(0)?,
                    predictor_model: row.get(1)?,
                    instrument: row.get(2)?,
                    acquisition_type: row.get(3)?,
                    activation_method: row.get(4)?,
                    energy_unit: row.get(5)?,
                    collision_energy_encoding: row.get(6)?,
                })
            },
        )
    }

    /// Candidate peptide ids overlapping a frame range, by the legacy occurrence
    /// columns. P4a2 uses this as the lazy candidate index for BOTH the columns
    /// and projector paths; P4d replaces it with a scalar RT-support index that
    /// does not depend on the occurrence columns. For LegacyCompat this is exact;
    /// for Accurate it is the same conservative window the columns already define.
    fn candidate_peptide_ids_for_frame_range(
        &self,
        frame_min: u32,
        frame_max: u32,
    ) -> rusqlite::Result<Vec<u32>> {
        let mut stmt = self.connection.prepare(
            "SELECT peptide_id FROM peptides WHERE frame_occurrence_start <= ?1 AND frame_occurrence_end >= ?2 ORDER BY peptide_id",
        )?;
        let iter = stmt.query_map([frame_max, frame_min], |row| row.get::<_, u32>(0))?;
        let mut out = Vec::new();
        for id in iter {
            out.push(id?);
        }
        Ok(out)
    }

    /// Lazy per-batch peptide read, source-aware. `Columns` returns the legacy
    /// column-fed `read_peptides_for_frame_range`; `Projector` selects candidates
    /// for the range, reads their scalars, and projects (same entities the eager
    /// projector reader produces, restricted to the batch).
    pub fn read_peptides_for_frame_range_with_source(
        &self,
        frame_min: u32,
        frame_max: u32,
        source: &crate::sim::projector::DistributionSource,
    ) -> rusqlite::Result<Vec<PeptidesSim>> {
        use crate::sim::projector::DistributionSource;
        let (mode, params) = match source {
            DistributionSource::Columns => {
                return self.read_peptides_for_frame_range(frame_min, frame_max)
            }
            DistributionSource::Projector { mode, params, .. } => (*mode, *params),
        };
        // Candidate selection by the legacy occurrence-column range query. Exact
        // for LegacyCompat (the projector reproduces the column kernel); P4d
        // replaces this with a scalar RT-support index that (a) does not depend
        // on the occurrence columns and (b) conservatively covers Accurate
        // support that may extend beyond the legacy stored range.
        //
        // Until P4d lands that index, lazy + Accurate would silently OMIT
        // peptides whose accurate RT support extends past the stored occurrence
        // window at the batch boundary. The eager Accurate path (read_peptides_
        // with_source) has no such window and is complete; production wires lazy
        // to Columns only. Fail loudly here rather than truncate the signal: the
        // connector exposes lazy+Accurate directly, and silent omission is worse
        // than an explicit "not yet supported".
        if matches!(mode, crate::sim::projector::ProjectionMode::Accurate) {
            return Err(rusqlite::Error::InvalidParameterName(
                "lazy loading with projection_mode='accurate' is not yet supported \
                 (the per-batch candidate index uses legacy occurrence columns and \
                 would omit peptides whose accurate RT support crosses a batch \
                 boundary); use eager loading for accurate projection, or \
                 projection_mode='legacy_compat' with lazy loading"
                    .to_string(),
            ));
        }
        let candidate_ids = self.candidate_peptide_ids_for_frame_range(frame_min, frame_max)?;
        // Load + project only the batch's candidate scalars (filtered in SQL).
        let scalars = self.read_peptides_scalar_for_ids(&candidate_ids)?;
        self.project_peptide_scalars(scalars, mode, params)
    }

    /// Lazy per-batch ion read, source-aware. `Columns` returns the legacy
    /// `read_ions_for_peptides`; `Projector` reads the scalars for these peptides
    /// and projects their scan distributions.
    pub fn read_ions_for_peptides_with_source(
        &self,
        peptide_ids: &[u32],
        source: &crate::sim::projector::DistributionSource,
    ) -> rusqlite::Result<Vec<IonSim>> {
        use crate::sim::projector::DistributionSource;
        let (mode, env, params) = match source {
            DistributionSource::Columns => return self.read_ions_for_peptides(peptide_ids),
            DistributionSource::Projector { mode, env, params } => (*mode, *env, *params),
        };
        // Filter ions to the requested peptides in SQL — avoids loading and
        // JSON-deserializing every ion's simulated spectrum per batch.
        let scalars = self.read_ions_scalar_for_peptides(peptide_ids, &env)?;
        self.project_ion_scalars(scalars, mode, env, params)
    }

    pub fn read_window_group_settings(&self) -> rusqlite::Result<Vec<WindowGroupSettingsSim>> {
        let mut stmt = self.connection.prepare("SELECT * FROM dia_ms_ms_windows")?;
        let window_group_settings_iter = stmt.query_map([], |row| {
            Ok(WindowGroupSettingsSim::new(
                row.get("window_group")?,
                row.get("scan_start")?,
                row.get("scan_end")?,
                row.get("isolation_mz")?,
                row.get("isolation_width")?,
                row.get("collision_energy")?,
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
                row.get("frame")?,
                row.get("window_group")?,
            ))
        })?;

        let mut frame_to_window_groups: Vec<FrameToWindowGroupSim> = Vec::new();
        for frame_to_window_group in frame_to_window_group_iter {
            frame_to_window_groups.push(frame_to_window_group?);
        }

        Ok(frame_to_window_groups)
    }

    pub fn read_pasef_meta(&self) -> rusqlite::Result<Vec<PASEFMeta>> {
        let mut stmt = self.connection.prepare(
            "SELECT frame, scan_start, scan_end, isolation_mz, isolation_width, collision_energy, precursor FROM pasef_meta"
        )?;
        let pasef_meta_iter = stmt.query_map([], |row| {
            Ok(PASEFMeta::new(
                row.get("frame")?,
                row.get("scan_start")?,
                row.get("scan_end")?,
                row.get("isolation_mz")?,
                row.get("isolation_width")?,
                row.get("collision_energy")?,
                row.get("precursor")?,
            ))
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
            "SELECT * FROM peptides WHERE frame_occurrence_start <= ?1 AND frame_occurrence_end >= ?2 ORDER BY peptide_id"
        )?;

        let peptides_iter = stmt.query_map([frame_max, frame_min], |row| {
            Self::peptide_from_row(row)
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

            // ORDER BY peptide_id, ion_id matches the eager full-table read
            // (`read_ions`). Ion order within a peptide is load-bearing: the DDA
            // fragment builder selects an ion via `charges.position(charge)`, so a
            // different order can fragment a different ion. Chunking is by
            // peptide_id, so each peptide's ions live in one chunk and are fully
            // ordered here.
            let sql = format!(
                "SELECT * FROM ions WHERE peptide_id IN ({}) ORDER BY peptide_id, ion_id",
                placeholders
            );

            let mut stmt = self.connection.prepare(&sql)?;

            let ions_iter = stmt.query_map(
                rusqlite::params_from_iter(chunk.iter()),
                |row| Self::ion_from_row(row),
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
                "SELECT * FROM fragment_ions WHERE peptide_id IN ({}) ORDER BY peptide_id",
                placeholders
            );

            let mut stmt = self.connection.prepare(&sql)?;

            let fragment_ion_iter = stmt.query_map(
                rusqlite::params_from_iter(chunk.iter()),
                |row| Self::fragment_ion_from_row(row),
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
            Self::fragment_ion_from_row(row)
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
        _precursor_frames: &HashSet<u32>,
        transmission: &TimsTransmissionDDA,
    ) -> BTreeSet<(u32, u32, String, i8, i32)> {
        let peptide = peptide_map.get(&ion.peptide_id).unwrap();
        let mut ret_tree: BTreeSet<(u32, u32, String, i8, i32)> = BTreeSet::new();

        // Get all frames where this ion was explicitly selected for fragmentation
        // using the precursor ID from pasef_meta (instead of re-computing m/z transmission)
        let selections = transmission.get_selections_for_precursor(ion.ion_id as i32);

        for (_, collision_energy) in selections {
            // Quantize with same factor as DIA (line 614) so return conversion (line 702)
            // divides by 100 to recover raw CE, matching the Python normalization flow
            let quantized_energy = (collision_energy * 100.0).round() as i32;

            ret_tree.insert((
                ion.peptide_id,
                ion.ion_id,
                peptide.sequence.sequence.clone(),
                ion.charge,
                quantized_energy,
            ));
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

#[cfg(test)]
mod scalar_reader_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn prediction_set_compatibility() {
        // Legacy / Bruker set: render-compatible.
        assert!(PredictionSet::legacy_bruker().assert_render_compatible().is_ok());
        // A set with a different CE encoding (e.g. a future Thermo NCE set) is
        // rejected by the renderer rather than silently mis-keyed.
        let mut s = PredictionSet::legacy_bruker();
        s.instrument = "orbitrap_astral".to_string();
        s.collision_energy_encoding = "nce_raw".to_string();
        assert!(s.assert_render_compatible().is_err());
    }

    #[test]
    fn resolve_fragment_ce_key_tolerates_quantization_noise() {
        // Map keyed at the stored 0.1-eV resolution (round(stored*1e3) ~ raw*10).
        let mut map: BTreeMap<(u32, i8, i32), ()> = BTreeMap::new();
        map.insert((1, 2, 330), ()); // stored key for ~33.0 eV

        // Exact natural key (raw 33.0 -> 330) resolves.
        assert_eq!(resolve_fragment_ce_key(&map, 1, 2, 33.00), Some(330));
        // Raw 33.0499 -> natural key 331, but stored is 330: the ±1 probe finds it.
        assert_eq!(resolve_fragment_ce_key(&map, 1, 2, 33.0499), Some(330));
        // Raw 32.95 -> natural key 330 (rounds up) — direct hit.
        assert_eq!(resolve_fragment_ce_key(&map, 1, 2, 32.95), Some(330));
        // A genuinely different CE (~35 eV, key ~350) is >1 away -> real miss.
        assert_eq!(resolve_fragment_ce_key(&map, 1, 2, 35.0), None);
        // Different (peptide, charge) -> miss.
        assert_eq!(resolve_fragment_ce_key(&map, 9, 2, 33.0), None);
    }

    /// Build a minimal legacy-shaped synthetic_data.db (1/K0 only, no `ccs`,
    /// no `condition_id`) and prove the scalar readers work + the legacy
    /// 1/K0 -> CCS -> 1/K0 adapter round-trips under the read mobility env.
    #[test]
    fn scalar_readers_on_legacy_db() {
        let path = std::env::temp_dir().join(format!(
            "rustdf_scalar_test_{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let one_over_k0 = 0.85_f64;
        let (mz, charge) = (500.25_f64, 2_i8);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE peptides (
                    protein_id INTEGER, peptide_id INTEGER, sequence TEXT, protein TEXT,
                    decoy INTEGER, missed_cleavages INTEGER, n_term, c_term,
                    \"monoisotopic-mass\" REAL,
                    retention_time_gru_predictor REAL, rt_sigma REAL, rt_lambda REAL, events REAL
                 );
                 CREATE TABLE ions (
                    ion_id INTEGER, peptide_id INTEGER, sequence TEXT, charge INTEGER,
                    relative_abundance REAL, mz REAL, inv_mobility_gru_predictor REAL,
                    inv_mobility_gru_predictor_std REAL, simulated_spectrum TEXT
                 );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO peptides VALUES (1,1,'PEPTIDEK','P',0,0,NULL,NULL,930.5,123.4,1.2,0.3,5.0)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO ions VALUES (1,1,'PEPTIDEK',?1,1.0,?2,?3,0.02,'{\"mz\":[500.25],\"intensity\":[1.0]}')",
                rusqlite::params![charge, mz, one_over_k0],
            )
            .unwrap();
        }

        let handle = TimsTofSyntheticsDataHandle::new(&path).unwrap();

        // No experiment_conditions table -> default timsTOF env.
        let env = handle.read_mobility_env().unwrap();
        assert_eq!(env, MobilityEnv::default());

        let peptides = handle.read_peptides_scalar().unwrap();
        assert_eq!(peptides.len(), 1);
        assert_eq!(peptides[0].rt_sigma, 1.2);
        assert_eq!(peptides[0].condition_id, None);

        let ions = handle.read_ions_scalar(&env).unwrap();
        assert_eq!(ions.len(), 1);
        // CCS was derived from the legacy 1/K0; re-deriving 1/K0 from that CCS
        // under the same env must return the original (lossless migration).
        let back = ions[0].inv_mobility(&env);
        assert!((back - one_over_k0).abs() < 1e-9, "1/K0 round-trip: {back}");
        assert_eq!(ions[0].condition_id, None);

        let _ = std::fs::remove_file(&path);
    }

    /// P4a0: the source-aware reads produce same-shape PeptidesSim/IonSim from
    /// both `Columns` and `Projector`, and the projector path fills a non-empty
    /// distribution for a mid-gradient analyte.
    #[test]
    fn source_aware_reads_both_paths() {
        use crate::sim::containers::MobilityEnv;
        use crate::sim::projector::{DistributionSource, ProjectionMode, ProjectionParams};
        let path = std::env::temp_dir().join(format!("rustdf_p4a0_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE frames (frame_id INTEGER, time REAL, ms_type INTEGER);
                 CREATE TABLE scans (scan INTEGER, mobility REAL);
                 CREATE TABLE peptides (
                    protein_id INTEGER, peptide_id INTEGER, sequence TEXT, protein TEXT,
                    decoy INTEGER, missed_cleavages INTEGER, n_term, c_term,
                    \"monoisotopic-mass\" REAL, retention_time_gru_predictor REAL, rt_mu REAL,
                    rt_sigma REAL, rt_lambda REAL, events REAL, frame_occurrence TEXT,
                    frame_abundance TEXT, frame_occurrence_start INTEGER, frame_occurrence_end INTEGER);
                 CREATE TABLE ions (
                    ion_id INTEGER, peptide_id INTEGER, sequence TEXT, charge INTEGER,
                    relative_abundance REAL, mz REAL, inv_mobility_gru_predictor REAL,
                    inv_mobility_gru_predictor_std REAL, simulated_spectrum TEXT,
                    scan_occurrence TEXT, scan_abundance TEXT);",
            )
            .unwrap();
            // 40 uniform frames over ~4 s; the peptide elutes mid-gradient (rt_mu=2).
            for fid in 1..=40 {
                conn.execute(
                    "INSERT INTO frames VALUES (?1, ?2, 0)",
                    rusqlite::params![fid, fid as f64 * 0.1],
                )
                .unwrap();
            }
            for s in 0..20 {
                conn.execute(
                    "INSERT INTO scans VALUES (?1, ?2)",
                    rusqlite::params![s, 1.3 - s as f64 * 0.01],
                )
                .unwrap();
            }
            conn.execute(
                "INSERT INTO peptides VALUES (0,1,'PEPTIDEK','P',0,0,NULL,NULL,930.5,2.0,2.0,0.3,0.6,5.0,'[1]','[0.0]',1,1)",
                [],
            ).unwrap();
            conn.execute(
                "INSERT INTO ions VALUES (1,1,'PEPTIDEK',2,1.0,500.0,1.1,0.02,'{\"mz\":[500.0],\"intensity\":[1.0]}','[5]','[1.0]')",
                [],
            ).unwrap();
        }
        let handle = TimsTofSyntheticsDataHandle::new(&path).unwrap();

        // Columns path == read_peptides/read_ions (stored values).
        let pc = handle.read_peptides_with_source(&DistributionSource::Columns).unwrap();
        let ic = handle.read_ions_with_source(&DistributionSource::Columns).unwrap();
        assert_eq!(pc.len(), 1);
        assert_eq!(ic.len(), 1);
        assert_eq!(pc[0].frame_distribution.occurrence, vec![1]); // the stored value

        // Projector path produces a real distribution for the mid-gradient peptide.
        let src = DistributionSource::Projector {
            mode: ProjectionMode::LegacyCompat,
            env: MobilityEnv::default(),
            params: ProjectionParams::default(),
        };
        let pp = handle.read_peptides_with_source(&src).unwrap();
        let ip = handle.read_ions_with_source(&src).unwrap();
        assert_eq!(pp.len(), 1);
        assert_eq!(ip.len(), 1);
        assert!(
            pp[0].frame_distribution.occurrence.len() > 1,
            "projector should populate multiple frames for a mid-gradient peptide"
        );
        assert!(!ip[0].scan_distribution.occurrence.is_empty(), "projector should populate scans");
        // Accurate mode also runs.
        let src_acc = DistributionSource::Projector {
            mode: ProjectionMode::Accurate,
            env: MobilityEnv::default(),
            params: ProjectionParams::default(),
        };
        assert_eq!(handle.read_peptides_with_source(&src_acc).unwrap().len(), 1);
        assert_eq!(handle.read_ions_with_source(&src_acc).unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
    }
}
