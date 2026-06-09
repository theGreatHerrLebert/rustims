use mscore::chemistry::formulas::{
    ccs_to_one_over_reduced_mobility, one_over_reduced_mobility_to_ccs,
};
use mscore::data::peptide::PeptideSequence;
use mscore::data::spectrum::{MsType, MzSpectrum};
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignalDistribution {
    pub mean: f32,
    pub variance: f32,
    pub error: f32,
    pub occurrence: Vec<u32>,
    pub abundance: Vec<f32>,
}

impl SignalDistribution {
    pub fn new(
        mean: f32,
        variance: f32,
        error: f32,
        occurrence: Vec<u32>,
        abundance: Vec<f32>,
    ) -> Self {
        SignalDistribution {
            mean,
            variance,
            error,
            occurrence,
            abundance,
        }
    }

    pub fn add_noise(&self, noise_level: f32) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let noise_dist = Uniform::new(0.0, noise_level);

        let noise: Vec<f32> = self
            .abundance
            .iter()
            .map(|_| noise_dist.sample(&mut rng))
            .collect();
        let noise_relative: Vec<f32> = self
            .abundance
            .iter()
            .zip(noise.iter())
            .map(|(&abu, &noise)| abu * noise)
            .collect();
        let noised_signal: Vec<f32> = self
            .abundance
            .iter()
            .zip(noise_relative.iter())
            .map(|(&abu, &noise_rel)| abu + noise_rel)
            .collect();

        let sum_noised_signal: f32 = noised_signal.iter().sum();
        let sum_rt_abu: f32 = self.abundance.iter().sum();

        noised_signal
            .iter()
            .map(|&x| (x / sum_noised_signal) * sum_rt_abu)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PeptidesSim {
    pub protein_id: u32,
    pub peptide_id: u32,
    pub sequence: PeptideSequence,
    pub proteins: String,
    pub decoy: bool,
    pub missed_cleavages: i8,
    pub n_term: Option<bool>,
    pub c_term: Option<bool>,
    pub mono_isotopic_mass: f32,
    pub retention_time: f32,
    pub events: f32,
    pub frame_start: u32,
    pub frame_end: u32,
    pub frame_distribution: SignalDistribution,
}

impl PeptidesSim {
    pub fn new(
        protein_id: u32,
        peptide_id: u32,
        sequence: String,
        proteins: String,
        decoy: bool,
        missed_cleavages: i8,
        n_term: Option<bool>,
        c_term: Option<bool>,
        mono_isotopic_mass: f32,
        retention_time: f32,
        events: f32,
        frame_start: u32,
        frame_end: u32,
        frame_occurrence: Vec<u32>,
        frame_abundance: Vec<f32>,
    ) -> Self {
        PeptidesSim {
            protein_id,
            peptide_id,
            sequence: PeptideSequence::new(sequence, Some(peptide_id as i32)),
            proteins,
            decoy,
            missed_cleavages,
            n_term,
            c_term,
            mono_isotopic_mass,
            retention_time,
            events,
            frame_start,
            frame_end,
            frame_distribution: SignalDistribution::new(
                0.0,
                0.0,
                0.0,
                frame_occurrence,
                frame_abundance,
            ),
        }
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
    pub window_group: u32,
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
pub struct IonSim {
    pub ion_id: u32,
    pub peptide_id: u32,
    pub sequence: String,
    pub charge: i8,
    pub relative_abundance: f32,
    pub mobility: f32,
    pub simulated_spectrum: MzSpectrum,
    pub scan_distribution: SignalDistribution,
}

impl IonSim {
    pub fn new(
        ion_id: u32,
        peptide_id: u32,
        sequence: String,
        charge: i8,
        relative_abundance: f32,
        mobility: f32,
        simulated_spectrum: MzSpectrum,
        scan_occurrence: Vec<u32>,
        scan_abundance: Vec<f32>,
    ) -> Self {
        IonSim {
            ion_id,
            peptide_id,
            sequence,
            charge,
            relative_abundance,
            mobility,
            simulated_spectrum,
            scan_distribution: SignalDistribution::new(
                0.0,
                0.0,
                0.0,
                scan_occurrence,
                scan_abundance,
            ),
        }
    }
}

// --------------------------------------------------------------------------- //
// Instrument-dispatch P1: parallel scalar-native entities.
//
// These mirror PeptidesSim / IonSim but hold ONLY the vendor-neutral scalar
// physics (the trunk / "ionized sample" of INSTRUMENT_DISPATCH.md) — no
// device-sampled occurrence/abundance vectors. They are additive: the legacy
// SignalDistribution-bearing entities and their readers are untouched.
//
// Mobility ownership (plan §2.3): the trunk stores CCS (intrinsic); 1/K0 is
// derived per instrument under that instrument's mobility environment. For
// legacy DBs that persisted only 1/K0, the scalar reader converts 1/K0 -> CCS
// under a declared reference `MobilityEnv` (see handle.rs).
// --------------------------------------------------------------------------- //

/// Drift-gas mobility environment for the CCS <-> 1/K0 conversion. Defaults
/// match the timsTOF constants used by `mscore::chemistry::formulas` (N2,
/// 31.85 °C, 273.15 K offset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MobilityEnv {
    pub gas_mass: f64,
    pub temp_c: f64,
    pub t_diff: f64,
}

impl Default for MobilityEnv {
    fn default() -> Self {
        MobilityEnv { gas_mass: 28.013, temp_c: 31.85, t_diff: 273.15 }
    }
}

impl MobilityEnv {
    /// CCS for an ion observed at `one_over_k0` (legacy 1/K0 -> trunk CCS).
    /// `charge` is clamped to >= 1 (the conversion is only defined for real
    /// ions; the pipeline never produces charge < 1, asserted in debug).
    pub fn ccs_from_inv_mobility(&self, one_over_k0: f64, mz: f64, charge: i8) -> f64 {
        debug_assert!(charge >= 1, "ion charge must be >= 1, got {charge}");
        one_over_reduced_mobility_to_ccs(
            one_over_k0,
            mz,
            charge.max(1) as u32,
            self.gas_mass,
            self.temp_c,
            self.t_diff,
        )
    }
    /// 1/K0 for an ion of `ccs` under this environment (trunk CCS -> device 1/K0).
    /// `charge` is clamped to >= 1 (see `ccs_from_inv_mobility`).
    pub fn inv_mobility_from_ccs(&self, ccs: f64, mz: f64, charge: i8) -> f64 {
        debug_assert!(charge >= 1, "ion charge must be >= 1, got {charge}");
        ccs_to_one_over_reduced_mobility(
            ccs,
            mz,
            charge.max(1) as u32,
            self.gas_mass,
            self.temp_c,
            self.t_diff,
        )
    }
}

/// Scalar-native peptide: trunk physics with no frame-occurrence vectors.
#[derive(Debug, Clone)]
pub struct PeptideScalar {
    pub protein_id: u32,
    pub peptide_id: u32,
    pub sequence: PeptideSequence,
    pub proteins: String,
    pub decoy: bool,
    pub missed_cleavages: i8,
    pub mono_isotopic_mass: f32,
    /// EMG retention-time profile (seconds): apex + shape, not frame indices.
    pub retention_time: f32,
    pub rt_sigma: f32,
    pub rt_lambda: f32,
    pub events: f32,
    /// Reserved per-analyte condition override (NULL -> the run's single row).
    pub condition_id: Option<i64>,
}

/// Scalar-native ion: trunk physics with no scan-occurrence vectors. Stores CCS
/// as canonical; 1/K0 is derived per instrument via `inv_mobility`.
#[derive(Debug, Clone)]
pub struct IonScalar {
    pub ion_id: u32,
    pub peptide_id: u32,
    pub sequence: String,
    pub charge: i8,
    pub relative_abundance: f32,
    pub mz: f64,
    pub ccs: f64,
    /// Legacy mobility spread (conformer width) in **1/K0 units** as stored — not
    /// CCS-space (renamed from a misleading `ccs_std`). Transforming the spread
    /// into CCS space is deferred; until then read it as a 1/K0 std.
    pub inv_mobility_std: f32,
    /// Isotope composition (m/z + relative intensity), pre-detector.
    pub simulated_spectrum: MzSpectrum,
    pub condition_id: Option<i64>,
}

impl IonScalar {
    /// Derive this ion's 1/K0 under the given instrument mobility environment.
    pub fn inv_mobility(&self, env: &MobilityEnv) -> f64 {
        env.inv_mobility_from_ccs(self.ccs, self.mz, self.charge)
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
    pub ion_id: u32,
    pub collision_energy: f64,
    pub charge: i8,
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl FragmentIonSim {
    pub fn new(
        peptide_id: u32,
        ion_id: u32,
        collision_energy: f64,
        charge: i8,
        indices: Vec<u32>,
        values: Vec<f64>,
    ) -> Self {
        FragmentIonSim {
            peptide_id,
            ion_id,
            charge,
            collision_energy,
            indices,
            values,
        }
    }

    pub fn to_dense(&self, length: usize) -> Vec<f64> {
        let mut dense = vec![0.0; length];
        for (i, &idx) in self.indices.iter().enumerate() {
            dense[idx as usize] = self.values[i];
        }
        dense
    }
}

/// Mode for quad-selection dependent isotope transmission calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsotopeTransmissionMode {
    /// Disabled - no transmission-dependent calculation
    None,
    /// Precursor-based scaling - calculate transmission factor from precursor isotope
    /// distribution and apply uniform scaling to all fragment intensities.
    /// This is computationally efficient and captures the main intensity reduction effect.
    PrecursorScaling,
    /// Per-fragment calculation - calculate transmission-dependent isotope distribution
    /// for each individual fragment ion based on its complementary fragment.
    /// This is more accurate but computationally expensive.
    /// Implements the algorithm from OpenMS's CoarseIsotopePatternGenerator.
    PerFragment,
}

impl Default for IsotopeTransmissionMode {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for quad-selection dependent isotope transmission.
///
/// When enabled, fragment ion isotope distributions and/or intensities are adjusted
/// based on which precursor isotopes were transmitted through the quadrupole isolation
/// window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotopeTransmissionConfig {
    /// Mode for transmission-dependent calculation
    pub mode: IsotopeTransmissionMode,
    /// Minimum probability threshold for isotope transmission
    pub min_probability: f64,
    /// Maximum number of isotope peaks to consider
    pub max_isotopes: usize,
    /// Minimum fraction of precursor ions that survive fragmentation intact (0.0-1.0)
    pub precursor_survival_min: f64,
    /// Maximum fraction of precursor ions that survive fragmentation intact (0.0-1.0)
    pub precursor_survival_max: f64,
}

impl Default for IsotopeTransmissionConfig {
    fn default() -> Self {
        Self {
            mode: IsotopeTransmissionMode::None,
            min_probability: 0.5,
            max_isotopes: 10,
            precursor_survival_min: 0.0,
            precursor_survival_max: 0.0,
        }
    }
}

impl IsotopeTransmissionConfig {
    pub fn new(
        mode: IsotopeTransmissionMode,
        min_probability: f64,
        max_isotopes: usize,
        precursor_survival_min: f64,
        precursor_survival_max: f64,
    ) -> Self {
        Self {
            mode,
            min_probability,
            max_isotopes,
            precursor_survival_min,
            precursor_survival_max,
        }
    }

    /// Create config with precursor scaling mode
    pub fn precursor_scaling(min_probability: f64) -> Self {
        Self {
            mode: IsotopeTransmissionMode::PrecursorScaling,
            min_probability,
            max_isotopes: 10,
            precursor_survival_min: 0.0,
            precursor_survival_max: 0.0,
        }
    }

    /// Create config with per-fragment mode
    pub fn per_fragment(min_probability: f64, max_isotopes: usize) -> Self {
        Self {
            mode: IsotopeTransmissionMode::PerFragment,
            min_probability,
            max_isotopes,
            precursor_survival_min: 0.0,
            precursor_survival_max: 0.0,
        }
    }

    /// Check if precursor survival is enabled
    pub fn has_precursor_survival(&self) -> bool {
        self.precursor_survival_max > 0.0
    }

    /// Check if any transmission mode is enabled
    pub fn is_enabled(&self) -> bool {
        self.mode != IsotopeTransmissionMode::None
    }
}

#[cfg(test)]
mod scalar_entity_tests {
    use super::*;

    #[test]
    fn mobility_env_ccs_inv_mobility_round_trips() {
        // 1/K0 -> CCS -> 1/K0 must be identity under the same environment
        // (the legacy-1/K0 -> trunk-CCS migration must lose nothing).
        let env = MobilityEnv::default();
        let (mz, charge, one_over_k0) = (1000.0_f64, 2_i8, 0.85_f64);
        let ccs = env.ccs_from_inv_mobility(one_over_k0, mz, charge);
        let back = env.inv_mobility_from_ccs(ccs, mz, charge);
        assert!((back - one_over_k0).abs() < 1e-9, "round-trip drift: {back} vs {one_over_k0}");
    }

    #[test]
    fn ion_scalar_derives_inv_mobility_per_env() {
        let warm = MobilityEnv { gas_mass: 28.013, temp_c: 40.0, t_diff: 273.15 };
        let cold = MobilityEnv { gas_mass: 28.013, temp_c: 20.0, t_diff: 273.15 };
        let ion = IonScalar {
            ion_id: 1,
            peptide_id: 1,
            sequence: "PEPTIDEK".to_string(),
            charge: 2,
            relative_abundance: 1.0,
            mz: 500.0,
            ccs: 350.0,
            inv_mobility_std: 0.0,
            simulated_spectrum: MzSpectrum::new(vec![500.0], vec![1.0]),
            condition_id: None,
        };
        // Same CCS yields different 1/K0 in different drift environments — the
        // whole point of storing CCS in the trunk, not 1/K0.
        assert!(ion.inv_mobility(&warm) != ion.inv_mobility(&cold));
    }
}
