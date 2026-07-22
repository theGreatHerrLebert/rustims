//! Vendor-neutral DIA acquisition **scheme** (input/design side).
//!
//! An [`AcquisitionScheme`] describes one DIA cycle as an *ordered sequence of
//! physical events* ([`AcquisitionEvent`]) and how that cycle tiles the
//! gradient ([`RepeatPolicy`]). It is the design counterpart to the
//! the acquisition writer (output side): the simulator
//! generates scans for the scheme, and a writer materializes them.
//!
//! The key modelling choice (per review) is that a *physical acquisition unit*
//! is explicit: [`AcquisitionEvent::DiaMs2Frame`] carries a `Vec<DiaWindow>`, so
//! a timsTOF MS2 frame holding several mobility-partitioned windows and a linear
//! Astral/SCIEX MS2 scan (one window) are both represented without overloading a
//! `window_group` id to imply simultaneity.

/// Current scheme schema version.
pub const SCHEME_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentKind {
    TimsTofDia,
    OrbitrapAstral,
    SciexZenoTof,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Analyzer {
    Ftms,
    Astms,
    Tof,
    Itms,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataMode {
    Profile,
    Centroid,
}

/// An m/z isolation window (no ion mobility — see [`DiaGeometry`]).
#[derive(Clone, Copy, Debug)]
pub struct IsolationWindow {
    pub center_mz: f64,
    pub width_mz: f64,
}

impl IsolationWindow {
    pub fn lower(&self) -> f64 {
        self.center_mz - self.width_mz / 2.0
    }
    pub fn upper(&self) -> f64 {
        self.center_mz + self.width_mz / 2.0
    }
}

/// How a window's collision energy is determined.
///
/// `Value` covers both a per-window CE and a scheme-wide fixed CE (every window
/// carries the same value) — there is intentionally no separate `Fixed` variant
/// (they were structurally identical). `Unknown` means extraction could not
/// recover it (e.g. SCIEX rolling CE) and a model must be supplied downstream —
/// it is never silently invented.
#[derive(Clone, Copy, Debug)]
pub enum CollisionEnergyPolicy {
    Value(f64),
    Linear { intercept: f64, slope_per_mz: f64 },
    Unknown,
}

impl CollisionEnergyPolicy {
    /// Resolve the CE for a window centered at `center_mz`, if determinable.
    pub fn at(&self, center_mz: f64) -> Option<f64> {
        match *self {
            CollisionEnergyPolicy::Value(v) => Some(v),
            CollisionEnergyPolicy::Linear {
                intercept,
                slope_per_mz,
            } => Some(intercept + slope_per_mz * center_mz),
            CollisionEnergyPolicy::Unknown => None,
        }
    }
}

/// Dissociation method used to activate precursors. Today the simulator only
/// models collisional activation (CID/HCD are treated identically by the
/// timsTOF-trained intensity models); the enum exists so activation type is
/// **persisted and load-bearing** rather than implicit, and so future
/// electron-based methods are representable. `Unknown` = not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationMethod {
    Cid,
    Hcd,
    Etd,
    Ecd,
    Unknown,
}

/// Unit a collision/activation energy is expressed in. A bare number is
/// meaningless across vendors: Bruker/SCIEX report an absolute lab-frame energy
/// (eV), Thermo reports a charge/m-z-normalized collision energy (NCE). Carrying
/// the unit makes conversions explicit and lets a [`crate::sim`] fragment
/// predictor reject inputs it was not calibrated for instead of silently
/// mis-encoding them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyUnit {
    /// Absolute lab-frame collision energy in electron-volts.
    ElectronVolt,
    /// Thermo normalized collision energy (instrument-normalized; requires a
    /// predictor-specific calibration to become a model input).
    NormalizedCe,
    /// Unit not recorded (legacy / unspecified).
    Unknown,
}

/// A fully-typed activation condition: *what* dissociation, at *what* energy, in
/// *what* unit. Produced by the acquisition's activation policy from an event +
/// precursor context; consumed by a fragment predictor, which maps it into its
/// own feature encoding. This is deliberately NOT a bare `f64` — see [`EnergyUnit`].
#[derive(Clone, Copy, Debug)]
pub struct ActivationCondition {
    pub method: ActivationMethod,
    pub value: f64,
    pub unit: EnergyUnit,
}

impl ActivationCondition {
    /// Collisional activation at an absolute eV energy (the Bruker/SCIEX case).
    pub fn collisional_ev(value: f64) -> Self {
        ActivationCondition {
            method: ActivationMethod::Hcd,
            value,
            unit: EnergyUnit::ElectronVolt,
        }
    }

    /// The legacy provenance: timsTOF collisional activation in eV, the implicit
    /// assumption for every DB written before activation type was recorded.
    pub fn legacy_bruker() -> Self {
        Self::collisional_ev(0.0)
    }
}

/// What an instrument is physically capable of, used to gate vendor-specific
/// behaviour WITHOUT using sampling geometry as a proxy. An Astral, for example,
/// has quadrupole isolation (so isolation-window transmission still applies) but
/// no TIMS mobility separation and no mobility-dependent quad isotope
/// transmission — so only the latter two are gated off for it. Defaults are the
/// Bruker timsTOF case (everything present) so existing behaviour is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstrumentCapabilities {
    /// TIMS ion-mobility separation (per-scan mobility filtering of fragments).
    pub has_tims_mobility: bool,
    /// Mobility-dependent quadrupole isotope transmission scaling of fragments.
    pub has_quad_isotope_transmission: bool,
}

impl Default for InstrumentCapabilities {
    fn default() -> Self {
        // Bruker timsTOF: both present — preserves current behaviour.
        Self::bruker_timstof()
    }
}

impl InstrumentCapabilities {
    /// Bruker timsTOF: TIMS mobility + mobility-dependent quad isotope
    /// transmission both present (the default; preserves current behaviour).
    pub fn bruker_timstof() -> Self {
        InstrumentCapabilities {
            has_tims_mobility: true,
            has_quad_isotope_transmission: true,
        }
    }

    /// Orbitrap Astral: quadrupole isolation IS present (so isolation-window
    /// transmission still applies — that is not gated here), but there is NO TIMS
    /// mobility axis and therefore NO mobility-dependent quad isotope transmission.
    /// Both capability flags are false; the isotope-transmission config is forced
    /// to `None` mode for an Astral render (see `IsotopeTransmissionConfig::gated_by`).
    pub fn astral() -> Self {
        InstrumentCapabilities {
            has_tims_mobility: false,
            has_quad_isotope_transmission: false,
        }
    }
}

/// How an acquisition's collision energy is produced, as a vendor-neutral model.
/// Composes the existing per-window [`CollisionEnergyPolicy`] (DIA / fixed CE,
/// already DATA on each window) and adds the Bruker DDA-PASEF mobility-dependent
/// form, so the DDA CE formula is an *instrument* decision instead of being
/// hardcoded in the selection job. A Thermo NCE model becomes another variant
/// (P6) without touching the selection code.
#[derive(Clone, Copy, Debug)]
pub enum CollisionEnergyModel {
    /// CE from a per-window policy (DIA / fixed); evaluated at a window m/z.
    PerWindow(CollisionEnergyPolicy),
    /// Bruker DDA-PASEF: `ce_bias + ce_slope * scan` (scan = Bruker mobility
    /// coordinate). Reproduces the legacy `dda_selection_scheme` formula exactly.
    BrukerPasef { ce_bias: f64, ce_slope: f64 },
}

/// Vendor-neutral activation policy: the typed [`ActivationCondition`] producer
/// for an acquisition. Pairs a dissociation method + energy unit with a
/// [`CollisionEnergyModel`]. The Bruker timsTOF default reproduces the legacy CE
/// numbers byte-for-byte; P6 supplies a Thermo policy with the same interface.
#[derive(Clone, Copy, Debug)]
pub struct ActivationPolicy {
    pub method: ActivationMethod,
    pub unit: EnergyUnit,
    pub model: CollisionEnergyModel,
}

impl ActivationPolicy {
    /// Bruker timsTOF DDA-PASEF: collisional activation (HCD), eV, CE linear in
    /// scan. `ce_bias`/`ce_slope` default (in `dda_selection_scheme`) to
    /// 54.1984 / -0.0345 — pass those to reproduce the legacy output exactly.
    pub fn bruker_pasef(ce_bias: f64, ce_slope: f64) -> Self {
        ActivationPolicy {
            method: ActivationMethod::Hcd,
            unit: EnergyUnit::ElectronVolt,
            model: CollisionEnergyModel::BrukerPasef { ce_bias, ce_slope },
        }
    }

    /// Thermo Orbitrap Astral: collisional activation (HCD) reported as a
    /// **normalized** collision energy (NCE), produced per isolation window. The
    /// per-window NCE is carried in the window's own [`CollisionEnergyPolicy`]
    /// (`ce`), and the unit is [`EnergyUnit::NormalizedCe`] — so a downstream
    /// fragment predictor that was calibrated in eV will *reject* it rather than
    /// silently mis-encode an NCE as an absolute energy. There is no scan
    /// dependence (no IMS): `condition_for_scan` is `None`; use
    /// [`Self::condition_for_window`].
    pub fn thermo_nce(ce: CollisionEnergyPolicy) -> Self {
        ActivationPolicy {
            method: ActivationMethod::Hcd,
            unit: EnergyUnit::NormalizedCe,
            model: CollisionEnergyModel::PerWindow(ce),
        }
    }

    /// Collision energy (eV) the instrument applies at a Bruker mobility scan.
    /// Only meaningful for scan-parameterised models (Bruker DDA-PASEF). A
    /// per-window model has NO scan dependence — it returns `None` (use
    /// [`Self::condition_for_window`]) rather than silently misreading the scan
    /// number as an m/z.
    pub fn collision_energy_for_scan(&self, scan: u32) -> Option<f64> {
        match self.model {
            CollisionEnergyModel::BrukerPasef { ce_bias, ce_slope } => {
                Some(ce_bias + ce_slope * scan as f64)
            }
            CollisionEnergyModel::PerWindow(_) => None,
        }
    }

    /// Typed activation condition at a Bruker mobility scan (scan-parameterised
    /// models only; `None` for per-window models).
    pub fn condition_for_scan(&self, scan: u32) -> Option<ActivationCondition> {
        self.collision_energy_for_scan(scan)
            .map(|value| ActivationCondition {
                method: self.method,
                value,
                unit: self.unit,
            })
    }

    /// Typed activation condition for a per-window model at `center_mz`.
    pub fn condition_for_window(&self, center_mz: f64) -> Option<ActivationCondition> {
        match self.model {
            CollisionEnergyModel::PerWindow(p) => {
                p.at(center_mz).map(|value| ActivationCondition {
                    method: self.method,
                    value,
                    unit: self.unit,
                })
            }
            CollisionEnergyModel::BrukerPasef { .. } => None,
        }
    }
}

/// Window geometry: m/z only, or (timsTOF) an additional ion-mobility partition.
/// Mobility is kept off [`IsolationWindow`] so "mobility on a non-IMS instrument"
/// is unrepresentable. Scan numbers are Bruker-grid coordinates (they require the
/// reference dataset's calibration to map to physical mobility).
#[derive(Clone, Copy, Debug)]
pub enum DiaGeometry {
    MzOnly,
    TimsMobility { scan_start: u32, scan_end: u32 },
}

/// One DIA isolation window within a frame.
#[derive(Clone, Copy, Debug)]
pub struct DiaWindow {
    pub isolation: IsolationWindow,
    pub collision_energy: CollisionEnergyPolicy,
    pub geometry: DiaGeometry,
}

/// An MS1 (precursor) acquisition.
#[derive(Clone, Debug)]
pub struct Ms1Event {
    pub analyzer: Analyzer,
    pub data_mode: DataMode,
    /// The MS1 acquisition m/z range, if known. `None` when not recoverable
    /// (e.g. Thermo `scan_event` does not carry it — it must not be confused
    /// with the DIA window coverage in [`AcquisitionScheme::mz_range`]).
    pub mz_range: Option<(f64, f64)>,
    pub duration_s: Option<f64>,
}

/// One physical MS2 acquisition unit. Holds one window for Astral/SCIEX; several
/// mobility-partitioned windows (sharing the frame) for timsTOF.
#[derive(Clone, Debug)]
pub struct DiaMs2Frame {
    pub windows: Vec<DiaWindow>,
    pub analyzer: Analyzer,
    pub data_mode: DataMode,
    pub duration_s: Option<f64>,
    /// The vendor's native group id for this frame, preserved for round-trip
    /// fidelity (Bruker `WindowGroup`). `None` for vendors without one. The cycle
    /// order is authoritative for *simultaneity/ordering*; this is *identity* only
    /// — needed so a Bruker adapter regenerates the original `WindowGroup` values
    /// that companion tables (`DiaFrameMsMsInfo`) reference.
    pub vendor_group_id: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum AcquisitionEvent {
    Ms1(Ms1Event),
    DiaMs2Frame(DiaMs2Frame),
}

/// How the cycle repeats over the gradient.
#[derive(Clone, Copy, Debug)]
pub enum RepeatPolicy {
    /// The cycle repeats every `cycle_time_s`, starting at `start_time_s`, until
    /// `gradient_length_s`. Per-event `duration_s` (when present) takes
    /// precedence for fine RT placement; otherwise events are spread across the
    /// cycle uniformly.
    FixedCycleTime {
        cycle_time_s: f64,
        gradient_length_s: f64,
        start_time_s: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemeSource {
    ExtractedBruker,
    ExtractedThermo,
    ExtractedSciex,
    UserTable,
    Programmatic,
}

/// Where the scheme came from (scheme-level attribution; not per-field).
#[derive(Clone, Debug)]
pub struct Provenance {
    pub source: SchemeSource,
    pub notes: String,
}

/// A vendor-neutral DIA acquisition design.
#[derive(Clone, Debug)]
pub struct AcquisitionScheme {
    pub version: u16,
    pub instrument: InstrumentKind,
    pub cycle: Vec<AcquisitionEvent>,
    pub repeat: RepeatPolicy,
    pub mz_range: (f64, f64),
    pub provenance: Provenance,
}

impl AcquisitionScheme {
    /// Iterate every DIA window across all MS2 frames in the cycle.
    pub fn windows(&self) -> impl Iterator<Item = &DiaWindow> {
        self.cycle.iter().flat_map(|e| {
            match e {
                AcquisitionEvent::DiaMs2Frame(f) => f.windows.as_slice(),
                AcquisitionEvent::Ms1(_) => &[][..],
            }
            .iter()
        })
    }

    /// Number of MS1 events in one cycle.
    pub fn ms1_count(&self) -> usize {
        self.cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::Ms1(_)))
            .count()
    }

    /// Number of full cycles that fit the gradient (if derivable).
    pub fn num_cycles(&self) -> Option<u64> {
        match self.repeat {
            RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s,
            } if cycle_time_s.is_finite()
                && cycle_time_s > 0.0
                && gradient_length_s.is_finite()
                && start_time_s.is_finite() =>
            {
                Some(((gradient_length_s - start_time_s).max(0.0) / cycle_time_s) as u64)
            }
            _ => None,
        }
    }

    /// Expand a `FixedCycleTime` DIA scheme into a per-frame schedule over the whole
    /// gradient — the synthesized analogue of `thermo_frame_schedule` for schemes that
    /// carry no per-scan timing (e.g. a SCIEX `.wiff` SWATH method). Each cycle emits its
    /// events (MS1 then one MS2 frame per window) at evenly-spaced times within the
    /// cycle. Rows are `(scan, retention_time_s, ms_level, isolation_center_mz,
    /// isolation_width_mz, collision_energy)`; the MS1 row's isolation/CE are `None`, and
    /// a window's CE is resolved from its `CollisionEnergyPolicy` at the window center
    /// (`None` if the policy is `Unknown`). Empty unless `repeat` is `FixedCycleTime`.
    pub fn dia_frame_schedule(&self) -> Vec<(u32, f64, u8, Option<f64>, Option<f64>, Option<f64>)> {
        let RepeatPolicy::FixedCycleTime {
            cycle_time_s,
            start_time_s,
            ..
        } = self.repeat;
        let n_cycles = self.num_cycles().unwrap_or(0);
        let n_events = self.cycle.len();
        if n_cycles == 0 || n_events == 0 {
            return Vec::new();
        }
        let per_event_dt = cycle_time_s / n_events as f64;
        let mut out = Vec::with_capacity(n_cycles as usize * n_events);
        let mut scan: u32 = 1;
        for c in 0..n_cycles {
            let cycle_start = start_time_s + c as f64 * cycle_time_s;
            for (j, ev) in self.cycle.iter().enumerate() {
                let rt = cycle_start + j as f64 * per_event_dt;
                match ev {
                    AcquisitionEvent::Ms1(_) => out.push((scan, rt, 1u8, None, None, None)),
                    AcquisitionEvent::DiaMs2Frame(f) => {
                        // SCIEX from_sciex_wiff builds one window per MS2 frame.
                        if let Some(w) = f.windows.first() {
                            let center = w.isolation.center_mz;
                            let ce = w.collision_energy.at(center);
                            out.push((scan, rt, 2u8, Some(center), Some(w.isolation.width_mz), ce));
                        }
                    }
                }
                scan += 1;
            }
        }
        out
    }

    /// Build a scheme from an explicit window list (injected / CSV). One MS1
    /// followed by one single-window MS2 frame per window.
    pub fn from_window_table(
        instrument: InstrumentKind,
        ms1: Ms1Event,
        windows: Vec<DiaWindow>,
        repeat: RepeatPolicy,
        mz_range: (f64, f64),
    ) -> Self {
        let mut cycle = vec![AcquisitionEvent::Ms1(ms1.clone())];
        for w in windows {
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![w],
                analyzer: ms1.analyzer,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            }));
        }
        AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument,
            cycle,
            repeat,
            mz_range,
            provenance: Provenance {
                source: SchemeSource::UserTable,
                notes: "built from explicit window table".to_string(),
            },
        }
    }

    /// Validate internal consistency. Returns a human-readable error on the first
    /// problem found.
    pub fn validate(&self) -> Result<(), String> {
        if self.version != SCHEME_VERSION {
            return Err(format!(
                "unsupported scheme version {} (this build supports {})",
                self.version, SCHEME_VERSION
            ));
        }
        if self.cycle.is_empty() {
            return Err("empty cycle".into());
        }
        let (lo, hi) = self.mz_range;
        if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
            return Err(format!("invalid mz_range ({lo}, {hi})"));
        }
        // Structure: exactly one MS1, as the first event, then >= 1 MS2 frame.
        if !matches!(self.cycle.first(), Some(AcquisitionEvent::Ms1(_))) {
            return Err("cycle must begin with an MS1 event".into());
        }
        if self.ms1_count() != 1 {
            return Err(format!(
                "cycle must contain exactly one MS1 event (has {})",
                self.ms1_count()
            ));
        }
        if !self
            .cycle
            .iter()
            .any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
        {
            return Err("cycle has no MS2 frames".into());
        }

        let check_dur = |d: Option<f64>, what: &str| -> Result<(), String> {
            match d {
                Some(v) if !(v.is_finite() && v > 0.0) => {
                    Err(format!("{what} duration must be finite and > 0"))
                }
                _ => Ok(()),
            }
        };

        for ev in &self.cycle {
            match ev {
                AcquisitionEvent::Ms1(m) => {
                    check_dur(m.duration_s, "MS1")?;
                    if let Some((a, b)) = m.mz_range {
                        if !(a.is_finite() && b.is_finite()) || a >= b {
                            return Err(format!("invalid MS1 mz_range ({a}, {b})"));
                        }
                    }
                }
                AcquisitionEvent::DiaMs2Frame(frame) => {
                    check_dur(frame.duration_s, "MS2 frame")?;
                    if frame.windows.is_empty() {
                        return Err("MS2 frame has no windows".into());
                    }
                    let multi = frame.windows.len() > 1;
                    if multi && self.instrument != InstrumentKind::TimsTofDia {
                        return Err(
                            "multi-window MS2 frame is only valid for timsTOF (mobility-partitioned)"
                                .into(),
                        );
                    }
                    for (i, w) in frame.windows.iter().enumerate() {
                        if !(w.isolation.width_mz.is_finite() && w.isolation.width_mz > 0.0) {
                            return Err("window width must be finite and > 0".into());
                        }
                        if !w.isolation.center_mz.is_finite()
                            || !w.isolation.lower().is_finite()
                            || !w.isolation.upper().is_finite()
                        {
                            return Err("window m/z is not finite".into());
                        }
                        if w.isolation.lower() < lo - 1e-6 || w.isolation.upper() > hi + 1e-6 {
                            return Err(format!(
                                "window [{:.4}, {:.4}] outside mz_range [{lo:.4}, {hi:.4}]",
                                w.isolation.lower(),
                                w.isolation.upper()
                            ));
                        }
                        match w.collision_energy {
                            CollisionEnergyPolicy::Value(v) => {
                                if !v.is_finite() || v < 0.0 {
                                    return Err("collision energy must be finite and >= 0".into());
                                }
                            }
                            CollisionEnergyPolicy::Linear {
                                intercept,
                                slope_per_mz,
                            } => {
                                if !intercept.is_finite() || !slope_per_mz.is_finite() {
                                    return Err("non-finite linear CE coefficients".into());
                                }
                                match w.collision_energy.at(w.isolation.center_mz) {
                                    Some(ce) if ce.is_finite() && ce >= 0.0 => {}
                                    _ => {
                                        return Err(
                                            "linear CE resolves to non-finite or negative".into()
                                        )
                                    }
                                }
                            }
                            CollisionEnergyPolicy::Unknown => {}
                        }
                        match w.geometry {
                            DiaGeometry::TimsMobility {
                                scan_start,
                                scan_end,
                            } => {
                                if self.instrument != InstrumentKind::TimsTofDia {
                                    return Err(
                                        "mobility geometry is only valid for timsTOF".into()
                                    );
                                }
                                if scan_start > scan_end {
                                    return Err("mobility scan_start > scan_end".into());
                                }
                            }
                            DiaGeometry::MzOnly => {
                                if multi {
                                    return Err(
                                        "multi-window timsTOF frame requires mobility geometry on every window"
                                            .into(),
                                    );
                                }
                            }
                        }
                        // Within one frame, windows must not overlap in BOTH m/z
                        // and mobility (overlap across sequential frames is fine).
                        for w2 in &frame.windows[..i] {
                            let mz_overlap = w.isolation.lower() < w2.isolation.upper()
                                && w2.isolation.lower() < w.isolation.upper();
                            let im_overlap = match (w.geometry, w2.geometry) {
                                (
                                    DiaGeometry::TimsMobility {
                                        scan_start: a0,
                                        scan_end: a1,
                                    },
                                    DiaGeometry::TimsMobility {
                                        scan_start: b0,
                                        scan_end: b1,
                                    },
                                ) => a0 <= b1 && b0 <= a1,
                                _ => true, // m/z-only windows share all mobility
                            };
                            if mz_overlap && im_overlap {
                                return Err(
                                    "windows within one frame overlap in both m/z and mobility"
                                        .into(),
                                );
                            }
                        }
                    }
                }
            }
        }
        match self.repeat {
            RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s,
            } => {
                if !(cycle_time_s.is_finite() && cycle_time_s > 0.0) {
                    return Err("cycle_time_s must be finite and > 0".into());
                }
                if !(gradient_length_s.is_finite() && gradient_length_s > 0.0) {
                    return Err("gradient_length_s must be finite and > 0".into());
                }
                if !start_time_s.is_finite() || start_time_s < 0.0 {
                    return Err("start_time_s must be finite and >= 0".into());
                }
                if start_time_s > gradient_length_s {
                    return Err("start_time_s must be <= gradient_length_s".into());
                }
            }
        }
        Ok(())
    }
}

/// One template scan in the build-from-template frame schedule (P6e). The Astral
/// acquisition mirrors these 1:1 so the trunk's frames align with the template's
/// real scan RTs and windows.
#[derive(Clone, Copy, Debug)]
pub struct TemplateScan {
    pub scan: u32,
    pub retention_time_s: f64,
    pub ms_level: u8,
    /// Present for MS2 (the precursor isolation window; m/z only — no IMS).
    pub isolation: Option<IsolationWindow>,
    /// Present for MS2 (the template's recorded collision energy).
    pub collision_energy: Option<f64>,
}
