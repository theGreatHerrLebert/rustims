//! Vendor-neutral DIA acquisition **scheme** (input/design side).
//!
//! An [`AcquisitionScheme`] describes one DIA cycle as an *ordered sequence of
//! physical events* ([`AcquisitionEvent`]) and how that cycle tiles the
//! gradient ([`RepeatPolicy`]). It is the design counterpart to the
//! [`crate::sim::acquisition::AcquisitionWriter`] (output side): the simulator
//! generates scans for the scheme, and a writer materializes them.
//!
//! The key modelling choice (per review) is that a *physical acquisition unit*
//! is explicit: [`AcquisitionEvent::DiaMs2Frame`] carries a `Vec<DiaWindow>`, so
//! a timsTOF MS2 frame holding several mobility-partitioned windows and a linear
//! Astral/SCIEX MS2 scan (one window) are both represented without overloading a
//! `window_group` id to imply simultaneity.

use std::io;

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
        self.collision_energy_for_scan(scan).map(|value| ActivationCondition {
            method: self.method,
            value,
            unit: self.unit,
        })
    }

    /// Typed activation condition for a per-window model at `center_mz`.
    pub fn condition_for_window(&self, center_mz: f64) -> Option<ActivationCondition> {
        match self.model {
            CollisionEnergyModel::PerWindow(p) => p.at(center_mz).map(|value| ActivationCondition {
                method: self.method,
                value,
                unit: self.unit,
            }),
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
        self.cycle
            .iter()
            .flat_map(|e| match e {
                AcquisitionEvent::DiaMs2Frame(f) => f.windows.as_slice(),
                AcquisitionEvent::Ms1(_) => &[][..],
            }
            .iter())
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
                                    return Err("mobility geometry is only valid for timsTOF".into());
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

impl AcquisitionScheme {
    /// Render the scheme's MS2 windows as Bruker `DiaFrameMsMsWindows` rows — the
    /// backward-compatibility adapter for the existing timsTOF write path.
    ///
    /// Each MS2 frame is one window group; its windows must carry `TimsMobility`
    /// geometry (the Bruker-grid scan ranges). The `WindowGroup` id is the frame's
    /// preserved [`DiaMs2Frame::vendor_group_id`] when present (so a Bruker
    /// round-trip keeps the ids that `DiaFrameMsMsInfo` references), else a 1..N
    /// fallback in cycle order. A `Linear` CE policy is resolved at the window
    /// center (a lossy materialization, not an inverse); `Unknown` CE and
    /// non-timsTOF schemes are rejected. The scheme is validated first.
    ///
    /// Row order is not meaningful (the SQLite table has none); the companion
    /// frame→group table `DiaFrameMsMsInfo` needs the run frame schedule and is a
    /// separate step.
    pub fn to_bruker_windows(&self) -> Result<Vec<crate::data::meta::DiaMsMsWindow>, String> {
        let layout = self.bruker_group_layout()?;
        let mut rows = Vec::new();
        for (ev, slot) in self.cycle.iter().zip(&layout) {
            if let (AcquisitionEvent::DiaMs2Frame(frame), Some(group)) = (ev, slot) {
                for w in &frame.windows {
                    let (scan_num_begin, scan_num_end) = match w.geometry {
                        DiaGeometry::TimsMobility {
                            scan_start,
                            scan_end,
                        } => (scan_start, scan_end),
                        DiaGeometry::MzOnly => {
                            return Err("timsTOF window lacks mobility geometry".into())
                        }
                    };
                    let collision_energy = match w.collision_energy {
                        CollisionEnergyPolicy::Value(v) => v,
                        CollisionEnergyPolicy::Linear { .. } => w
                            .collision_energy
                            .at(w.isolation.center_mz)
                            .ok_or("could not resolve linear CE")?,
                        CollisionEnergyPolicy::Unknown => {
                            return Err("window has unknown collision energy".into())
                        }
                    };
                    rows.push(crate::data::meta::DiaMsMsWindow {
                        window_group: *group,
                        scan_num_begin,
                        scan_num_end,
                        isolation_mz: w.isolation.center_mz,
                        isolation_width: w.isolation.width_mz,
                        collision_energy,
                    });
                }
            }
        }
        Ok(rows)
    }

    /// Per-cycle-position window-group id: `None` for an MS1 event, `Some(group)`
    /// for an MS2 frame. The group id is the frame's preserved `vendor_group_id`
    /// (Bruker `WindowGroup`) when set, else a collision-safe positional id drawn
    /// from the unused set (so preserved and fallback ids never collide).
    /// Duplicate explicit ids are rejected. Validates first; requires a timsTOF
    /// scheme. This is the single source of truth shared by `to_bruker_windows`
    /// and `to_bruker_info`, so the two tables always agree.
    fn bruker_group_layout(&self) -> Result<Vec<Option<u32>>, String> {
        self.validate()?;
        if self.instrument != InstrumentKind::TimsTofDia {
            return Err("Bruker tables require a timsTOF (TimsTofDia) scheme".into());
        }
        use std::collections::HashSet;
        let mut reserved: HashSet<u32> = HashSet::new();
        for ev in &self.cycle {
            if let AcquisitionEvent::DiaMs2Frame(f) = ev {
                if let Some(g) = f.vendor_group_id {
                    if !reserved.insert(g) {
                        return Err(format!("duplicate vendor window-group id {g}"));
                    }
                }
            }
        }
        let mut layout = Vec::with_capacity(self.cycle.len());
        let mut assigned: HashSet<u32> = HashSet::new();
        let mut next_seq: u32 = 1;
        for ev in &self.cycle {
            match ev {
                AcquisitionEvent::Ms1(_) => layout.push(None),
                AcquisitionEvent::DiaMs2Frame(frame) => {
                    let group = match frame.vendor_group_id {
                        Some(g) => g,
                        None => {
                            while reserved.contains(&next_seq) || assigned.contains(&next_seq) {
                                next_seq += 1;
                            }
                            next_seq
                        }
                    };
                    if !assigned.insert(group) {
                        return Err(format!("window-group id {group} collides"));
                    }
                    layout.push(Some(group));
                }
            }
        }
        Ok(layout)
    }

    /// Generate the Bruker `DiaFrameMsMsInfo` (frame → window group) rows for a run
    /// of `num_frames` total frames, tiling the scheme's cycle. Cycle position
    /// `(frame_id - 1) % cycle_len` selects the event (1-based frame ids), MS1
    /// frames produce no row, and each MS2 frame maps to its window-group id (the
    /// same ids `to_bruker_windows` emits, so the two tables reference the same
    /// groups). The final cycle may be partial.
    ///
    /// Precondition: this models a **clean generated run** — frame 1 is the
    /// cycle's leading MS1 and frame ids are contiguous (as TimSim produces). It
    /// is not a reproducer for arbitrary real files with prefix/calibration
    /// frames, gaps, or acquisition starting mid-cycle.
    pub fn to_bruker_info(
        &self,
        num_frames: u32,
    ) -> Result<Vec<crate::data::meta::DiaMsMisInfo>, String> {
        // Bound the work so an absurd frame count errors cleanly instead of OOM.
        const MAX_FRAMES: u32 = 100_000_000;
        if num_frames > MAX_FRAMES {
            return Err(format!("num_frames {num_frames} exceeds the {MAX_FRAMES} limit"));
        }
        let layout = self.bruker_group_layout()?;
        let fpc = layout.len() as u32;
        if fpc == 0 {
            return Err("empty cycle".into());
        }
        let mut rows = Vec::new();
        for frame_id in 1..=num_frames {
            let pos = ((frame_id - 1) % fpc) as usize;
            if let Some(group) = layout[pos] {
                rows.push(crate::data::meta::DiaMsMisInfo {
                    frame_id,
                    window_group: group,
                });
            }
        }
        Ok(rows)
    }

    /// Both Bruker DIA tables for a `num_frames`-frame run: the
    /// `DiaFrameMsMsWindows` rows and the `DiaFrameMsMsInfo` (frame→group) rows,
    /// with consistent window-group ids.
    pub fn to_bruker_tables(
        &self,
        num_frames: u32,
    ) -> Result<
        (
            Vec<crate::data::meta::DiaMsMsWindow>,
            Vec<crate::data::meta::DiaMsMisInfo>,
        ),
        String,
    > {
        Ok((self.to_bruker_windows()?, self.to_bruker_info(num_frames)?))
    }

    /// Extract the acquisition scheme from a real Bruker timsTOF DIA `.d`.
    ///
    /// Reads `DiaFrameMsMsWindows` and the frame table. Each **window group**
    /// becomes one [`DiaMs2Frame`] holding its mobility-partitioned windows
    /// (`TimsMobility` geometry — the scan ranges are Bruker-grid coordinates),
    /// preceded by a single MS1 event. Cycle timing comes from the spacing of
    /// the precursor (MS1) frames. The returned scheme is validated.
    ///
    /// Window groups (frames) are ordered by their **first occurrence in
    /// `DiaFrameMsMsInfo`** (ascending frame id), i.e. the real acquisition order
    /// within a cycle — not merely ascending `WindowGroup` id — so the cycle
    /// faithfully represents permuted/reused group numbering. (If the info table
    /// is absent, falls back to ascending group id.)
    pub fn from_bruker_d<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        use crate::data::meta::{read_dia_ms_ms_info, read_dia_ms_ms_windows, read_meta_data_sql};
        use std::collections::BTreeMap;

        let folder = path.as_ref().to_string_lossy().into_owned();
        let to_io =
            |e: Box<dyn std::error::Error>| io::Error::new(io::ErrorKind::InvalidData, e.to_string());
        let windows = read_dia_ms_ms_windows(&folder).map_err(to_io)?;
        let frames = read_meta_data_sql(&folder).map_err(to_io)?;
        if windows.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no DiaFrameMsMsWindows rows (not a DIA .d?)",
            ));
        }

        // Group windows by window group (ascending); each group is a frame.
        let mut by_group: BTreeMap<u32, Vec<DiaWindow>> = BTreeMap::new();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for w in &windows {
            let dw = DiaWindow {
                isolation: IsolationWindow {
                    center_mz: w.isolation_mz,
                    width_mz: w.isolation_width,
                },
                collision_energy: CollisionEnergyPolicy::Value(w.collision_energy),
                geometry: DiaGeometry::TimsMobility {
                    scan_start: w.scan_num_begin,
                    scan_end: w.scan_num_end,
                },
            };
            lo = lo.min(dw.isolation.lower());
            hi = hi.max(dw.isolation.upper());
            by_group.entry(w.window_group).or_default().push(dw);
        }

        // Order groups by first occurrence (ascending frame id) in DiaFrameMsMsInfo
        // = real intra-cycle acquisition order. Missing info -> ascending group id.
        let info = read_dia_ms_ms_info(&folder).unwrap_or_default();
        let mut first_frame: BTreeMap<u32, u32> = BTreeMap::new();
        for r in &info {
            first_frame
                .entry(r.window_group)
                .and_modify(|f| {
                    if r.frame_id < *f {
                        *f = r.frame_id
                    }
                })
                .or_insert(r.frame_id);
        }
        let mut ordered_groups: Vec<u32> = by_group.keys().copied().collect();
        // Stable sort by first-occurrence frame; groups absent from the info table
        // (key u32::MAX) keep their ascending-id relative order.
        ordered_groups.sort_by_key(|g| first_frame.get(g).copied().unwrap_or(u32::MAX));

        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        })];
        let n_groups = ordered_groups.len();
        for group in ordered_groups {
            let ws = by_group.remove(&group).expect("group present");
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: Some(group), // preserve Bruker WindowGroup identity
            }));
        }

        // Cycle timing from the precursor (MS1, MsMsType == 0) frame spacing.
        // Use the MEDIAN of the positive gaps between distinct, finite precursor
        // times, so one anomalous interval doesn't set the whole cycle.
        let mut prec_times: Vec<f64> = frames
            .iter()
            .filter(|f| f.ms_ms_type == 0 && f.time.is_finite())
            .map(|f| f.time)
            .collect();
        prec_times.sort_by(f64::total_cmp);
        prec_times.dedup();
        let mut gaps: Vec<f64> = prec_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|g| *g > 0.0)
            .collect();
        if gaps.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "could not determine cycle time (need >= 2 distinct precursor frames)",
            ));
        }
        gaps.sort_by(f64::total_cmp);
        let cycle_time_s = gaps[gaps.len() / 2];

        let times: Vec<f64> = frames.iter().map(|f| f.time).filter(|t| t.is_finite()).collect();
        let start_time_s = times.iter().copied().fold(f64::INFINITY, f64::min);
        let gradient_length_s = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let scheme = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: if start_time_s.is_finite() {
                    start_time_s
                } else {
                    0.0
                },
            },
            mz_range: (lo, hi),
            provenance: Provenance {
                source: SchemeSource::ExtractedBruker,
                notes: format!("extracted from Bruker .d ({n_groups} window groups)"),
            },
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
    }
}

#[cfg(feature = "sciex")]
impl AcquisitionScheme {
    /// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
    ///
    /// Each SWATH window becomes a single-window [`DiaMs2Frame`] (no ion
    /// mobility) preceded by an MS1.
    ///
    /// **Collision energy** is caller-supplied (`collision_energy`), because
    /// SCIEX SWATH uses *rolling* CE computed by the instrument from an m/z
    /// formula at acquisition time — it is **not** stored per-window in the
    /// `.wiff` method (verified: the per-window MS2 parameter streams are
    /// byte-identical across windows). Pass [`CollisionEnergyPolicy::Unknown`]
    /// to leave it unset, or a [`CollisionEnergyPolicy::Linear`] rolling model.
    /// The `.wiff` method also lacks run timing, so `cycle_time_s` /
    /// `gradient_length_s` are caller-supplied. The returned scheme is validated.
    pub fn from_sciex_wiff<P: AsRef<std::path::Path>>(
        path: P,
        cycle_time_s: f64,
        gradient_length_s: f64,
        collision_energy: CollisionEnergyPolicy,
    ) -> io::Result<Self> {
        let method = sciexwiff::read_method(path)?;
        if method.swath_windows.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no SWATH windows in the .wiff method",
            ));
        }
        let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            mz_range: None,
            duration_s: None,
        })];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let n = method.swath_windows.len();
        for w in &method.swath_windows {
            let iso = IsolationWindow {
                center_mz: w.center_mz(),
                width_mz: w.width_mz(),
            };
            lo = lo.min(iso.lower());
            hi = hi.max(iso.upper());
            cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: iso,
                    collision_energy,
                    geometry: DiaGeometry::MzOnly,
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            }));
        }
        let scheme = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::SciexZenoTof,
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: 0.0,
            },
            mz_range: (lo, hi),
            provenance: Provenance {
                source: SchemeSource::ExtractedSciex,
                notes: format!(
                    "extracted from SCIEX .wiff method ({n} SWATH windows; CE caller-supplied, timing caller-supplied)"
                ),
            },
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
    }
}

#[cfg(feature = "thermo")]
impl AcquisitionScheme {
    /// Extract the acquisition scheme from a real Thermo `.raw` by walking the
    /// first complete cycle (first MS1 up to the next MS1). Each MS2 scan becomes
    /// a single-window [`DiaMs2Frame`] (Thermo has no mobility), with the
    /// observed isolation center/width and CE.
    pub fn from_thermo_raw<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        use thermorawfile::RawFile;
        let raw = RawFile::open(path)?;

        let analyzer_of = |a: u8| match a {
            4 => Analyzer::Ftms,
            7 => Analyzer::Astms,
            0 => Analyzer::Itms,
            _ => Analyzer::Unknown,
        };
        let data_mode_of = |scan: u32| -> DataMode {
            let i = (scan - raw.first_scan) as usize;
            let pkt = (raw.data_addr + raw.index[i].offset) as usize;
            if pkt + 8 <= raw.bytes.len() {
                let ps = u32::from_le_bytes(raw.bytes[pkt + 4..pkt + 8].try_into().unwrap());
                if ps > 0 {
                    DataMode::Profile
                } else {
                    DataMode::Centroid
                }
            } else {
                DataMode::Centroid
            }
        };
        let rt_of = |scan: u32| raw.index[(scan - raw.first_scan) as usize].time;

        let mut cycle: Vec<AcquisitionEvent> = Vec::new();
        let mut seen_ms1 = false;
        let mut closed = false;
        let mut start_rt = 0.0f64;
        let mut cycle_time_s = 0.0f64;
        let mut win_lo = f64::INFINITY;
        let mut win_hi = f64::NEG_INFINITY;

        for scan in raw.first_scan..=raw.last_scan {
            let ev = match raw.scan_event(scan) {
                Some(e) => e,
                None => {
                    // A missing event *inside* the selected cycle would silently
                    // drop a window — reject it. Before the first MS1 it's ignorable.
                    if seen_ms1 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("missing scan event for scan {scan} inside the first cycle"),
                        ));
                    }
                    continue;
                }
            };
            if ev.ms_order <= 1 {
                if seen_ms1 {
                    // start of the next cycle -> close this one
                    cycle_time_s = (rt_of(scan) - start_rt).max(0.0);
                    closed = true;
                    break;
                }
                seen_ms1 = true;
                start_rt = rt_of(scan);
                cycle.push(AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: analyzer_of(ev.analyzer),
                    data_mode: data_mode_of(scan),
                    mz_range: None, // scan_event does not carry the MS1 range
                    duration_s: None,
                }));
            } else if seen_ms1 {
                let iso = IsolationWindow {
                    center_mz: ev.isolation_center,
                    width_mz: ev.isolation_width,
                };
                win_lo = win_lo.min(iso.lower());
                win_hi = win_hi.max(iso.upper());
                cycle.push(AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                    windows: vec![DiaWindow {
                        isolation: iso,
                        collision_energy: CollisionEnergyPolicy::Value(ev.collision_energy),
                        geometry: DiaGeometry::MzOnly,
                    }],
                    analyzer: analyzer_of(ev.analyzer),
                    data_mode: data_mode_of(scan),
                    duration_s: None,
                    vendor_group_id: None,
                }));
            }
        }

        if !seen_ms1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template has no MS1 scan",
            ));
        }
        if !closed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "template has only one cycle (no second MS1 to bound it); cannot determine cycle time",
            ));
        }
        let mz_range = if win_lo.is_finite() && win_hi > win_lo {
            (win_lo, win_hi)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "first cycle has no usable MS2 isolation windows",
            ));
        };
        let gradient_length_s = raw.index.last().map(|e| e.time).unwrap_or(0.0);
        let n_ms2 = cycle
            .iter()
            .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
            .count();
        // Instrument from the analyzers of events actually in the cycle (an ASTMS
        // MS2 frame ⇒ Orbitrap Astral), not from incidental scans elsewhere.
        let any_astms = cycle.iter().any(|e| {
            matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.analyzer == Analyzer::Astms)
        });

        let scheme = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: if any_astms {
                InstrumentKind::OrbitrapAstral
            } else {
                InstrumentKind::Other
            },
            cycle,
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s,
                gradient_length_s,
                start_time_s: start_rt,
            },
            mz_range,
            provenance: Provenance {
                source: SchemeSource::ExtractedThermo,
                notes: format!("extracted from Thermo .raw (first cycle: 1 MS1 + {n_ms2} MS2)"),
            },
        };
        scheme
            .validate()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(scheme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_condition_carries_typed_unit() {
        let c = ActivationCondition::collisional_ev(30.8);
        assert_eq!(c.method, ActivationMethod::Hcd);
        assert_eq!(c.unit, EnergyUnit::ElectronVolt);
        assert_eq!(c.value, 30.8);

        // The legacy provenance is collisional eV (timsTOF assumption).
        let legacy = ActivationCondition::legacy_bruker();
        assert_eq!(legacy.unit, EnergyUnit::ElectronVolt);

        // An NCE value is a distinct unit and must not compare equal to eV.
        let nce = ActivationCondition { method: ActivationMethod::Hcd, value: 30.0, unit: EnergyUnit::NormalizedCe };
        assert_ne!(nce.unit, c.unit);
    }

    #[test]
    fn bruker_pasef_activation_policy_reproduces_legacy_ce() {
        // Legacy dda_selection_scheme: collision_energy = ce_bias + ce_slope*scan.
        let (ce_bias, ce_slope) = (54.1984, -0.0345);
        let p = ActivationPolicy::bruker_pasef(ce_bias, ce_slope);
        assert_eq!(p.method, ActivationMethod::Hcd);
        assert_eq!(p.unit, EnergyUnit::ElectronVolt);
        for scan in [0u32, 1, 250, 451, 917] {
            assert_eq!(
                p.collision_energy_for_scan(scan),
                Some(ce_bias + ce_slope * scan as f64),
                "CE must match the legacy formula at scan {scan}"
            );
            assert_eq!(p.condition_for_scan(scan).unwrap().value, ce_bias + ce_slope * scan as f64);
        }
        // A per-window (DIA) model has no scan dependence: scan evaluation is
        // None (not a silently-wrong value), and CE comes from the window m/z.
        let w = ActivationPolicy {
            method: ActivationMethod::Hcd,
            unit: EnergyUnit::ElectronVolt,
            model: CollisionEnergyModel::PerWindow(CollisionEnergyPolicy::Value(25.0)),
        };
        assert_eq!(w.condition_for_window(700.0).unwrap().value, 25.0);
        assert_eq!(w.collision_energy_for_scan(100), None);
        assert!(w.condition_for_scan(100).is_none());
    }

    #[test]
    fn instrument_capabilities_default_is_bruker() {
        // Default = full timsTOF capability so existing behaviour is unchanged;
        // an Astral-like instrument keeps quad isolation but drops mobility +
        // mobility-dependent isotope transmission.
        let bruker = InstrumentCapabilities::default();
        assert!(bruker.has_tims_mobility);
        assert!(bruker.has_quad_isotope_transmission);
        assert_eq!(bruker, InstrumentCapabilities::bruker_timstof());

        // P6c: the named Astral constructor is both-false (isotope mode gated off).
        let astral = InstrumentCapabilities::astral();
        assert!(!astral.has_tims_mobility);
        assert!(!astral.has_quad_isotope_transmission);
        assert_ne!(bruker, astral);
    }

    #[test]
    fn thermo_nce_activation_policy_is_normalized_per_window() {
        // P6c: Astral NCE policy. Per-window normalized CE; no scan dependence.
        let p = ActivationPolicy::thermo_nce(CollisionEnergyPolicy::Value(27.0));
        assert_eq!(p.method, ActivationMethod::Hcd);
        assert_eq!(p.unit, EnergyUnit::NormalizedCe);

        // Resolves a window CE as an NCE-unit condition (NOT eV) — a downstream
        // eV-calibrated predictor must be able to reject it on the unit alone.
        let cond = p.condition_for_window(650.0).expect("NCE condition for window");
        assert_eq!(cond.value, 27.0);
        assert_eq!(cond.unit, EnergyUnit::NormalizedCe);
        assert_ne!(cond.unit, EnergyUnit::ElectronVolt);

        // No IMS: there is no scan-parameterised CE.
        assert_eq!(p.collision_energy_for_scan(100), None);
        assert!(p.condition_for_scan(100).is_none());

        // A rolling (linear) NCE model resolves per window center.
        let rolling = ActivationPolicy::thermo_nce(CollisionEnergyPolicy::Linear {
            intercept: 20.0,
            slope_per_mz: 0.01,
        });
        assert_eq!(rolling.condition_for_window(700.0).unwrap().value, 20.0 + 0.01 * 700.0);
        assert_eq!(rolling.condition_for_window(700.0).unwrap().unit, EnergyUnit::NormalizedCe);
    }

    fn linear_windows(n: usize) -> Vec<DiaWindow> {
        (0..n)
            .map(|i| DiaWindow {
                isolation: IsolationWindow {
                    center_mz: 400.0 + i as f64 * 10.0,
                    width_mz: 10.0,
                },
                collision_energy: CollisionEnergyPolicy::Value(25.0),
                geometry: DiaGeometry::MzOnly,
            })
            .collect()
    }

    #[test]
    fn from_window_table_validates() {
        let s = AcquisitionScheme::from_window_table(
            InstrumentKind::OrbitrapAstral,
            Ms1Event {
                analyzer: Analyzer::Ftms,
                data_mode: DataMode::Profile,
                mz_range: Some((390.0, 900.0)),
                duration_s: None,
            },
            linear_windows(5),
            RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            (390.0, 900.0),
        );
        s.validate().unwrap();
        assert_eq!(s.windows().count(), 5);
        assert_eq!(s.ms1_count(), 1);
        assert_eq!(s.num_cycles(), Some(600));
    }

    #[test]
    fn multi_window_frame_only_tims() {
        let frame = DiaMs2Frame {
            windows: linear_windows(3),
            analyzer: Analyzer::Tof,
            data_mode: DataMode::Centroid,
            duration_s: None,
            vendor_group_id: None,
        };
        let s = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::OrbitrapAstral, // wrong: multi-window on non-tims
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Ftms,
                    data_mode: DataMode::Profile,
                    mz_range: Some((390.0, 900.0)),
                    duration_s: None,
                }),
                AcquisitionEvent::DiaMs2Frame(frame),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (390.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_schemes() {
        let repeat = RepeatPolicy::FixedCycleTime {
            cycle_time_s: 1.0,
            gradient_length_s: 600.0,
            start_time_s: 0.0,
        };
        let ms1 = || Ms1Event {
            analyzer: Analyzer::Ftms,
            data_mode: DataMode::Profile,
            mz_range: Some((390.0, 900.0)),
            duration_s: None,
        };
        let win = || DiaWindow {
            isolation: IsolationWindow {
                center_mz: 500.0,
                width_mz: 10.0,
            },
            collision_energy: CollisionEnergyPolicy::Value(25.0),
            geometry: DiaGeometry::MzOnly,
        };
        let frame = |w: DiaWindow| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![w],
                analyzer: Analyzer::Astms,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            })
        };
        let mk = |cycle: Vec<AcquisitionEvent>| AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::OrbitrapAstral,
            cycle,
            repeat,
            mz_range: (390.0, 900.0),
            provenance: Provenance {
                source: SchemeSource::Programmatic,
                notes: String::new(),
            },
        };

        // valid baseline
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_ok());
        // MS2 first (no leading MS1)
        assert!(mk(vec![frame(win()), AcquisitionEvent::Ms1(ms1())]).validate().is_err());
        // two MS1 in a cycle
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), AcquisitionEvent::Ms1(ms1()), frame(win())]).validate().is_err());
        // MS1 with no MS2 frame
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1())]).validate().is_err());
        // window outside mz_range
        let mut oob = win();
        oob.isolation.center_mz = 2000.0;
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(oob)]).validate().is_err());
        // negative collision energy
        let mut neg = win();
        neg.collision_energy = CollisionEnergyPolicy::Value(-5.0);
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), frame(neg)]).validate().is_err());
        // bad event duration
        let bad_dur = AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
            windows: vec![win()],
            analyzer: Analyzer::Astms,
            data_mode: DataMode::Centroid,
            duration_s: Some(-1.0),
            vendor_group_id: None,
        });
        assert!(mk(vec![AcquisitionEvent::Ms1(ms1()), bad_dur]).validate().is_err());
    }

    // Gated: set TIMSIM_BRUKER_DIA_D to a real Bruker DIA-PASEF .d folder.
    #[test]
    fn from_bruker_d_extracts_cycle() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_bruker_d_extracts_cycle: set TIMSIM_BRUKER_DIA_D=<dia .d>");
                return;
            }
        };
        let s = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::TimsTofDia);
        assert_eq!(s.ms1_count(), 1);
        let n_win = s.windows().count();
        assert!(n_win > 1, "expected several windows, got {n_win}");
        // timsTOF windows carry mobility geometry with a known CE.
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::TimsMobility { .. }));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_some());
        }
        // At least one frame should be mobility-partitioned (>1 window).
        let multi = s.cycle.iter().any(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(f) if f.windows.len() > 1));
        let n_frames = s.cycle.iter().filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_))).count();
        eprintln!(
            "from_bruker_d OK: TimsTofDia, {} frames, {} windows ({}), mz {:.1}..{:.1}",
            n_frames, n_win, if multi { "mobility-partitioned" } else { "single-window" },
            s.mz_range.0, s.mz_range.1
        );
    }

    // Gated: set TIMSIM_SCIEX_WIFF to a real ZenoTOF .wiff (OLE2 method).
    // Unconditional: to_bruker_windows must preserve explicit vendor_group_ids
    // (incl. non-canonical), allocate non-colliding ids for None frames, and
    // reject duplicate explicit ids. (The gated round-trip uses a real .d whose
    // ids are already 1..N, so it can't prove non-canonical preservation alone.)
    #[test]
    fn to_bruker_windows_group_ids() {
        let frame = |gid: Option<u32>, center: f64| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: IsolationWindow { center_mz: center, width_mz: 10.0 },
                    collision_energy: CollisionEnergyPolicy::Value(25.0),
                    geometry: DiaGeometry::TimsMobility { scan_start: 0, scan_end: 100 },
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: gid,
            })
        };
        let mk = |frames: Vec<AcquisitionEvent>| {
            let mut cycle = vec![AcquisitionEvent::Ms1(Ms1Event {
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                mz_range: None,
                duration_s: None,
            })];
            cycle.extend(frames);
            AcquisitionScheme {
                version: SCHEME_VERSION,
                instrument: InstrumentKind::TimsTofDia,
                cycle,
                repeat: RepeatPolicy::FixedCycleTime {
                    cycle_time_s: 1.0,
                    gradient_length_s: 600.0,
                    start_time_s: 0.0,
                },
                mz_range: (300.0, 900.0),
                provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
            }
        };
        let ids = |s: &AcquisitionScheme| {
            s.to_bruker_windows().map(|r| r.iter().map(|w| w.window_group).collect::<Vec<_>>())
        };

        // preserved non-canonical ids
        assert_eq!(ids(&mk(vec![frame(Some(7), 500.0), frame(Some(42), 600.0)])).unwrap(), vec![7, 42]);
        // all-None -> sequential 1..N
        assert_eq!(ids(&mk(vec![frame(None, 500.0), frame(None, 600.0)])).unwrap(), vec![1, 2]);
        // mixed: None allocates a free id (1), avoiding the reserved 7
        assert_eq!(ids(&mk(vec![frame(Some(7), 500.0), frame(None, 600.0)])).unwrap(), vec![7, 1]);
        // duplicate explicit ids rejected
        assert!(ids(&mk(vec![frame(Some(5), 500.0), frame(Some(5), 600.0)])).is_err());
    }

    // Gated: round-trip a real Bruker DIA .d through the scheme and back to the
    // DiaFrameMsMsWindows rows; the regenerated table must match the source.
    #[test]
    fn bruker_windows_round_trip() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_windows_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let regenerated = scheme.to_bruker_windows().expect("to_bruker_windows");
        let original = crate::data::meta::read_dia_ms_ms_windows(&d).expect("read source");
        assert_eq!(regenerated.len(), original.len(), "row count differs");

        // Compare as a sorted multiset of EXACT field tuples, including the
        // WindowGroup id (preserved via vendor_group_id) — no normalization, so a
        // real id mismatch would fail. Row order in the table is not guaranteed.
        fn tuples(
            rows: &[crate::data::meta::DiaMsMsWindow],
        ) -> Vec<(u32, u32, u32, u64, u64, u64)> {
            let mut out: Vec<_> = rows
                .iter()
                .map(|r| {
                    (
                        r.window_group,
                        r.scan_num_begin,
                        r.scan_num_end,
                        r.isolation_mz.to_bits(),
                        r.isolation_width.to_bits(),
                        r.collision_energy.to_bits(),
                    )
                })
                .collect();
            out.sort_unstable();
            out
        }
        assert_eq!(
            tuples(&regenerated),
            tuples(&original),
            "round-trip windows differ from source (incl. WindowGroup id)"
        );
        eprintln!(
            "bruker_windows_round_trip OK: {} rows match across {} groups",
            original.len(),
            scheme
                .cycle
                .iter()
                .filter(|e| matches!(e, AcquisitionEvent::DiaMs2Frame(_)))
                .count()
        );
    }

    // Unconditional: DiaFrameMsMsInfo tiling + windows/info group-id consistency.
    #[test]
    fn to_bruker_info_tiling() {
        let frame = |c: f64| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: vec![DiaWindow {
                    isolation: IsolationWindow { center_mz: c, width_mz: 10.0 },
                    collision_energy: CollisionEnergyPolicy::Value(25.0),
                    geometry: DiaGeometry::TimsMobility { scan_start: 0, scan_end: 100 },
                }],
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: None,
            })
        };
        let s = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Tof,
                    data_mode: DataMode::Centroid,
                    mz_range: None,
                    duration_s: None,
                }),
                frame(500.0),
                frame(600.0),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (300.0, 900.0),
            provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
        };
        // cycle = [MS1, g1, g2]; num_frames=6: 1->skip, 2->g1, 3->g2, 4->skip, 5->g1, 6->g2
        let info: Vec<(u32, u32)> = s
            .to_bruker_info(6)
            .unwrap()
            .iter()
            .map(|r| (r.frame_id, r.window_group))
            .collect();
        assert_eq!(info, vec![(2, 1), (3, 2), (5, 1), (6, 2)]);
        // The two tables must reference the same set of window-group ids.
        let (windows, info2) = s.to_bruker_tables(6).unwrap();
        let wg: std::collections::BTreeSet<u32> = windows.iter().map(|w| w.window_group).collect();
        let ig: std::collections::BTreeSet<u32> = info2.iter().map(|r| r.window_group).collect();
        assert_eq!(wg, ig, "windows/info group ids disagree");

        // Non-ascending explicit ids + a multi-window (mobility-partitioned) frame:
        // info must follow CYCLE order (not sorted id) and emit ONE row per frame.
        let win = |center: f64, s0: u32, s1: u32| DiaWindow {
            isolation: IsolationWindow { center_mz: center, width_mz: 10.0 },
            collision_energy: CollisionEnergyPolicy::Value(20.0),
            geometry: DiaGeometry::TimsMobility { scan_start: s0, scan_end: s1 },
        };
        let dia = |gid: u32, ws: Vec<DiaWindow>| {
            AcquisitionEvent::DiaMs2Frame(DiaMs2Frame {
                windows: ws,
                analyzer: Analyzer::Tof,
                data_mode: DataMode::Centroid,
                duration_s: None,
                vendor_group_id: Some(gid),
            })
        };
        let s2 = AcquisitionScheme {
            version: SCHEME_VERSION,
            instrument: InstrumentKind::TimsTofDia,
            cycle: vec![
                AcquisitionEvent::Ms1(Ms1Event {
                    analyzer: Analyzer::Tof,
                    data_mode: DataMode::Centroid,
                    mz_range: None,
                    duration_s: None,
                }),
                dia(7, vec![win(500.0, 0, 50), win(500.0, 51, 100)]), // 2 mobility windows
                dia(2, vec![win(600.0, 0, 100)]),
            ],
            repeat: RepeatPolicy::FixedCycleTime {
                cycle_time_s: 1.0,
                gradient_length_s: 600.0,
                start_time_s: 0.0,
            },
            mz_range: (300.0, 900.0),
            provenance: Provenance { source: SchemeSource::Programmatic, notes: String::new() },
        };
        // cycle len 3; frame2->g7 (first), frame3->g2 — acquisition order, NOT sorted.
        let info3: Vec<(u32, u32)> = s2
            .to_bruker_info(3)
            .unwrap()
            .iter()
            .map(|r| (r.frame_id, r.window_group))
            .collect();
        assert_eq!(info3, vec![(2, 7), (3, 2)], "info must follow cycle order, one row/frame");
        // windows: the 2-window frame emits 2 rows (group 7), the other 1 (group 2).
        let w3 = s2.to_bruker_windows().unwrap();
        assert_eq!(w3.iter().filter(|w| w.window_group == 7).count(), 2);
        assert_eq!(w3.iter().filter(|w| w.window_group == 2).count(), 1);
    }

    // Gated: regenerate DiaFrameMsMsInfo for the .d's full frame count and match.
    #[test]
    fn bruker_info_round_trip() {
        let d = match std::env::var("TIMSIM_BRUKER_DIA_D") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP bruker_info_round_trip: set TIMSIM_BRUKER_DIA_D");
                return;
            }
        };
        let scheme = AcquisitionScheme::from_bruker_d(&d).expect("extract");
        let frames = crate::data::meta::read_meta_data_sql(&d).expect("frames");
        let num_frames = frames.iter().map(|f| f.id).max().unwrap_or(0) as u32;
        let regenerated = scheme.to_bruker_info(num_frames).expect("to_bruker_info");
        let original = crate::data::meta::read_dia_ms_ms_info(&d).expect("read source");
        let key = |r: &crate::data::meta::DiaMsMisInfo| (r.frame_id, r.window_group);
        let mut a: Vec<_> = regenerated.iter().map(key).collect();
        let mut b: Vec<_> = original.iter().map(key).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "DiaFrameMsMsInfo round-trip differs from source");
        eprintln!("bruker_info_round_trip OK: {} (frame, group) rows over {num_frames} frames", a.len());
    }

    #[cfg(feature = "sciex")]
    #[test]
    fn from_sciex_wiff_extracts_windows() {
        let wiff = match std::env::var("TIMSIM_SCIEX_WIFF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
                return;
            }
        };
        // Default: CE unknown (rolling CE isn't in the .wiff method).
        let s = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
            .expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::SciexZenoTof);
        assert_eq!(s.ms1_count(), 1);
        let n = s.windows().count();
        assert!(n > 10, "expected many SWATH windows, got {n}");
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::MzOnly));
            assert!(matches!(w.collision_energy, CollisionEnergyPolicy::Unknown));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_none());
        }
        // Supplying a rolling-CE Linear model gives a resolvable, finite CE.
        let rolling = CollisionEnergyPolicy::Linear { intercept: 5.0, slope_per_mz: 0.045 };
        let s2 = AcquisitionScheme::from_sciex_wiff(&wiff, 3.5, 1800.0, rolling).expect("extract");
        s2.validate().expect("valid with rolling CE");
        for w in s2.windows() {
            let ce = w.collision_energy.at(w.isolation.center_mz).expect("resolvable CE");
            assert!(ce.is_finite() && ce > 0.0);
        }
        eprintln!(
            "from_sciex_wiff OK: SciexZenoTof, {} SWATH windows, mz {:.1}..{:.1}",
            n, s.mz_range.0, s.mz_range.1
        );
    }

    #[cfg(feature = "thermo")]
    #[test]
    fn from_thermo_raw_extracts_cycle() {
        let template = match std::env::var("TIMSIM_ASTRAL_TEMPLATE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_thermo_raw_extracts_cycle: set TIMSIM_ASTRAL_TEMPLATE");
                return;
            }
        };
        let s = AcquisitionScheme::from_thermo_raw(&template).expect("extract");
        s.validate().expect("valid scheme");
        assert_eq!(s.instrument, InstrumentKind::OrbitrapAstral);
        assert_eq!(s.ms1_count(), 1, "one MS1 per cycle");
        let n_win = s.windows().count();
        assert!(n_win > 1, "expected several MS2 windows, got {n_win}");
        // Thermo: every window is m/z-only with a known CE.
        for w in s.windows() {
            assert!(matches!(w.geometry, DiaGeometry::MzOnly));
            assert!(w.collision_energy.at(w.isolation.center_mz).is_some());
            assert!(w.isolation.width_mz > 0.0);
        }
        // centers should be (weakly) increasing across the first cycle is not
        // guaranteed by Astral, but coverage must be a sane m/z span.
        assert!(s.mz_range.0 > 100.0 && s.mz_range.1 < 3000.0 && s.mz_range.0 < s.mz_range.1);
        eprintln!(
            "from_thermo_raw OK: instrument={:?}, {} MS2 windows, mz {:.1}..{:.1}, cycle {:.4}s",
            s.instrument, n_win, s.mz_range.0, s.mz_range.1,
            match s.repeat { RepeatPolicy::FixedCycleTime { cycle_time_s, .. } => cycle_time_s }
        );
    }
}
