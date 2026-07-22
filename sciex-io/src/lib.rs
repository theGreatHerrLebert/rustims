//! SCIEX `.wiff` acquisition-scheme reader, layered on `ms-io`.
//!
//! This is the satellite that keeps SCIEX support working locally + in tests while the published
//! `ms-io` crate stays SCIEX-free. It reads a real ZenoTOF `.wiff` method (via `sciexwiff`) and
//! builds an `ms_io::sim::scheme::AcquisitionScheme` from it — using only ms-io's *public* API, so
//! ms-io never needs to know about SCIEX.
//!
//! It is deliberately NOT published to crates.io (`publish = false`): it carries a private git
//! dependency on `sciexwiff`, which is on legal hold. Because rustims itself is never published,
//! that private dep is legal here. When `sciexwiff` clears, this either publishes as its own crate
//! or folds back into `ms-io` as a feature.

use std::io;
use std::path::Path;

use ms_io::sim::scheme::{
    AcquisitionEvent, AcquisitionScheme, Analyzer, CollisionEnergyPolicy, DataMode, DiaGeometry,
    DiaMs2Frame, DiaWindow, InstrumentKind, IsolationWindow, Ms1Event, Provenance, RepeatPolicy,
    SchemeSource, SCHEME_VERSION,
};

/// Extract the SWATH scheme from a real SCIEX ZenoTOF `.wiff` method.
///
/// Each SWATH window becomes a single-window MS2 frame (no ion mobility) preceded by an MS1.
///
/// **Collision energy** is caller-supplied: SCIEX SWATH uses rolling CE computed by the instrument
/// at acquisition time — it is not stored per-window in the `.wiff` method (the per-window MS2
/// parameter streams are byte-identical across windows). Pass [`CollisionEnergyPolicy::Unknown`] to
/// leave it unset, or a [`CollisionEnergyPolicy::Linear`] rolling model. The `.wiff` method also
/// lacks run timing, so `cycle_time_s` / `gradient_length_s` are caller-supplied. The returned
/// scheme is validated.
pub fn from_sciex_wiff<P: AsRef<Path>>(
    path: P,
    cycle_time_s: f64,
    gradient_length_s: f64,
    collision_energy: CollisionEnergyPolicy,
) -> io::Result<AcquisitionScheme> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Gated: set TIMSIM_SCIEX_WIFF to a real ZenoTOF .wiff (OLE2 method).
    #[test]
    fn from_sciex_wiff_extracts_windows() {
        let wiff = match std::env::var("TIMSIM_SCIEX_WIFF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP from_sciex_wiff_extracts_windows: set TIMSIM_SCIEX_WIFF=<.wiff>");
                return;
            }
        };
        let s = from_sciex_wiff(&wiff, 3.5, 1800.0, CollisionEnergyPolicy::Unknown)
            .expect("extract");
        assert_eq!(s.instrument, InstrumentKind::SciexZenoTof);
        assert!(s.cycle.len() >= 2);
        eprintln!(
            "from_sciex_wiff OK: SciexZenoTof, {} windows, mz {:.1}..{:.1}",
            s.cycle.len() - 1,
            s.mz_range.0,
            s.mz_range.1
        );
    }
}
