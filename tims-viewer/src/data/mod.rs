//! Data layer: metadata, streaming loader, GPU point format, and the synthetic source.

pub mod demo;
pub mod point;
// Native-only: the Bruker `.d` reader (rustdf/SQLite) and its metadata index.
#[cfg(feature = "native")]
pub mod loader;
#[cfg(feature = "native")]
pub mod meta;
// Native-only: the MOBILion `.mbi` reader (HDF5). Opt-in, since it pulls libhdf5.
#[cfg(feature = "mbi")]
pub mod mbi;

/// Build a metadata index for any supported source, dispatching on the path.
///
/// Bruker `.d` folders and MOBILion `.mbi` files carry the same three axes, so the
/// rest of the viewer never needs to know which it is looking at.
#[cfg(feature = "native")]
pub fn load_meta_any(path: &str) -> anyhow::Result<meta::MetaIndex> {
    #[cfg(feature = "mbi")]
    if mbi::is_mbi_path(path) {
        return mbi::load_meta(path);
    }
    meta::MetaIndex::load(path)
}

/// Pick the loader mode matching a run's source format.
#[cfg(feature = "native")]
pub fn loader_mode_for(meta: &meta::MetaIndex, frame_ids: Vec<u32>) -> loader::LoaderMode {
    #[cfg(feature = "mbi")]
    if mbi::is_mbi_path(&meta.data_path) {
        // Carry the FrameInfo through: the MBI loader needs RT and the MS1/MS2 split,
        // which live in per-frame HDF5 attributes the index has already read.
        let wanted: std::collections::HashSet<u32> = frame_ids.iter().copied().collect();
        let frames = meta
            .frames
            .iter()
            .filter(|f| wanted.contains(&f.id))
            .copied()
            .collect();
        return loader::LoaderMode::Mbi { path: meta.data_path.clone(), frames };
    }
    loader::LoaderMode::Real { path: meta.data_path.clone(), frame_ids }
}

/// Whether a source can carry DIA isolation windows at all.
///
/// MOBILion MAF acquisitions are quadrupole-free — there are no isolation windows
/// to draw, and trying to read them would mean opening the path as a Bruker
/// dataset, which it is not.
#[cfg(feature = "native")]
pub fn has_isolation_windows(path: &str) -> bool {
    #[cfg(feature = "mbi")]
    if mbi::is_mbi_path(path) {
        return false;
    }
    let _ = path;
    true
}
