//! A **reference-free**, Rust-native writer for a minimal Bruker timsTOF `.d` directory.
//!
//! v1's `.d` authoring lives in Python (`imspy_simulation.tdf.TDFWriter`) and *copies* every
//! calibration / metadata table from a real reference `.d` — so you need a genuine Bruker file to
//! produce a synthetic one. This writer synthesises those tables instead, which is the right shape for
//! timsim v2: an instrument-independent feature space renders straight to a `.d` with no reference.
//!
//! Scope of this first milestone: **MS1-only** (`MsMsType = 0`). It writes exactly what the rustims
//! reader ([`crate::data::handle::TimsRawDataLayout`] + `TimsLazyLoder`) needs to open the file and
//! read raw `(scan, tof, intensity)` frames back — that round-trip is the oracle (see the test at the
//! bottom). Physical calibration accuracy (the coefficients a vendor reader like DiaNN needs to derive
//! m/z and 1/K0) is deliberately **not** claimed here; the calibration tables are written structurally
//! valid but placeholder, and making them physical is the next gate.
//!
//! On-disk contract this writer must honour (all verified against the reader in `handle.rs`):
//!   - `analysis.tdf_bin` begins with a 64-byte zero pad; the first frame's `TimsId` is therefore 64.
//!   - Each frame block is exactly [`reconstruct_compressed_data`]'s output: `[u32 len+8][u32
//!     num_scans][zstd blob]`, where the reader seeks to `TimsId`, reads `len`, then `len-8` bytes.
//!   - `Frames.Id` is 1-based and contiguous; the reader indexes `frame_meta_data[frame_id - 1]`.
//!   - `GlobalMetadata.TimsCompressionType = 2` (the zstd branch); `ScanMode` 9 reads as DIA, 8 as DDA.

use crate::data::utility::reconstruct_compressed_data;
use rusqlite::{params, Connection};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The 64-byte prefix pad Bruker `.d` files carry before the first frame block.
const BIN_PREFIX_PAD: u64 = 64;

/// The `Frames` table — exactly the 17 columns the reader SELECTs (meta.rs `read_meta_data_sql`).
/// Always synthesised (the simulation owns the frame content), whether or not other tables are copied.
const FRAMES_DDL: &str = "CREATE TABLE Frames (
    Id INTEGER PRIMARY KEY,
    Time REAL, ScanMode INTEGER, Polarity TEXT, MsMsType INTEGER,
    TimsId INTEGER, MaxIntensity REAL, SummedIntensities REAL,
    NumScans INTEGER, NumPeaks INTEGER, MzCalibration INTEGER,
    T1 REAL, T2 REAL, TimsCalibration INTEGER, PropertyGroup INTEGER,
    AccumulationTime REAL, RampTime REAL);";

/// Static acquisition geometry the writer stamps into `GlobalMetadata` and each `Frames` row.
#[derive(Debug, Clone)]
pub struct TdfWriterConfig {
    /// Mobility scans per frame (the `NumScans` and per-block scan count).
    pub num_scans: u32,
    /// Highest TOF index the digitizer produces; stored as `DigitizerNumSamples` (reader adds 1).
    pub digitizer_num_samples: u32,
    /// m/z acquisition range (lower, upper).
    pub mz_range: (f64, f64),
    /// 1/K0 acquisition range (lower, upper).
    pub one_over_k0_range: (f64, f64),
    /// zstd level for the block encoder.
    pub compression_level: i32,
    /// `ScanMode` for every frame: 9 = DIA survey (the reader maps 8→DDA, 9→DIA).
    pub scan_mode: i64,
    /// If set, copy the instrument-physics tables (`MzCalibration`, `TimsCalibration`,
    /// `GlobalMetadata`, `Segments`) verbatim from this reference `.d`, and inherit each `Frames`
    /// row's per-frame instrument state (Polarity, T1/T2, calibration ids, timings) from it — so a
    /// vendor reader (openTIMS/DiaNN via the Bruker SDK) derives correct m/z and 1/K0. Only the 9
    /// simulation fields are overwritten. Mirrors v1's `TDFWriter` copy semantics. When `None`, the
    /// writer synthesises structurally-valid but placeholder calibration (fine for raw round-trips).
    pub reference_d: Option<String>,
}

impl Default for TdfWriterConfig {
    fn default() -> Self {
        TdfWriterConfig {
            num_scans: 709,
            digitizer_num_samples: 400_000,
            mz_range: (100.0, 1700.0),
            one_over_k0_range: (0.6, 1.6),
            compression_level: 1,
            scan_mode: 9,
            reference_d: None,
        }
    }
}

use rusqlite::types::Value;

/// A `Frames` row template: the full column list and one row of values to clone. In reference mode
/// this is a real reference frame row (so every Bruker column the SDK needs — `Pressure`, `T1`, `T2`,
/// calibration ids, timings, ... — is present); the simulation overwrites only the 9 content fields.
/// In placeholder mode it is the minimal 17-column row.
struct FramesTemplate {
    columns: Vec<String>,
    values: Vec<Value>,
    /// Column index of each field the simulation overwrites (None if the column is absent).
    idx: OverrideIdx,
}

struct OverrideIdx {
    id: Option<usize>,
    time: Option<usize>,
    scan_mode: Option<usize>,
    ms_ms_type: Option<usize>,
    tims_id: Option<usize>,
    max_intensity: Option<usize>,
    summed: Option<usize>,
    num_scans: Option<usize>,
    num_peaks: Option<usize>,
}

impl FramesTemplate {
    fn resolve(columns: &[String]) -> OverrideIdx {
        let find = |name: &str| columns.iter().position(|c| c == name);
        OverrideIdx {
            id: find("Id"),
            time: find("Time"),
            scan_mode: find("ScanMode"),
            ms_ms_type: find("MsMsType"),
            tims_id: find("TimsId"),
            max_intensity: find("MaxIntensity"),
            summed: find("SummedIntensities"),
            num_scans: find("NumScans"),
            num_peaks: find("NumPeaks"),
        }
    }

    /// Copy the reference `.d`'s full `Frames` schema (into the already-attached `ref` db) and read
    /// one row as the template.
    fn from_reference(conn: &Connection) -> Result<Self, Box<dyn std::error::Error>> {
        // Copy the reference's Frames columns (not its DDL — that would drag in the reference's own foreign
        // keys to PropertyGroups/etc. which we don't create). `AS SELECT` drops the PK, so add a UNIQUE
        // index on Id: a valid target for the DDA `Precursors`/`PasefFrameMsMsInfo` foreign keys to
        // `Frames(Id)` (SQLite accepts a UNIQUE index, not only a PRIMARY KEY), and it matches the reader's
        // by-Id access.
        conn.execute_batch(
            "CREATE TABLE Frames AS SELECT * FROM ref.Frames WHERE 0;
             CREATE UNIQUE INDEX IF NOT EXISTS FramesIdIndex ON Frames(Id);",
        )?;
        let columns: Vec<String> = {
            let stmt = conn.prepare("SELECT * FROM ref.Frames LIMIT 0")?;
            stmt.column_names().into_iter().map(|s| s.to_string()).collect()
        };
        let ncol = columns.len();
        let values: Vec<Value> = conn.query_row("SELECT * FROM ref.Frames ORDER BY Id LIMIT 1", [], |row| {
            (0..ncol).map(|i| row.get::<_, Value>(i)).collect::<rusqlite::Result<Vec<Value>>>()
        })?;
        let idx = Self::resolve(&columns);
        Ok(FramesTemplate { columns, values, idx })
    }

    /// The placeholder 17-column template (no reference).
    fn placeholder(conn: &Connection) -> Result<Self, Box<dyn std::error::Error>> {
        conn.execute_batch(FRAMES_DDL)?;
        let columns: Vec<String> = vec![
            "Id", "Time", "ScanMode", "Polarity", "MsMsType", "TimsId", "MaxIntensity",
            "SummedIntensities", "NumScans", "NumPeaks", "MzCalibration", "T1", "T2",
            "TimsCalibration", "PropertyGroup", "AccumulationTime", "RampTime",
        ].into_iter().map(String::from).collect();
        // Inherited placeholders; the 9 content fields are overwritten per frame.
        let values: Vec<Value> = vec![
            Value::Integer(0), Value::Real(0.0), Value::Integer(9), Value::Text("+".into()),
            Value::Integer(0), Value::Integer(0), Value::Real(0.0), Value::Real(0.0),
            Value::Integer(0), Value::Integer(0), Value::Integer(1), Value::Real(100.0),
            Value::Real(100.0), Value::Integer(1), Value::Integer(1), Value::Real(100.0),
            Value::Real(100.0),
        ];
        let idx = Self::resolve(&columns);
        Ok(FramesTemplate { columns, values, idx })
    }
}

/// One rendered frame's raw, integer-indexed spectrum. `(scan, tof)` may repeat — the block encoder
/// sums and sorts them, so callers need not pre-dedup. `ms_ms_type` is the Bruker frame type: 0 for an
/// MS1/precursor frame, 9 for a DIA fragment frame.
pub struct RenderedFrame {
    pub frame_id: u32,
    pub retention_time: f64,
    pub ms_ms_type: u8,
    pub scans: Vec<u32>,
    pub tofs: Vec<u32>,
    pub intensities: Vec<u32>,
}

/// An encoded frame block plus the `Frames`-table summaries derived from it. Produced by the pure
/// [`encode_frame_block`] (no `&self`, so it parallelises) and consumed by [`TdfWriter::append_encoded_frame`].
pub struct EncodedBlock {
    pub block: Vec<u8>,
    pub num_peaks: i64,
    pub max_intensity: f64,
    pub summed_intensities: f64,
}

/// Encode one frame's raw triples into a Bruker `.d` block and compute its `Frames` summaries — the pure,
/// `&self`-free core of [`TdfWriter::write_frame`], so a caller can run it across frames on a rayon pool
/// and then append the blocks in order via [`TdfWriter::append_encoded_frame`].
///
/// The summaries describe the block AFTER the encoder's `(scan, tof)` dedup (the encoder SUMS duplicate
/// keys the caller may pass): `num_peaks` = unique keys, `max_intensity` = max SUMMED bin (a bin fed
/// 10 + 20 encodes as 30, not 20), `summed_intensities` = the grand total (unchanged by dedup). Otherwise
/// the `Frames` metadata and the reader's cumulative peak pointer would disagree with the data.
pub fn encode_frame_block(
    scans: &[u32],
    tofs: &[u32],
    intensities: &[u32],
    num_scans: u32,
    compression_level: i32,
) -> Result<EncodedBlock, Box<dyn std::error::Error>> {
    let block = reconstruct_compressed_data(
        scans.to_vec(),
        tofs.to_vec(),
        intensities.to_vec(),
        num_scans,
        compression_level,
    )?;
    let mut acc: std::collections::HashMap<(u32, u32), u64> =
        std::collections::HashMap::with_capacity(scans.len());
    for i in 0..scans.len() {
        *acc.entry((scans[i], tofs[i])).or_insert(0) += intensities[i] as u64;
    }
    let summed: f64 = intensities.iter().map(|&x| x as f64).sum();
    let max_i = acc.values().copied().max().unwrap_or(0) as f64;
    Ok(EncodedBlock {
        block,
        num_peaks: acc.len() as i64,
        max_intensity: max_i,
        summed_intensities: summed,
    })
}

/// One selected DDA precursor — the `Precursors` table row (vendor schema). `id` is per-ion; pass **one
/// canonical row per `id`** (validated unique). A re-selected ion is represented by several PASEF bands,
/// not several precursor rows; its per-event Parent/Intensity/ScanNumber live in the render's sidecar
/// answer key, since the vendor `Precursors` table has room for only one row per ion.
pub struct DdaPrecursor {
    pub id: i64,
    pub largest_peak_mz: f64,
    pub average_mz: f64,
    pub monoisotopic_mz: f64,
    pub charge: i64,
    pub scan_number: f64,
    pub intensity: f64,
    /// The survey MS1 `Frames.Id` this precursor was selected from.
    pub parent: i64,
}

/// One PASEF MS2 selection band — a `PasefFrameMsMsInfo` row. A precursor may have MANY of these (one per
/// fragmentation event across the run), each in its own mobility band of an MS2 `frame`.
pub struct DdaPasefWindow {
    pub frame: i64,
    pub scan_num_begin: i64,
    pub scan_num_end: i64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
    /// References `Precursors.Id`.
    pub precursor: i64,
}

/// One accumulated `Frames` row, materialised into SQLite at [`TdfWriter::finalize`].
struct FrameMetaRow {
    id: u32,
    time: f64,
    ms_ms_type: i64,
    tims_id: u64,
    max_intensity: f64,
    summed_intensities: f64,
    num_peaks: i64,
}

/// The writer. Create it, push MS1 frames in ascending `frame_id`, then finalize.
pub struct TdfWriter {
    dir: PathBuf,
    conn: Connection,
    bin: fs::File,
    position: u64,
    config: TdfWriterConfig,
    frames: Vec<FrameMetaRow>,
    /// The `Frames` row template (full reference schema, or the placeholder 17 columns).
    template: FramesTemplate,
    /// True when the calibration / GlobalMetadata tables were copied from a reference `.d`.
    copied_from_reference: bool,
    /// For a DIA run: our replayed `(frame, window_group)` for each MS2 frame. When set, finalize
    /// writes `DiaFrameMsMsInfo` (this map) and copies `DiaFrameMsMsWindows` from the reference.
    dia_frame_to_group: Option<Vec<(u32, u32)>>,
    /// For a DDA run: the selected precursors + their PASEF selection bands. When set, finalize writes
    /// the `Precursors` and `PasefFrameMsMsInfo` tables. Mutually exclusive with `dia_frame_to_group`.
    dda_schedule: Option<(Vec<DdaPrecursor>, Vec<DdaPasefWindow>)>,
}

impl TdfWriter {
    /// Create the `.d` directory and open `analysis.tdf` + `analysis.tdf_bin`. If
    /// `config.reference_d` is set, copy the instrument-physics tables from it (see [`TdfWriterConfig`]).
    pub fn create(dir: impl AsRef<Path>, config: TdfWriterConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let conn = Connection::open(dir.join("analysis.tdf"))?;

        let (template, copied_from_reference) = match &config.reference_d {
            Some(ref_path) => {
                let ref_tdf = Path::new(ref_path).join("analysis.tdf");
                if !ref_tdf.exists() {
                    return Err(format!("reference .d has no analysis.tdf: {}", ref_tdf.display()).into());
                }
                conn.execute("ATTACH DATABASE ?1 AS ref", params![ref_tdf.to_str().unwrap()])?;
                // Copy the instrument-physics tables verbatim (schema + rows).
                conn.execute_batch(
                    "CREATE TABLE MzCalibration   AS SELECT * FROM ref.MzCalibration;
                     CREATE TABLE TimsCalibration AS SELECT * FROM ref.TimsCalibration;
                     CREATE TABLE GlobalMetadata  AS SELECT * FROM ref.GlobalMetadata;",
                )?;
                // Segments is copied with LastFrame patched at finalize (needs the frame count).
                conn.execute_batch("CREATE TABLE Segments AS SELECT * FROM ref.Segments;")?;
                // Frames: full reference schema (so the Bruker SDK finds every column it needs).
                let template = FramesTemplate::from_reference(&conn)?;
                conn.execute("DETACH DATABASE ref", [])?;
                (template, true)
            }
            None => {
                Self::create_placeholder_tables(&conn)?;
                let template = FramesTemplate::placeholder(&conn)?;
                (template, false)
            }
        };

        // tdf_bin: 64-byte zero pad, so the first frame starts at offset 64 (Bruker convention).
        let mut bin = fs::OpenOptions::new().write(true).create(true).truncate(true).open(dir.join("analysis.tdf_bin"))?;
        bin.write_all(&[0u8; BIN_PREFIX_PAD as usize])?;

        Ok(TdfWriter {
            dir,
            conn,
            bin,
            position: BIN_PREFIX_PAD,
            config,
            frames: Vec::new(),
            template,
            copied_from_reference,
            dia_frame_to_group: None,
            dda_schedule: None,
        })
    }

    /// Mark this a DIA run: `frame_to_group` is our replayed `(frame_id, window_group)` for each MS2
    /// frame. Finalize will write `DiaFrameMsMsInfo` from it and copy `DiaFrameMsMsWindows` from the
    /// reference. Requires reference mode (the window definitions come from the reference `.d`).
    pub fn set_dia_schedule(&mut self, frame_to_group: Vec<(u32, u32)>) {
        self.dia_frame_to_group = Some(frame_to_group);
    }

    /// Mark this a DDA-PASEF run: `precursors` are the selected precursors — **one canonical row per `id`**
    /// (per-event Parent/Intensity/ScanNumber belong in the render's sidecar answer key, not the vendor
    /// `Precursors` table) — and `pasef` the per-event selection bands (`PasefFrameMsMsInfo` rows, all kept;
    /// a re-selected ion has several). Finalize validates the schedule then writes both tables with the
    /// vendor schema. Forces `scan_mode = 8` (DDA) so the frames can't be labelled DIA.
    pub fn set_dda_schedule(&mut self, precursors: Vec<DdaPrecursor>, pasef: Vec<DdaPasefWindow>) {
        self.config.scan_mode = 8;
        self.dda_schedule = Some((precursors, pasef));
    }

    /// Encode one frame (MS1 or MS2), append its block, and record its `TimsId` offset. Frames must be
    /// pushed in ascending `frame_id` starting at 1 (the reader indexes by `frame_id - 1`).
    pub fn write_frame(&mut self, frame: &RenderedFrame) -> Result<(), Box<dyn std::error::Error>> {
        let blk = encode_frame_block(
            &frame.scans,
            &frame.tofs,
            &frame.intensities,
            self.config.num_scans,
            self.config.compression_level,
        )?;
        self.append_encoded_frame(frame.frame_id, frame.retention_time, frame.ms_ms_type, blk)
    }

    /// Append an already-encoded frame block in order, recording its `TimsId` offset and `Frames`
    /// metadata. Frames must be appended in ascending `frame_id` starting at 1. This is the sequential
    /// tail of [`write_frame`]; splitting it out lets a caller encode frames in parallel (via the pure
    /// [`encode_frame_block`]) and then append the blocks here in frame order — byte-identical to a
    /// serial `write_frame` loop, since block bytes are position-independent and appended in the same
    /// order.
    pub fn append_encoded_frame(
        &mut self,
        frame_id: u32,
        retention_time: f64,
        ms_ms_type: u8,
        blk: EncodedBlock,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected = self.frames.len() as u32 + 1;
        if frame_id != expected {
            return Err(format!(
                "frames must be written in order: expected id {expected}, got {frame_id}"
            )
            .into());
        }
        let tims_id = self.position;
        self.bin.write_all(&blk.block)?;
        self.position += blk.block.len() as u64;
        self.frames.push(FrameMetaRow {
            id: frame_id,
            time: retention_time,
            ms_ms_type: ms_ms_type as i64,
            tims_id,
            max_intensity: blk.max_intensity,
            summed_intensities: blk.summed_intensities,
            num_peaks: blk.num_peaks,
        });
        Ok(())
    }

    /// Write the `Frames`, `GlobalMetadata`, and (placeholder) calibration tables, mark the file closed
    /// properly, and flush. Consumes the writer.
    pub fn finalize(mut self) -> Result<(), Box<dyn std::error::Error>> {
        // A run is either DIA or DDA, not both — writing both table families is a corrupt mixed acquisition.
        if self.dia_frame_to_group.is_some() && self.dda_schedule.is_some() {
            return Err("acquisition is either DIA or DDA, not both (both schedules were set)".into());
        }
        // Validate the DDA schedule BEFORE any table is written, so an invalid schedule (orphan band, bad
        // parent, non-MS2 PASEF frame, duplicate id or band) fails cleanly instead of leaving a partial
        // `.d` — SQLite's FKs are declared but not enforced (pragma off), so we check them ourselves.
        if self.dda_schedule.is_some() {
            self.validate_dda_schedule()?;
        }
        self.bin.flush()?;
        self.write_frames_table()?;
        if self.copied_from_reference {
            // Instrument tables came from the reference; only patch what the simulation changes.
            let last_frame = self.frames.iter().map(|f| f.id).max().unwrap_or(0) as i64;
            // Segments.LastFrame must point at the simulated run's end, not the reference's.
            let _ = self.conn.execute("UPDATE Segments SET LastFrame = ?1", params![last_frame]);
            // Not closed yet — finalize stamps ClosedProperly = 1 only after every table is written.
            self.conn.execute(
                "UPDATE GlobalMetadata SET Value = '0' WHERE Key = 'ClosedProperly'",
                [],
            )?;
            // Our blocks are ALWAYS zstd (via reconstruct_compressed_data), regardless of what the
            // reference used. If the reference was LZF (type 1), the copied metadata would point every
            // reader at the wrong decompressor. Force type 2.
            self.conn.execute(
                "UPDATE GlobalMetadata SET Value = '2' WHERE Key = 'TimsCompressionType'",
                [],
            )?;
        } else {
            self.write_global_metadata()?;
            self.write_calibration_tables()?;
        }
        if self.dia_frame_to_group.is_some() {
            self.write_dia_tables()?;
        }
        if self.dda_schedule.is_some() {
            self.write_dda_tables()?;
        }
        // Mark closed cleanly ONLY after every table is written — a crash/error before here leaves
        // ClosedProperly = 0, so a truncated `.d` is detectable rather than falsely complete.
        self.conn.execute(
            "UPDATE GlobalMetadata SET Value = '1' WHERE Key = 'ClosedProperly'",
            [],
        )?;
        Ok(())
    }

    /// Validate a DDA schedule against the written frames — the checks SQLite's (unenforced) foreign keys
    /// and primary keys would make, done explicitly so a bad schedule is rejected before anything is
    /// written. Contract: one canonical `Precursors` row per id; every PASEF band references a known
    /// precursor and an MS2 frame with a distinct start scan; every precursor parent is an MS1 frame.
    fn validate_dda_schedule(&self) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::{HashMap, HashSet};
        let (precursors, pasef) = self.dda_schedule.as_ref().unwrap();
        let frame_type: HashMap<u32, i64> = self.frames.iter().map(|f| (f.id, f.ms_ms_type)).collect();

        let mut ids: HashSet<i64> = HashSet::new();
        for p in precursors {
            if !ids.insert(p.id) {
                return Err(format!(
                    "duplicate Precursors.Id {} — pass one canonical row per precursor (per-event \
                     Parent/Intensity/ScanNumber belong in the sidecar answer key)",
                    p.id
                )
                .into());
            }
            match frame_type.get(&(p.parent as u32)) {
                Some(0) => {}
                Some(t) => return Err(format!("Precursor {} Parent {} is a MsMsType={} frame, not MS1", p.id, p.parent, t).into()),
                None => return Err(format!("Precursor {} Parent {} is not a written frame", p.id, p.parent).into()),
            }
        }

        let mut bands: HashSet<(i64, i64)> = HashSet::new();
        for w in pasef {
            if !ids.contains(&w.precursor) {
                return Err(format!("PASEF band (frame {}) references unknown Precursor {}", w.frame, w.precursor).into());
            }
            match frame_type.get(&(w.frame as u32)) {
                Some(8) => {}
                Some(t) => return Err(format!("PASEF Frame {} is MsMsType={}, expected 8 (MS2)", w.frame, t).into()),
                None => return Err(format!("PASEF Frame {} is not a written frame", w.frame).into()),
            }
            if w.scan_num_begin < 0 || w.scan_num_end <= w.scan_num_begin
                || w.scan_num_end as u32 > self.config.num_scans
            {
                return Err(format!(
                    "PASEF band [{}, {}] invalid for {} scans", w.scan_num_begin, w.scan_num_end, self.config.num_scans
                )
                .into());
            }
            if !bands.insert((w.frame, w.scan_num_begin)) {
                return Err(format!("duplicate PASEF (Frame {}, ScanNumBegin {}) — bands need distinct start scans", w.frame, w.scan_num_begin).into());
            }
        }
        Ok(())
    }

    /// Persist the DIA schedule: `DiaFrameMsMsInfo` from OUR replayed frame→group (a reference of a
    /// different length would be wrong), and `DiaFrameMsMsWindows` copied verbatim from the reference
    /// (the window geometry we replayed and gated against).
    fn write_dia_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        let frame_to_group = self.dia_frame_to_group.as_ref().unwrap();
        let ref_path = self
            .config
            .reference_d
            .as_ref()
            .ok_or("DIA output needs --reference-d for the window definitions")?;

        // Copy the window definitions verbatim (same-schema copy as the calibration tables).
        let ref_tdf = Path::new(ref_path).join("analysis.tdf");
        self.conn.execute("ATTACH DATABASE ?1 AS refdia", params![ref_tdf.to_str().unwrap()])?;
        self.conn
            .execute_batch("CREATE TABLE DiaFrameMsMsWindows AS SELECT * FROM refdia.DiaFrameMsMsWindows;")?;
        self.conn.execute("DETACH DATABASE refdia", [])?;

        // Synthesise our frame → window group.
        self.conn.execute_batch(
            "CREATE TABLE DiaFrameMsMsInfo (Frame INTEGER PRIMARY KEY, WindowGroup INTEGER);",
        )?;
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare("INSERT INTO DiaFrameMsMsInfo (Frame, WindowGroup) VALUES (?1, ?2)")?;
            for &(frame, group) in frame_to_group {
                stmt.execute(params![frame, group])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Persist the DDA schedule (already validated): `Precursors` (vendor schema, one canonical row per
    /// `Id`) and `PasefFrameMsMsInfo` (every selection band). Per-event detail that the single-row-per-ion
    /// `Precursors` table can't hold is the render's sidecar answer key's job, not the `.d`'s.
    fn write_dda_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (precursors, pasef) = self.dda_schedule.as_ref().unwrap();

        self.conn.execute_batch(
            "CREATE TABLE Precursors (
                 Id INTEGER PRIMARY KEY,
                 LargestPeakMz REAL NOT NULL, AverageMz REAL NOT NULL, MonoisotopicMz REAL,
                 Charge INTEGER, ScanNumber REAL NOT NULL, Intensity REAL NOT NULL, Parent INTEGER,
                 FOREIGN KEY(Parent) REFERENCES Frames(Id));
             CREATE INDEX PrecursorsParentIndex ON Precursors (Parent);",
        )?;
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO Precursors
                 (Id, LargestPeakMz, AverageMz, MonoisotopicMz, Charge, ScanNumber, Intensity, Parent)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            // Ids are validated unique in validate_dda_schedule (called before any write).
            for p in precursors {
                stmt.execute(params![
                    p.id, p.largest_peak_mz, p.average_mz, p.monoisotopic_mz,
                    p.charge, p.scan_number, p.intensity, p.parent
                ])?;
            }
        }
        tx.commit()?;

        self.conn.execute_batch(
            "CREATE TABLE PasefFrameMsMsInfo (
                 Frame INTEGER NOT NULL, ScanNumBegin INTEGER NOT NULL, ScanNumEnd INTEGER NOT NULL,
                 IsolationMz REAL NOT NULL, IsolationWidth REAL NOT NULL, CollisionEnergy REAL NOT NULL,
                 Precursor INTEGER, PRIMARY KEY(Frame, ScanNumBegin),
                 FOREIGN KEY(Frame) REFERENCES Frames(Id),
                 FOREIGN KEY(Precursor) REFERENCES Precursors(Id)) WITHOUT ROWID;",
        )?;
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO PasefFrameMsMsInfo
                 (Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, Precursor)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for w in pasef {
                stmt.execute(params![
                    w.frame, w.scan_num_begin, w.scan_num_end,
                    w.isolation_mz, w.isolation_width, w.collision_energy, w.precursor
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Path to the finished `.d` directory.
    pub fn path(&self) -> &Path {
        &self.dir
    }

    // ── SQLite ───────────────────────────────────────────────────────────────

    /// Placeholder instrument-physics tables, used only when there is no reference `.d`.
    fn create_placeholder_tables(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
        conn.execute_batch(
            "CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT);
             CREATE TABLE MzCalibration (
                 Id INTEGER PRIMARY KEY, ModelType INTEGER,
                 DigitizerTimebase REAL, DigitizerDelay REAL, T1 REAL, T2 REAL,
                 dC1 REAL, dC2 REAL, C0 REAL, C1 REAL, C2 REAL, C3 REAL, C4 REAL);
             CREATE TABLE TimsCalibration (
                 Id INTEGER PRIMARY KEY, ModelType INTEGER,
                 C0 REAL, C1 REAL, C2 REAL, C3 REAL, C4 REAL,
                 C5 REAL, C6 REAL, C7 REAL, C8 REAL, C9 REAL);",
        )?;
        Ok(())
    }

    fn write_frames_table(&self) -> Result<(), Box<dyn std::error::Error>> {
        let cols = &self.template.columns;
        let collist = cols.join(", ");
        let placeholders = (1..=cols.len()).map(|i| format!("?{i}")).collect::<Vec<_>>().join(", ");
        let sql = format!("INSERT INTO Frames ({collist}) VALUES ({placeholders})");
        let idx = &self.template.idx;

        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(&sql)?;
            for f in &self.frames {
                // Clone the template row (reference frame, or placeholder), then overwrite ONLY the 9
                // fields the simulation owns — v1's copy-then-overwrite semantics, so every other
                // Bruker column (Pressure, T1/T2, calibration ids, timings, ...) is inherited.
                let mut row = self.template.values.clone();
                if let Some(i) = idx.id { row[i] = Value::Integer(f.id as i64); }
                if let Some(i) = idx.time { row[i] = Value::Real(f.time); }
                if let Some(i) = idx.scan_mode { row[i] = Value::Integer(self.config.scan_mode); }
                if let Some(i) = idx.ms_ms_type { row[i] = Value::Integer(f.ms_ms_type); }
                if let Some(i) = idx.tims_id { row[i] = Value::Integer(f.tims_id as i64); }
                if let Some(i) = idx.max_intensity { row[i] = Value::Integer(f.max_intensity as i64); }
                if let Some(i) = idx.summed { row[i] = Value::Integer(f.summed_intensities as i64); }
                if let Some(i) = idx.num_scans { row[i] = Value::Integer(self.config.num_scans as i64); }
                if let Some(i) = idx.num_peaks { row[i] = Value::Integer(f.num_peaks); }
                stmt.execute(rusqlite::params_from_iter(row.iter()))?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn write_global_metadata(&self) -> Result<(), Box<dyn std::error::Error>> {
        let kv: Vec<(&str, String)> = vec![
            ("SchemaType", "TDF".into()),
            ("SchemaVersionMajor", "3".into()),
            ("SchemaVersionMinor", "7".into()),
            ("AcquisitionSoftwareVendor", "timsim".into()),
            ("InstrumentVendor", "Bruker".into()),
            ("TimsCompressionType", "2".into()), // zstd branch — required by the reader
            ("MaxNumPeaksPerScan", "1000".into()),
            ("MzAcqRangeLower", format!("{}", self.config.mz_range.0)),
            ("MzAcqRangeUpper", format!("{}", self.config.mz_range.1)),
            ("OneOverK0AcqRangeLower", format!("{}", self.config.one_over_k0_range.0)),
            ("OneOverK0AcqRangeUpper", format!("{}", self.config.one_over_k0_range.1)),
            ("DigitizerNumSamples", format!("{}", self.config.digitizer_num_samples)),
            // "0" here; finalize flips it to "1" only after every table is written (truncation-detectable).
            ("ClosedProperly", "0".into()),
        ];
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare("INSERT OR REPLACE INTO GlobalMetadata (Key, Value) VALUES (?1, ?2)")?;
            for (k, v) in kv {
                stmt.execute(params![k, v])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Structurally valid ModelType-2 calibration rows with PLACEHOLDER coefficients. They are not
    /// touched by raw `(scan, tof, intensity)` reads (the round-trip oracle); making them physical
    /// (so a vendor reader derives correct m/z and 1/K0) is the next gate.
    fn write_calibration_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.execute(
            "INSERT INTO MzCalibration
             (Id, ModelType, DigitizerTimebase, DigitizerDelay, T1, T2, dC1, dC2, C0, C1, C2, C3, C4)
             VALUES (1, 2, 1.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)",
            [],
        )?;
        self.conn.execute(
            "INSERT INTO TimsCalibration
             (Id, ModelType, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)
             VALUES (1, 2, 1.0, ?1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)",
            params![(self.config.num_scans as i64 - 1)], // C1+1 is read as num_scans elsewhere
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::handle::{
        SimpleIndexConverter, TimsData, TimsIndexConverter, TimsLazyLoder, TimsRawDataLayout,
    };

    /// The writer oracle: write a few MS1 frames with known `(scan, tof, intensity)`, reopen the `.d`
    /// with the rustims reader, and require every raw point to round-trip. This proves the directory,
    /// SQLite schema, `TimsId` offsets, 64-byte pad, and block format are all correct — independent of
    /// calibration accuracy (raw reads never touch it).
    #[test]
    fn ms1_d_round_trips_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("timsim_tdf_writer_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let cfg = TdfWriterConfig { num_scans: 100, ..Default::default() };
        // Inputs are pre-sorted-unique by (scan, tof) so read-back (which the encoder sorts) matches.
        let frames = vec![
            RenderedFrame {
                frame_id: 1,
                retention_time: 0.5,
                ms_ms_type: 0,
                scans: vec![2, 2, 10, 40],
                tofs: vec![100, 350, 200, 500],
                intensities: vec![10, 20, 30, 40],
            },
            RenderedFrame {
                frame_id: 2,
                retention_time: 1.0,
                ms_ms_type: 0,
                scans: vec![5, 50],
                tofs: vec![123, 456],
                intensities: vec![7, 99],
            },
            RenderedFrame {
                frame_id: 3, // deliberately empty frame (gap-fill analog)
                retention_time: 1.5,
                ms_ms_type: 0,
                scans: vec![],
                tofs: vec![],
                intensities: vec![],
            },
        ];

        let mut w = TdfWriter::create(&dir, cfg).unwrap();
        for f in &frames {
            w.write_frame(f).unwrap();
        }
        w.finalize().unwrap();

        let layout = TimsRawDataLayout::new(dir.to_str().unwrap());
        assert_eq!(layout.frame_meta_data.len(), 3, "reader saw wrong frame count");
        // get_raw_frame returns raw indices and never touches the converter; any converter suffices.
        let converter = TimsIndexConverter::Simple(SimpleIndexConverter::from_boundaries(
            100.0, 1700.0, 400_000, 0.6, 1.6, 99,
        ));
        let reader = TimsLazyLoder { raw_data_layout: layout, index_converter: converter };

        for f in &frames {
            let raw = reader.get_raw_frame(f.frame_id);
            // get_raw_frame returns Bruker's native layout: `scan` is per-scan peak COUNTS, so expand
            // it to one scan index per peak before comparing.
            let scans = crate::data::utility::flatten_scan_values(&raw.scan, true);
            // Sort both sides by (scan, tof) and compare — the encoder emits ascending (scan, tof).
            let mut got: Vec<(u32, u32, u32)> = scans
                .iter()
                .zip(raw.tof.iter())
                .zip(raw.intensity.iter())
                .map(|((&s, &t), &i)| (s, t, i as u32))
                .collect();
            got.sort_unstable();
            let mut want: Vec<(u32, u32, u32)> = f
                .scans
                .iter()
                .zip(f.tofs.iter())
                .zip(f.intensities.iter())
                .map(|((&s, &t), &i)| (s, t, i))
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "frame {} raw data did not round-trip", f.frame_id);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// M1 DDA gate: write a trivial DDA schedule where ONE ion (precursor 100) is selected in TWO
    /// different MS2 frames (two events), plus a second precursor, and assert:
    ///   1. every raw PASEF band round-trips through the reader (`read_pasef_frame_ms_ms_info`), so the
    ///      re-selected ion keeps BOTH bands — the identity-safe API, not the lossy `get_selected_precursors`;
    ///   2. `Precursors` is deduped to one row per id (the duplicate 100 collapses);
    ///   3. the tables are relationally consistent (every PASEF `Precursor` exists in `Precursors`,
    ///      every `Parent` is an MS1 frame).
    #[test]
    fn dda_d_round_trips_with_reselected_precursor() {
        use crate::data::meta::read_pasef_frame_ms_ms_info;

        let dir = std::env::temp_dir().join(format!("timsim_tdf_dda_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        // Frames: 1=MS1, 2=MS2, 3=MS2, 4=MS1, 5=MS2.
        let ms_types = [0u8, 8, 8, 0, 8];
        let cfg = TdfWriterConfig { num_scans: 100, scan_mode: 8, ..Default::default() };
        let mut w = TdfWriter::create(&dir, cfg).unwrap();
        for (i, &t) in ms_types.iter().enumerate() {
            w.write_frame(&RenderedFrame {
                frame_id: i as u32 + 1,
                retention_time: i as f64 * 0.1,
                ms_ms_type: t,
                scans: vec![10 + i as u32, 40],
                tofs: vec![200, 500],
                intensities: vec![50, 90],
            })
            .unwrap();
        }

        // One canonical Precursors row per ion (unique ids). The re-selection of ion 100 is carried by
        // its TWO PASEF bands below, not by a second precursor row.
        let precursors = vec![
            DdaPrecursor { id: 100, largest_peak_mz: 596.9, average_mz: 596.7, monoisotopic_mz: 596.6, charge: 2, scan_number: 20.5, intensity: 4600.0, parent: 1 },
            DdaPrecursor { id: 200, largest_peak_mz: 712.4, average_mz: 712.2, monoisotopic_mz: 712.1, charge: 3, scan_number: 50.0, intensity: 2100.0, parent: 1 },
        ];
        // Three PASEF bands: precursor 100 in frames 2 AND 5, precursor 200 in frame 2.
        let pasef = vec![
            DdaPasefWindow { frame: 2, scan_num_begin: 10, scan_num_end: 32, isolation_mz: 596.9, isolation_width: 2.0, collision_energy: 31.5, precursor: 100 },
            DdaPasefWindow { frame: 2, scan_num_begin: 40, scan_num_end: 62, isolation_mz: 712.4, isolation_width: 3.0, collision_energy: 34.0, precursor: 200 },
            DdaPasefWindow { frame: 5, scan_num_begin: 10, scan_num_end: 32, isolation_mz: 596.9, isolation_width: 2.0, collision_energy: 31.6, precursor: 100 },
        ];
        w.set_dda_schedule(precursors, pasef);
        w.finalize().unwrap();

        // (1) every raw PASEF band survives — the re-selected ion 100 keeps BOTH.
        let bands = read_pasef_frame_ms_ms_info(dir.to_str().unwrap()).unwrap();
        assert_eq!(bands.len(), 3, "all PASEF bands must round-trip");
        let for_100: Vec<&_> = bands.iter().filter(|b| b.precursor_id == 100).collect();
        assert_eq!(for_100.len(), 2, "re-selected precursor 100 must keep both bands");
        let frames_100: std::collections::HashSet<i64> = for_100.iter().map(|b| b.frame_id).collect();
        assert_eq!(frames_100, [2i64, 5].into_iter().collect(), "precursor 100's two events are frames 2 and 5");

        // (2)+(3) relational + semantic invariants.
        let conn = Connection::open(dir.join("analysis.tdf")).unwrap();
        let n_prec: i64 = conn.query_row("SELECT COUNT(*) FROM Precursors", [], |r| r.get(0)).unwrap();
        assert_eq!(n_prec, 2, "one Precursors row per ion");
        // Frames are DDA (ScanMode 8); every PASEF frame is an MS2 (MsMsType 8) frame.
        let non8: i64 = conn.query_row("SELECT COUNT(*) FROM Frames WHERE ScanMode != 8", [], |r| r.get(0)).unwrap();
        assert_eq!(non8, 0, "DDA frames must have ScanMode 8");
        let pasef_not_ms2: i64 = conn.query_row(
            "SELECT COUNT(*) FROM PasefFrameMsMsInfo p LEFT JOIN Frames f ON p.Frame = f.Id WHERE f.MsMsType != 8 OR f.Id IS NULL",
            [], |r| r.get(0)).unwrap();
        assert_eq!(pasef_not_ms2, 0, "every PASEF Frame must be an MS2 frame");
        let bad_parent: i64 = conn.query_row(
            "SELECT COUNT(*) FROM Precursors pr LEFT JOIN Frames f ON pr.Parent = f.Id WHERE f.MsMsType != 0 OR f.Id IS NULL",
            [], |r| r.get(0)).unwrap();
        assert_eq!(bad_parent, 0, "every Precursor.Parent must be an MS1 survey frame");
        // SQLite's own FK checker must find no violations, and the file must be marked closed.
        let fk_violations: i64 = conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| r.get(0)).unwrap();
        assert_eq!(fk_violations, 0, "foreign_key_check must be clean");
        let closed: String = conn.query_row("SELECT Value FROM GlobalMetadata WHERE Key='ClosedProperly'", [], |r| r.get(0)).unwrap();
        assert_eq!(closed, "1", "ClosedProperly stamped after all writes");

        let _ = fs::remove_dir_all(&dir);
    }

    /// The writer must REJECT an invalid DDA schedule up front (no partial `.d`), because SQLite's FKs
    /// are declared but not enforced. Here a PASEF band references a precursor that doesn't exist.
    #[test]
    fn dda_rejects_orphan_pasef_band() {
        let dir = std::env::temp_dir().join(format!("timsim_tdf_dda_reject_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cfg = TdfWriterConfig { num_scans: 100, scan_mode: 8, ..Default::default() };
        let mut w = TdfWriter::create(&dir, cfg).unwrap();
        for (i, &t) in [0u8, 8].iter().enumerate() {
            w.write_frame(&RenderedFrame {
                frame_id: i as u32 + 1, retention_time: 0.0, ms_ms_type: t,
                scans: vec![10], tofs: vec![100], intensities: vec![5],
            }).unwrap();
        }
        let precursors = vec![DdaPrecursor { id: 1, largest_peak_mz: 500.0, average_mz: 500.0, monoisotopic_mz: 500.0, charge: 2, scan_number: 10.0, intensity: 100.0, parent: 1 }];
        // Band references precursor 999 (not in `precursors`).
        let pasef = vec![DdaPasefWindow { frame: 2, scan_num_begin: 5, scan_num_end: 20, isolation_mz: 500.0, isolation_width: 2.0, collision_energy: 30.0, precursor: 999 }];
        w.set_dda_schedule(precursors, pasef);
        assert!(w.finalize().is_err(), "an orphan PASEF band must be rejected before writing");
        let _ = fs::remove_dir_all(&dir);
    }
}
