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
        conn.execute_batch("CREATE TABLE Frames AS SELECT * FROM ref.Frames WHERE 0;")?;
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
        })
    }

    /// Mark this a DIA run: `frame_to_group` is our replayed `(frame_id, window_group)` for each MS2
    /// frame. Finalize will write `DiaFrameMsMsInfo` from it and copy `DiaFrameMsMsWindows` from the
    /// reference. Requires reference mode (the window definitions come from the reference `.d`).
    pub fn set_dia_schedule(&mut self, frame_to_group: Vec<(u32, u32)>) {
        self.dia_frame_to_group = Some(frame_to_group);
    }

    /// Encode one frame (MS1 or MS2), append its block, and record its `TimsId` offset. Frames must be
    /// pushed in ascending `frame_id` starting at 1 (the reader indexes by `frame_id - 1`).
    pub fn write_frame(&mut self, frame: &RenderedFrame) -> Result<(), Box<dyn std::error::Error>> {
        let expected = self.frames.len() as u32 + 1;
        if frame.frame_id != expected {
            return Err(format!(
                "frames must be written in order: expected id {expected}, got {}",
                frame.frame_id
            )
            .into());
        }

        let block = reconstruct_compressed_data(
            frame.scans.clone(),
            frame.tofs.clone(),
            frame.intensities.clone(),
            self.config.num_scans,
            self.config.compression_level,
        )?;

        let tims_id = self.position;
        self.bin.write_all(&block)?;
        self.position += block.len() as u64;

        // Frame-level summaries must describe the block AFTER the encoder's `(scan, tof)` dedup — the
        // writer's contract lets callers pass duplicates, which the encoder SUMS. So aggregate here
        // too: `NumPeaks` = unique keys, `MaxIntensity` = max SUMMED bin (a bin fed 10 + 20 encodes as
        // 30, not 20), `SummedIntensities` = the grand total (unchanged by dedup). Otherwise the Frames
        // metadata and the reader's cumulative peak pointer disagree with the data.
        let mut acc: std::collections::HashMap<(u32, u32), u64> =
            std::collections::HashMap::with_capacity(frame.scans.len());
        for i in 0..frame.scans.len() {
            *acc.entry((frame.scans[i], frame.tofs[i])).or_insert(0) += frame.intensities[i] as u64;
        }
        let summed: f64 = frame.intensities.iter().map(|&x| x as f64).sum();
        let max_i = acc.values().copied().max().unwrap_or(0) as f64;
        self.frames.push(FrameMetaRow {
            id: frame.frame_id,
            time: frame.retention_time,
            ms_ms_type: frame.ms_ms_type as i64,
            tims_id,
            max_intensity: max_i,
            summed_intensities: summed,
            num_peaks: acc.len() as i64,
        });
        Ok(())
    }

    /// Write the `Frames`, `GlobalMetadata`, and (placeholder) calibration tables, mark the file closed
    /// properly, and flush. Consumes the writer.
    pub fn finalize(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.bin.flush()?;
        self.write_frames_table()?;
        if self.copied_from_reference {
            // Instrument tables came from the reference; only patch what the simulation changes.
            let last_frame = self.frames.iter().map(|f| f.id).max().unwrap_or(0) as i64;
            // Segments.LastFrame must point at the simulated run's end, not the reference's.
            let _ = self.conn.execute("UPDATE Segments SET LastFrame = ?1", params![last_frame]);
            // Mark closed cleanly (the reference's value is copied, but assert it regardless).
            self.conn.execute(
                "UPDATE GlobalMetadata SET Value = '1' WHERE Key = 'ClosedProperly'",
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
            // Written last so a truncated run is detectable: "1" only after a clean finalize.
            ("ClosedProperly", "1".into()),
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
}
