import os
import sqlite3
import warnings

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List

from numpy._typing import NDArray

from imspy_simulation.utility import get_compressible_data
from imspy_core.timstof import TimsDataset
from imspy_core.timstof.frame import TimsFrame
import zstd

import imspy_connector
ims = imspy_connector.py_dataset


def validate_frames_id_uniqueness(meta_df: pd.DataFrame) -> None:
    """Raise if the accumulated Frames rows have duplicate ``Id`` values.

    Bruker's `.tdf` schema relies on ``Frames.Id`` being unique and
    contiguous; downstream readers that hash by Id (OpenTIMS, ionmaiden's
    ``d2ms1``) throw ``IndexError: unordered_map::at`` on duplicates.
    The reference-noise sampler used to emit ``MsType::Unknown`` frames
    that propagated duplicate ``frame_id=1, retention_time=0.0`` metadata
    to the writer (see rustdf/src/data/dia.rs for the upstream fix).
    This guard catches the failure at the writer boundary so a corrupt
    `.d` is never emitted.
    """
    if meta_df.empty or "Id" not in meta_df.columns:
        return
    n_total = len(meta_df)
    n_unique = meta_df["Id"].nunique()
    if n_unique == n_total:
        return
    dupe_ids = (
        meta_df["Id"].value_counts().loc[lambda s: s > 1].head(5)
    )
    raise RuntimeError(
        f"Frames.Id is not unique — {n_total - n_unique} duplicates "
        f"among {n_total} rows. Top duplicate Ids "
        f"(Id -> count): {dupe_ids.to_dict()}. This indicates an "
        f"upstream bug in the simulation pipeline (most commonly "
        f"the reference-noise sampler emitting MsType::Unknown "
        f"frames; see rustdf/src/data/dia.rs)."
    )


# SQLite's default busy timeout (5 s via the sqlite3 module) is short enough
# that ordinary contention surfaces as a hard failure; 30 s absorbs a slow
# flush without masking a genuinely stuck lock.
_SQLITE_BUSY_TIMEOUT_S = 30.0

# Filesystems on which SQLite's POSIX advisory locking is unreliable or simply
# absent. Writing a `.d` onto one of these reports "database is locked" even
# with a single writer, which is the failure people hit when they move a run to
# a VM and point --save_path at a mounted share.
_UNSAFE_LOCK_FILESYSTEMS = {
    "nfs", "nfs4", "cifs", "smbfs", "smb3", "vboxsf", "9p", "virtiofs",
    "afs", "lustre", "gpfs", "fuse", "fuseblk", "fuse.sshfs", "fuse.s3fs",
    "fuse.rclone", "fuse.glusterfs",
}


def _filesystem_type(path) -> str:
    """Best-effort filesystem type of the mount that ``path`` lives on.

    Linux-only (reads ``/proc/mounts``); returns ``"unknown"`` anywhere the
    lookup is unavailable. Used only to enrich a diagnostic, never for control
    flow.
    """
    try:
        target = os.path.realpath(str(path))
        best_mount, best_type = "", "unknown"
        with open("/proc/mounts", "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = parts[1].replace("\\040", " ")
                fstype = parts[2]
                if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
                    if len(mount_point) >= len(best_mount):
                        best_mount, best_type = mount_point, fstype
        return best_type
    except OSError:
        return "unknown"


def _locked_database_error(db_path, exc: Exception) -> RuntimeError:
    """Turn SQLite's bare "database is locked" into an actionable diagnostic.

    The bare message says nothing about *why* the file is locked, and the two
    causes need opposite fixes: a live process still holding the lock, versus a
    filesystem that cannot take the lock at all.
    """
    fstype = _filesystem_type(Path(db_path).parent)
    if fstype in _UNSAFE_LOCK_FILESYSTEMS:
        fs_verdict = (
            f"Detected filesystem: '{fstype}' — SQLite locking does not work "
            f"reliably there, so this is almost certainly the cause."
        )
    else:
        fs_verdict = (
            f"Detected filesystem: '{fstype}' — locking should work there, so "
            f"cause (1) or (3) is more likely."
        )

    return RuntimeError(
        f"Cannot write the output .d, SQLite reports the database as locked:\n"
        f"    {db_path}\n"
        f"\n"
        f"'database is locked' means another process holds a write lock on this "
        f"file, or the filesystem it lives on cannot take the lock. Check, in "
        f"this order:\n"
        f"\n"
        f"  1. Is an earlier timsim still running? A run holds this lock from "
        f"start to finish, and a backgrounded / Ctrl+Z-suspended run (or one "
        f"alive in another ssh or tmux session) keeps holding it:\n"
        f"         ps aux | grep timsim\n"
        f"         fuser -v '{db_path}'      # or: lsof '{db_path}'\n"
        f"  2. Is the output on a network or shared mount (NFS, CIFS/SMB, "
        f"VirtualBox vboxsf, 9p/virtiofs, sshfs)? {fs_verdict}\n"
        f"     Fix: point --save_path at local disk (e.g. ~/timsim_out) and "
        f"copy the finished .d to the share afterwards.\n"
        f"  3. Is this output folder left over from an earlier run? Use a fresh "
        f"experiment name, or an empty output directory.\n"
        f"\n"
        f"Original error: {exc}"
    )


def dedup_scan_tof(scan: NDArray, tof: NDArray, intensity: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Merge peaks that fall on the same (scan, tof) cell and return them sorted by scan, then tof.

    m/z -> TOF is not injective, so several simulated peaks can land on one TOF index; their
    intensities are summed. The result is byte-identical to the previous
    ``np.unique(np.stack((scan, tof)), axis=0)`` + ``np.lexsort`` implementation, but ~10-15x faster:
    the pair is packed into one uint64 key (scan in the high 32 bits) so a single 1-D sort orders by
    scan first and tof second, which is exactly the lexicographic order the TDF encoder needs.
    """
    scan = np.asarray(scan, dtype=np.uint32)
    tof = np.asarray(tof, dtype=np.uint32)
    intensity = np.asarray(intensity)
    key = (scan.astype(np.uint64) << np.uint64(32)) | tof.astype(np.uint64)
    unique_key, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse.ravel(), weights=intensity)
    out_scan = (unique_key >> np.uint64(32)).astype(np.uint32)
    out_tof = (unique_key & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return out_scan, out_tof, summed.astype(np.uint32)


class TDFWriter:
    def __init__(self, helper_handle: TimsDataset, path: str = "./", exp_name: str = "RAW.d", offset_bytes: int = 64, verbose: bool=False, use_rust_compression: bool=False, expect_existing: bool = False) -> None:

        self.path = Path(path)
        self.exp_name = exp_name
        self.full_path = Path(path) / exp_name
        self.position = 0
        self.binary_file = self.full_path / "analysis.tdf_bin"
        self.frame_meta_data = []
        self.conn = None
        self.helper_handle = helper_handle
        self.offset_bytes = offset_bytes
        self.verbose = verbose
        # Set by the ``from_existing`` builders, which legitimately reopen an
        # already-written .d; suppresses the re-used-output-folder warning.
        self.expect_existing = expect_existing
        # When True, the per-frame tdf_bin realdata is produced by the Rust
        # encoder (imspy_connector get_data_for_compression) instead of the
        # NumPy/Numba get_compressible_data. The Python dedup+lexsort over
        # (scan, tof) is kept either way so frame metadata stays correct; the
        # Rust encoder reproduces it byte-for-byte (see scripts/parity_tof_writer.py).
        # Also togglable globally via TIMSIM_RUST_COMPRESSION=1 so a stock
        # `timsim` run can exercise the Rust writer without code changes.
        self.use_rust_compression = use_rust_compression or os.environ.get(
            "TIMSIM_RUST_COMPRESSION", "0"
        ) not in ("0", "", "false", "False")

        self.__conn_native = None
        # Binary output handle; opened lazily on first write_frame() and kept open for the
        # whole run instead of open/append/close per frame (34k+ syscalls, costly on NFS).
        self._bin_fh = None
        self._setup_connections()

    def _setup_connections(self) -> None:
        # Create the directory and connect to DB
        self.full_path.mkdir(parents=True, exist_ok=True)
        db_path = self.full_path / "analysis.tdf"
        self.conn = sqlite3.connect(str(db_path), timeout=_SQLITE_BUSY_TIMEOUT_S)

        # Take the write lock up front: if the file is locked by another process
        # (or sits on a filesystem that cannot lock), fail here with a diagnostic
        # that names the cause, instead of deep inside a pandas `to_sql` stack.
        self._acquire_write_lock(db_path)
        self._warn_if_output_reused(db_path)

        # Create the tables for the analysis.tdf
        frame_ms_ms_info = self.helper_handle.get_table("FrameMsmsInfo")
        segments = self.helper_handle.get_table("Segments")

        try:
            last_frame = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            last_frame = self.helper_handle.meta_data.frame_id.max()

        segments.iloc[0, segments.columns.get_loc("LastFrame")] = last_frame

        # Save table to analysis.tdf
        try:
            self._create_table(self.conn, self.helper_handle.mz_calibration, "MzCalibration")
            self._create_table(self.conn, self.helper_handle.tims_calibration, "TimsCalibration")
            self._create_table(self.conn, self.helper_handle.global_meta_data_pandas, "GlobalMetadata")
            self._create_table(self.conn, frame_ms_ms_info, "FrameMsmsInfo")
            self._create_table(self.conn, segments, "Segments")
        except Exception as e:
            # pandas wraps the sqlite3 error, so match on the message rather than
            # the exception type; anything else propagates untouched.
            if "database is locked" in str(e).lower():
                raise _locked_database_error(db_path, e) from e
            raise

    def _acquire_write_lock(self, db_path) -> None:
        """Probe that this process can actually take SQLite's write lock.

        ``BEGIN IMMEDIATE`` reserves the database without writing anything, so a
        contended or unlockable file is detected before the first table is
        touched.
        """
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute("COMMIT")
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise _locked_database_error(db_path, e) from e
            raise

    def _warn_if_output_reused(self, db_path) -> None:
        """Warn when writing into a `.d` that an earlier run already populated.

        ``mkdir(exist_ok=True)`` plus ``to_sql(if_exists='replace')`` silently
        overwrites the metadata tables while the old ``analysis.tdf_bin`` blob
        file is kept, so a re-run into a used folder can produce a `.d` whose
        metadata and binary disagree.
        """
        if self.expect_existing:
            return
        try:
            existing = self.conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0]
        except sqlite3.Error:
            return
        if existing == 0:
            return
        warnings.warn(
            f"Output .d already contains an analysis.tdf with {existing} tables: "
            f"{db_path}. Re-running into an existing output folder overwrites the "
            f"metadata tables while the old analysis.tdf_bin is kept, which can "
            f"silently produce a corrupt .d. Use a fresh experiment name or an "
            f"empty output directory unless you are resuming (--resume / "
            f"--from_existing).",
            RuntimeWarning,
            stacklevel=3,
        )

        # Create the binary file and add the offset bytes
        # TODO: check if this is necessary
        with open(self.binary_file, "wb") as bin_file:
            bin_file.write(b'\x00' * self.offset_bytes)
            self.position = bin_file.tell()

        if self.verbose:
            print(f"Setting up TDF file meta data, created: {self.full_path}/analysis.tdf and {self.full_path}/analysis.tdf_bin")

    @staticmethod
    def _get_table(conn, table_name: str) -> pd.DataFrame:
        # Get a table as a pandas DataFrame
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    @staticmethod
    def _create_table(conn, table, table_name: str) -> None:
        # Create a table from a pandas DataFrame
        table.to_sql(table_name, conn, if_exists='replace', index=False)

    def mz_to_tof(self, frame_id, mzs):
        """Convert m/z values to TOF values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """

        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()

        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()

        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id

        return np.array(self.helper_handle.mz_to_tof(frame_id, mzs))

    def tof_to_mz(self, frame_id, tofs):
        """Convert TOF values to m/z values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """

        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.tof_to_mz(frame_id, tofs))

    def inv_mobility_to_scan(self, frame_id, inv_mobs):
        """Convert inverse mobility values to scan values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.inverse_mobility_to_scan(frame_id, inv_mobs))

    def scan_to_inv_mobility(self, frame_id, scans):
        """Convert scan values to inverse mobility values for a given frame using the helper handle.
        # CAUTION: This will use the calibration data from the reference handle.
        """
        try:
            max_ref_frame_id = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_ref_frame_id = self.helper_handle.meta_data.frame_id.max()
        if frame_id > max_ref_frame_id:
            frame_id = max_ref_frame_id
        return np.array(self.helper_handle.scan_to_inverse_mobility(frame_id, scans))

    def __repr__(self) -> str:
        return f'TDFWriter(path={self.path}, db_name={self.exp_name}, num_scans={self.helper_handle.num_scans}, ' \
               f'im_lower={self.helper_handle.im_lower}, im_upper={self.helper_handle.im_upper}, mz_lower={self.helper_handle.mz_lower}, ' \
               f'mz_upper={self.helper_handle.mz_upper})'

    def build_frame_meta_row(
            self,
            intensity: NDArray,
            frame: TimsFrame,
            scan_mode: int,
            frame_start_pos: int,
            only_frame_one: bool = False
    ):
        """Thin wrapper: derive the Frames statistics from an intensity array, then delegate."""
        return self._frame_meta_row(
            frame, scan_mode, frame_start_pos, only_frame_one,
            num_peaks=len(intensity),
            max_intensity=int(np.max(intensity)) if len(intensity) > 0 else 0,
            summed_intensity=int(np.sum(intensity)) if len(intensity) > 0 else 0,
        )

    def _frame_meta_row(
            self,
            frame: TimsFrame,
            scan_mode: int,
            frame_start_pos: int,
            only_frame_one: bool,
            num_peaks: int,
            max_intensity: int,
            summed_intensity: int,
    ):
        """Build a row for the frame meta data table from a TimsFrame object.
            Arguments:
                intensity: NDArray
                frame: TimsFrame object
                scan_mode: int
                frame_start_pos: int
                only_frame_one: bool
        """
        try:
            max_index = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_index = self.helper_handle.meta_data.frame_id.max()

        r = self.helper_handle.meta_data.iloc[0, :].copy()
        if not only_frame_one:
            # check for index out of bounds since ref data handle might not hold same number of frames
            if frame.frame_id > max_index:
                r = self.helper_handle.meta_data.iloc[max_index - 1, :].copy()
            else:
                r = self.helper_handle.meta_data.iloc[frame.frame_id - 1, :].copy()

        r.Id = frame.frame_id
        r.Time = frame.retention_time
        r.ScanMode = scan_mode
        r.MsMsType = frame.ms_type
        r.TimsId = frame_start_pos
        r.MaxIntensity = int(max_intensity)
        r.SummedIntensities = int(summed_intensity)
        r.NumScans = self.helper_handle.num_scans
        r.NumPeaks = int(num_peaks)

        return r

    def compress_frame(self, frame: TimsFrame, only_frame_one: bool = False) -> (NDArray, bytes):
        """Compress a single frame using zstd.
            Arguments:
                frame: TimsFrame object
                only_frame_one: bool

            Returns:
                bytes: intensities, compressed data
        """
        # either use frame 1 or the ref handle frame for writing of calibration data and call to conversion function
        i = 1 if only_frame_one else frame.frame_id

        try:
            max_index = self.helper_handle.meta_data.Id.max()
        except AttributeError as e:
            max_index = self.helper_handle.meta_data.frame_id.max()

        if frame.frame_id > max_index and not only_frame_one:
            i = max_index

        # transform mz and mobility to tof and scan
        tof = self.mz_to_tof(i, frame.mz).astype(np.uint32)
        scan = self.inv_mobility_to_scan(i, frame.mobility).astype(np.uint32)
        intensity = frame.intensity.astype(np.uint32)

        # Since mz -> tof is not bijective, merge duplicate (scan, tof) cells and sort by scan, tof.
        scan, tof, intensity = dedup_scan_tof(scan, tof, intensity)

        # get the real data as interleaved bytes (Rust encoder or NumPy/Numba)
        if self.use_rust_compression:
            real_data = ims.get_data_for_compression(
                tof.astype(np.uint32).tolist(),
                scan.astype(np.uint32).tolist(),
                intensity.astype(np.uint32).tolist(),
                int(self.helper_handle.num_scans),
            )
        else:
            real_data = get_compressible_data(tof, scan, intensity, self.helper_handle.num_scans)
        # compress the data
        return intensity, zstd.ZSTD_compress(bytes(real_data), 0)

    def write_frame(self, frame: TimsFrame, scan_mode: int, only_frame_one: bool = False) -> None:
        """Write a single frame to the binary file.
            Arguments:
                frame: TimsFrame object
                scan_mode: int
                only_frame_one: bool
        """
        intensity, compressed_data = self.compress_frame(frame, only_frame_one)

        self.frame_meta_data.append(
            self.build_frame_meta_row(
                intensity,
                frame,
                scan_mode,
                self.position,
                only_frame_one
        ))

        bin_file = self._binary_handle()
        bin_file.write(
            (len(compressed_data) + 8).to_bytes(4, "little", signed=False)
        )
        bin_file.write(int(self.helper_handle.num_scans).to_bytes(4, "little", signed=False))
        bin_file.write(compressed_data)
        self.position = bin_file.tell()

    def _conversion_frame_id(self, frame: TimsFrame, only_frame_one: bool) -> int:
        """Frame id whose calibration is used for this frame, clamped to the reference's range."""
        try:
            max_index = self.helper_handle.meta_data.Id.max()
        except AttributeError:
            max_index = self.helper_handle.meta_data.frame_id.max()
        if only_frame_one:
            return 1
        return int(max_index) if frame.frame_id > max_index else int(frame.frame_id)

    def write_frames(self, frames: List[TimsFrame], scan_mode: int, only_frame_one: bool = False,
                     num_threads: int = 4) -> None:
        """Batched counterpart of :meth:`write_frame`.

        Convert, dedup, interleave and compress every frame in one parallel Rust call, then append
        the payloads in order — only the append and the running byte offset have to stay sequential.
        Previously this ran per frame in Python, where the m/z -> TOF and 1/K0 -> scan conversions
        went through the Bruker SDK and so could not be parallelised; measured at 56 % of writer time.

        Falls back to the per-frame path if the parallel builder is unavailable (old connector, or a
        reference that carries no calibration tables for the SDK-free converter).
        """
        if len(frames) == 0:
            return

        builder = getattr(self.helper_handle, "build_compressed_frames", None)
        if builder is not None:
            conv_ids = [self._conversion_frame_id(f, only_frame_one) for f in frames]
            mz = [np.ascontiguousarray(f.mz, dtype=np.float64) for f in frames]
            mobility = [np.ascontiguousarray(f.mobility, dtype=np.float64) for f in frames]
            intensity = [np.ascontiguousarray(f.intensity, dtype=np.float64) for f in frames]
            try:
                built = builder(conv_ids, mz, mobility, intensity,
                                int(self.helper_handle.num_scans), 0, int(num_threads))
            except Exception as e:
                warnings.warn(f"batched frame writer unavailable ({e}); falling back to per-frame writing")
                built = None
            if built is not None:
                bin_file = self._binary_handle()
                num_scans_bytes = int(self.helper_handle.num_scans).to_bytes(4, "little", signed=False)
                for frame, (num_peaks, max_i, sum_i, data) in zip(frames, built):
                    self.frame_meta_data.append(
                        self._frame_meta_row(frame, scan_mode, self.position, only_frame_one,
                                             num_peaks, max_i, sum_i))
                    bin_file.write((len(data) + 8).to_bytes(4, "little", signed=False))
                    bin_file.write(num_scans_bytes)
                    bin_file.write(data)
                    self.position = bin_file.tell()
                return

        for frame in frames:
            self.write_frame(frame, scan_mode, only_frame_one)

    def _binary_handle(self):
        """Return the persistent append handle for analysis.tdf_bin, opening it on first use."""
        if self._bin_fh is None or self._bin_fh.closed:
            self._bin_fh = open(self.binary_file, "ab")
        return self._bin_fh

    def close_binary(self) -> None:
        """Flush and close the analysis.tdf_bin handle (idempotent). Called once all frames are
        written; a later write_frame() re-opens it transparently."""
        if self._bin_fh is not None and not self._bin_fh.closed:
            self._bin_fh.flush()
            self._bin_fh.close()
        self._bin_fh = None

    def get_frame_meta_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.frame_meta_data)

    def write_frame_meta_data(self) -> None:
        """Materialize accumulated Frames rows to the analysis.tdf SQLite.

        Enforces a duplicate-Id check before write: Bruker's `.tdf` schema
        relies on `Frames.Id` being unique and contiguous, and downstream
        readers that hash by Id (OpenTIMS, ionmaiden's `d2ms1`) throw
        `IndexError: unordered_map::at` on duplicates. We surface the
        violation here with a diagnostic so it's caught at the writer
        boundary rather than emitting a corrupt `.d` that fails late.

        See rustdf/src/data/dia.rs `sample_precursor_signal` /
        `sample_fragment_signal` for the upstream sampler fix that
        prevents the noise-pipeline from producing duplicate-Id rows in
        the first place. The check below is the defence-in-depth.
        """
        self.close_binary()
        meta_df = self.get_frame_meta_data()
        validate_frames_id_uniqueness(meta_df)
        self._create_table(self.conn, meta_df, "Frames")
        # Defence-in-depth: a UNIQUE INDEX makes any future direct INSERTs
        # to Frames also fail loudly on duplicate Id.
        try:
            self.conn.execute(
                'CREATE UNIQUE INDEX IF NOT EXISTS idx_frames_id_unique '
                'ON "Frames"(Id)'
            )
        except sqlite3.IntegrityError as e:  # pragma: no cover
            raise RuntimeError(
                f"Frames.Id UNIQUE INDEX creation failed: {e}. The "
                f"in-memory dedupe check above missed something — please "
                f"file a bug with the analysis.tdf path."
            ) from e

    def write_calibration_info(self, mz_standard_deviation_ppm: float = 0.15) -> None:
        try:
            table = self.helper_handle.get_table("CalibrationInfo")
            table.iloc[5].Value = str(mz_standard_deviation_ppm)
            self._create_table(self.conn, table, "CalibrationInfo")

        except Exception as e:
            print(f"Error writing calibration info table: {e}")

    def write_pasef_frame_ms_ms_info(self) -> None:
        try:
            self._create_table(self.conn, self.helper_handle.get_table("PasefFrameMsMsInfo"), "PasefFrameMsMsInfo")
        except Exception as e:
            print(f"Error writing PasefFrameMsMsInfo table: {e}. In most cases, this is not a problem, since the table is empty in DIA mode.")

    def write_prm_frame_ms_ms_info(self) -> None:
        try:
            self._create_table(self.conn, self.helper_handle.get_table("PrmFrameMsMsInfo"), "PrmFrameMsMsInfo")
        except Exception as e:
            print(f"Error writing PrmFrameMsMsInfo table: {e}")

    def write_dia_ms_ms_info(self, dia_ms_ms_info: pd.DataFrame) -> None:
        out = dia_ms_ms_info.rename(columns={
            'frame': 'Frame',
            'window_group': 'WindowGroup',
        })

        self._create_table(self.conn, out, "DiaFrameMsMsInfo")

    def write_precursor_table(self, precursor_table: pd.DataFrame, id_mapping_table: pd.DataFrame = None) -> dict:
        """Write the Precursors table with proper vendor schema.

        Args:
            precursor_table: DataFrame with precursor data
            id_mapping_table: Optional DataFrame with 'ion_id' and 'tdf_precursor_id' columns.
                             If None, creates sequential mapping from existing IDs.

        Returns:
            dict: Mapping from old IDs to new sequential IDs (for use with pasef_meta table)
        """
        out = precursor_table.rename(columns={
            'id': 'Id',
            'largest_peak_mz': 'LargestPeakMz',
            'average_mz': 'AverageMz',
            'monoisotopic_mz': 'MonoisotopicMz',
            'charge': 'Charge',
            'scan_number': 'ScanNumber',
            'intensity': 'Intensity',
            'parent': 'Parent',
        })

        # Create ID mapping from table or generate new one
        if id_mapping_table is not None:
            id_mapping = dict(zip(id_mapping_table['ion_id'], id_mapping_table['tdf_precursor_id']))
        else:
            old_ids = out['Id'].unique()
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(old_ids), start=1)}

        # Deduplicate: each precursor ID should only appear once in the Precursors table
        # Keep the first occurrence (or could aggregate, but first is simpler)
        out_unique = out.drop_duplicates(subset=['Id'], keep='first')

        # Create table with proper vendor schema
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS Precursors")
        cursor.execute("""
            CREATE TABLE Precursors (
                Id INTEGER PRIMARY KEY,
                LargestPeakMz REAL NOT NULL,
                AverageMz REAL NOT NULL,
                MonoisotopicMz REAL,
                Charge INTEGER,
                ScanNumber REAL NOT NULL,
                Intensity REAL NOT NULL,
                Parent INTEGER,
                FOREIGN KEY(Parent) REFERENCES Frames(Id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS PrecursorsParentIndex ON Precursors (Parent)")

        # Insert data with remapped IDs (only unique precursors)
        for _, row in out_unique.iterrows():
            new_id = id_mapping.get(int(row['Id']), int(row['Id']))
            cursor.execute("""
                INSERT INTO Precursors (Id, LargestPeakMz, AverageMz, MonoisotopicMz, Charge, ScanNumber, Intensity, Parent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_id,
                float(row['LargestPeakMz']),
                float(row['AverageMz']),
                float(row['MonoisotopicMz']) if pd.notna(row['MonoisotopicMz']) else None,
                int(row['Charge']) if pd.notna(row['Charge']) else None,
                float(row['ScanNumber']),
                float(row['Intensity']),
                int(row['Parent']) if pd.notna(row['Parent']) else None
            ))
        self.conn.commit()

        return id_mapping

    def write_pasef_meta_table(self, pasef_meta_table: pd.DataFrame, id_mapping: dict = None) -> None:
        """Write the PasefFrameMsMsInfo table with proper vendor schema.

        Args:
            pasef_meta_table: DataFrame with PASEF metadata
            id_mapping: Optional dict mapping old precursor IDs to new sequential IDs.
                        If None, uses IDs as-is.
        """
        out = pasef_meta_table.rename(columns={
            'frame': 'Frame',
            'scan_start': 'ScanNumBegin',
            'scan_end': 'ScanNumEnd',
            'isolation_mz': 'IsolationMz',
            'isolation_width': 'IsolationWidth',
            'collision_energy': 'CollisionEnergy',
            'precursor': 'Precursor',
        })

        # Create table with proper vendor schema (WITHOUT ROWID for performance)
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS PasefFrameMsMsInfo")
        cursor.execute("""
            CREATE TABLE PasefFrameMsMsInfo (
                Frame INTEGER NOT NULL,
                ScanNumBegin INTEGER NOT NULL,
                ScanNumEnd INTEGER NOT NULL,
                IsolationMz REAL NOT NULL,
                IsolationWidth REAL NOT NULL,
                CollisionEnergy REAL NOT NULL,
                Precursor INTEGER,
                PRIMARY KEY(Frame, ScanNumBegin),
                FOREIGN KEY(Frame) REFERENCES Frames(Id),
                FOREIGN KEY(Precursor) REFERENCES Precursors(Id)
            ) WITHOUT ROWID
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS PasefFrameMsMsInfoPrecursorIndex ON PasefFrameMsMsInfo (Precursor)")

        # Insert data with remapped precursor IDs if mapping provided
        for _, row in out.iterrows():
            precursor_id = int(row['Precursor']) if pd.notna(row['Precursor']) else None
            if id_mapping is not None and precursor_id is not None:
                precursor_id = id_mapping.get(precursor_id, precursor_id)

            cursor.execute("""
                INSERT INTO PasefFrameMsMsInfo (Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, Precursor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row['Frame']),
                int(row['ScanNumBegin']),
                int(row['ScanNumEnd']),
                float(row['IsolationMz']),
                float(row['IsolationWidth']),
                float(row['CollisionEnergy']),
                precursor_id
            ))
        self.conn.commit()

    def write_dia_ms_ms_windows(self, dia_ms_ms_windows: pd.DataFrame) -> None:
        out = dia_ms_ms_windows.rename(columns={
            'window_group': 'WindowGroup',
            'scan_start': 'ScanNumBegin',
            'scan_end': 'ScanNumEnd',
            'isolation_mz': 'IsolationMz',
            'isolation_width': 'IsolationWidth',
            'collision_energy': 'CollisionEnergy',
        })

        self._create_table(self.conn, out, "DiaFrameMsMsWindows")

