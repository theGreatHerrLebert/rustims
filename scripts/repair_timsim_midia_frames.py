"""Salvage TIMSIM-MIDIA .d datasets corrupted by the noise-sampler ms_type bug.

The TimSim midiA writer composes simulated frames with reference noise via
`add_real_data_noise_to_frames` → `simulated + noise`. The noise frames
returned by `sample_precursor_signal` carry `MsType::Unknown` whenever the
reference's MS1 frame was empty (rustdf/src/data/handle.rs:692 falls back
to `TimsFrame::default()`). `TimsFrame::Add` correctly collapses ms_type to
Unknown on type mismatch, and a separate code path leaks the noise's
default `frame_id=1, retention_time=0.0` metadata into the writer.

Result on TIMSIM-MIDIA-150K-24-12.d:
  - 115 frames carry the real simulated MS1 binary content
  - all 115 share corrupted metadata: Id=1, Time=0.0, MsMsType=-1
  - the canonical MS1 IDs they should have occupied (stride-17, missing
    from Frames) are 18, 35, 52, ..., 34086

This script repairs the Frames table in place (after backing up
analysis.tdf to analysis.tdf.bak) by re-assigning each broken frame's
Id / Time / MsMsType to its rightful slot. The binary tdf_bin is not
touched — only the metadata is wrong; TimsId still points to a valid
compressed payload. The salvage makes the .d readable by OpenTIMS and
any downstream tool that hashes by Frame.Id.

Usage:
    python repair_timsim_midia_frames.py /path/to/TIMSIM-MIDIA-...d
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("d_path", type=Path, help="Path to the .d directory")
    ap.add_argument("--ms-type-replacement", type=int, default=0,
                    help="MsMsType to assign to repaired frames "
                         "(default 0 = Precursor, since the broken frames "
                         "occupy MS1 cycle slots).")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip the analysis.tdf -> analysis.tdf.bak backup. "
                         "Strongly discouraged.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing.")
    args = ap.parse_args(argv)

    tdf = args.d_path / "analysis.tdf"
    if not tdf.exists():
        print(f"error: {tdf} not found", file=sys.stderr)
        return 2

    if not args.no_backup and not args.dry_run:
        backup = tdf.with_suffix(".tdf.bak")
        if backup.exists():
            print(f"error: backup already exists at {backup}; remove or rename first",
                  file=sys.stderr)
            return 2
        shutil.copy2(tdf, backup)
        print(f"backed up {tdf} -> {backup}")

    con = sqlite3.connect(str(tdf))
    cur = con.cursor()

    # 1. Detect the broken frames + the missing canonical IDs.
    broken = cur.execute(
        "SELECT TimsId FROM Frames WHERE MsMsType = -1 ORDER BY TimsId"
    ).fetchall()
    broken_tims_ids = [r[0] for r in broken]
    n_broken = len(broken_tims_ids)

    max_id = cur.execute("SELECT MAX(Id) FROM Frames").fetchone()[0]
    # Canonical IDs that need to be assignable to the 115 broken rows are
    # those NOT held by any *clean* row. The broken rows themselves all
    # carry Id=1 (which makes a naive "x NOT IN Id" miss Id=1 because the
    # broken rows shadow the real Frame 1 slot — none of them is the
    # legit Frame 1 either).
    cur.execute(
        f"""
        WITH RECURSIVE seq(x) AS (
            SELECT 1 UNION ALL SELECT x + 1 FROM seq WHERE x < {max_id}
        )
        SELECT x FROM seq
        WHERE x NOT IN (SELECT Id FROM Frames WHERE MsMsType != -1)
        ORDER BY x
        """
    )
    missing_ids = [r[0] for r in cur.fetchall()]
    n_missing = len(missing_ids)

    print(f"broken frames (Id=1, Time=0, MsMsType=-1): {n_broken}")
    print(f"missing canonical IDs in [1, {max_id}]:    {n_missing}")
    if n_broken != n_missing:
        print(
            f"error: cannot repair — broken count ({n_broken}) does not equal "
            f"missing-canonical-ID count ({n_missing}). The dataset may have "
            f"a different corruption shape than this script handles.",
            file=sys.stderr,
        )
        return 3

    # 2. Derive rt_cycle_length from any clean row that has a sensible Time.
    rt_per_id, max_clean_id = cur.execute(
        "SELECT Time / Id, Id FROM Frames "
        "WHERE Time > 0 AND MsMsType >= 0 ORDER BY Id DESC LIMIT 1"
    ).fetchone()
    print(f"derived rt_cycle_length = {rt_per_id:.6f} s/frame "
          f"(from Time={rt_per_id * max_clean_id:.4f}, Id={max_clean_id})")

    # 3. Pair each broken row (sorted by TimsId = chronological binary
    #    order) with the corresponding missing-canonical-ID (sorted
    #    ascending = chronological cycle order). This is the only
    #    self-consistent mapping under the simulator's
    #    "time = frame_id * rt_cycle_length" invariant.
    updates: list[tuple[int, float, int, int]] = []  # (new_id, new_time, ms_type, tims_id)
    for tims_id, new_id in zip(broken_tims_ids, missing_ids):
        new_time = new_id * rt_per_id
        updates.append((new_id, new_time, args.ms_type_replacement, tims_id))

    # 4. Sanity-check: every new_id must NOT already exist in *clean*
    #    rows. The broken rows we are about to overwrite all carry Id=1,
    #    which is also a target ID — that overlap is fine because we
    #    update by TimsId, not by Id.
    occupied = set(
        r[0] for r in cur.execute(
            "SELECT Id FROM Frames WHERE MsMsType != -1"
        ).fetchall()
    )
    collisions = [u[0] for u in updates if u[0] in occupied]
    if collisions:
        print(
            f"error: {len(collisions)} candidate new IDs already exist in "
            f"Frames (first 5: {collisions[:5]}). Aborting.",
            file=sys.stderr,
        )
        return 4

    # 5. Preview what will change.
    print()
    print("sample of planned updates (first 3 + last 3):")
    for u in updates[:3] + updates[-3:]:
        print(f"  TimsId={u[3]:>12}  ->  Id={u[0]:>6}  Time={u[1]:.4f}  MsMsType={u[2]}")

    if args.dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    # 6. Apply in a single transaction; we update by TimsId since it's the
    #    only column that's still unique on the broken rows.
    cur.executemany(
        "UPDATE Frames SET Id = ?, Time = ?, MsMsType = ? WHERE TimsId = ?",
        updates,
    )
    con.commit()
    print(f"\nupdated {cur.rowcount} rows; vacuuming...")
    con.execute("VACUUM")

    # 7. Verify the post-condition: contiguous unique IDs, no MsMsType=-1.
    n_neg = cur.execute("SELECT COUNT(*) FROM Frames WHERE MsMsType = -1").fetchone()[0]
    n_total = cur.execute("SELECT COUNT(*) FROM Frames").fetchone()[0]
    n_distinct = cur.execute("SELECT COUNT(DISTINCT Id) FROM Frames").fetchone()[0]
    new_max = cur.execute("SELECT MAX(Id) FROM Frames").fetchone()[0]
    print(f"post-repair: COUNT={n_total}, DISTINCT(Id)={n_distinct}, "
          f"MAX(Id)={new_max}, MsMsType=-1: {n_neg}")
    if n_neg or n_distinct != n_total:
        print("warning: post-condition not satisfied — re-inspect manually.",
              file=sys.stderr)
        return 5

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
