"""Step-1 parity check: Python writer vs fixed Rust writer produce identical
Bruker tdf_bin realdata for the same raw (scan, tof, intensity) frame input.

Python path mirrors imspy_simulation.tdf.TimsTofWriter.compress_frame's
preprocessing (dedup (scan,tof) summing intensities + lexsort((tof,scan)))
followed by get_compressible_data. Rust path is the connector
get_data_for_compression, which now performs the same dedup+sort internally.
"""
import numpy as np
import imspy_connector
from imspy_simulation.utility import get_compressible_data

ims = imspy_connector.py_dataset


def python_writer_bytes(scan, tof, inten, num_scans):
    # --- replicate compress_frame preprocessing ---
    scan_tof = np.stack((scan, tof), axis=1)
    uniq, inv = np.unique(scan_tof, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    summed = np.bincount(inv, weights=inten)
    us, ut = uniq[:, 0], uniq[:, 1]
    sidx = np.lexsort((ut, us))
    ps = us[sidx].astype(np.uint32)
    pt = ut[sidx].astype(np.uint32)
    pi = summed[sidx].astype(np.uint32)
    return np.asarray(get_compressible_data(pt, ps, pi, num_scans), dtype=np.uint8)


def rust_writer_bytes(scan, tof, inten, num_scans):
    out = ims.get_data_for_compression(
        tof.astype(np.uint32).tolist(),
        scan.astype(np.uint32).tolist(),
        inten.astype(np.uint32).tolist(),
        int(num_scans),
    )
    return np.frombuffer(bytes(out), dtype=np.uint8)


def make_case(rng, n, num_scans, max_tof):
    scan = rng.integers(0, num_scans, n).astype(np.uint32)
    tof = rng.integers(1, max_tof, n).astype(np.uint32)
    inten = rng.integers(1, 1000, n).astype(np.uint32)
    return scan, tof, inten


def main():
    num_scans = 174
    cases = []
    rng = np.random.default_rng(0)
    # random, duplicate-heavy, unsorted
    cases.append(("random-1k", make_case(rng, 1000, num_scans, 400_000)))
    cases.append(("random-50k", make_case(rng, 50_000, num_scans, 400_000)))
    # force many duplicate (scan,tof) collisions via tiny tof range
    cases.append(("dup-heavy", make_case(rng, 20_000, num_scans, 500)))
    # single scan
    s = np.zeros(500, np.uint32)
    t = rng.integers(1, 400_000, 500).astype(np.uint32)
    i = rng.integers(1, 1000, 500).astype(np.uint32)
    cases.append(("single-scan", (s, t, i)))
    # already sorted+unique
    t2 = np.sort(rng.choice(np.arange(1, 400_000), 800, replace=False)).astype(np.uint32)
    s2 = np.sort(rng.integers(0, num_scans, 800)).astype(np.uint32)
    i2 = rng.integers(1, 1000, 800).astype(np.uint32)
    cases.append(("pre-sorted", (s2, t2, i2)))

    all_ok = True
    for name, (scan, tof, inten) in cases:
        py = python_writer_bytes(scan, tof, inten, num_scans)
        ru = rust_writer_bytes(scan, tof, inten, num_scans)
        ok = py.shape == ru.shape and np.array_equal(py, ru)
        all_ok &= ok
        status = "OK " if ok else "FAIL"
        extra = ""
        if not ok:
            extra = f"  py_len={py.size} ru_len={ru.size}"
            if py.size == ru.size:
                diff = np.flatnonzero(py != ru)
                extra += f" first_diff@{diff[0]} py={py[diff[0]]} ru={ru[diff[0]]} ({diff.size} bytes differ)"
        print(f"[{status}] {name:12s} n={scan.size:6d} bytes={ru.size}{extra}")

    print("\nPARITY:", "ALL MATCH ✅" if all_ok else "MISMATCH ❌")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
