"""vm.protocol.matched-complexity: pick the sim complexity whose depth matches real 10 ng.

Reads plain-library reports only, and must run before any FDP is computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vm_common import accepted_from_report, check_cohort, complexity_of, dump, is_entrapment, read_pairs

POOR_MATCH_FACTOR = 1.5


def target_count(report: Path, pairs: dict[str, str]) -> int:
    acc = accepted_from_report(report)
    return int((~is_entrapment(acc["stripped"], pairs, acc["protein"])).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True, help="searches/vm/recipe-v1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pairs = read_pairs(a.recipe / "entrap" / "human_entrap_pep.txt")

    real_runs, sim_runs = check_cohort(a.recipe)
    real = {r: target_count(a.recipe / "real10" / r / "plain" / "report.parquet", pairs) for r in real_runs}
    sim: dict[int, dict[str, int]] = {}
    for r in sim_runs:
        sim.setdefault(complexity_of(r), {})[r] = target_count(a.recipe / "sim" / r / "plain" / "report.parquet", pairs)

    real_med = float(np.median(list(real.values())))
    meds = {c: float(np.median(list(v.values()))) for c, v in sim.items()}
    if real_med <= 0 or min(meds.values()) <= 0:
        raise SystemExit("zero identification depth, the log-scale match is undefined")
    dist = {c: abs(np.log(m) - np.log(real_med)) for c, m in meds.items()}
    best = min(dist.values())
    matched = max(c for c, d in dist.items() if d == best)  # exact ties only; ties go to the higher complexity
    ratio = float(np.median(list(sim[matched].values())) / real_med)
    dump({
        "rule": "vm.protocol.matched-complexity",
        "real_targets": real,
        "real_median": real_med,
        "sim_targets": {str(c): v for c, v in sim.items()},
        "log_distance": {str(c): float(d) for c, d in dist.items()},
        "matched_complexity": matched,
        "matched_runs": sorted(sim[matched]),
        "sim_over_real": ratio,
        "poor_match": not (1 / POOR_MATCH_FACTOR <= ratio <= POOR_MATCH_FACTOR),
    }, a.out)
    print(f"matched complexity {matched} (sim/real depth {ratio:.2f})")


if __name__ == "__main__":
    main()
