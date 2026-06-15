"""Multi-vendor TimSim benchmark: consolidate per-vendor ground-truth eval results
into one side-by-side recall / FDP table.

TimSim simulates DIA data for four instrument families, each searched by an
independent engine and scored against the known ground truth:

    Bruker timsTOF (.d)   Thermo Astral/Orbitrap (.raw)
    SCIEX ZenoTOF (mzML)  Waters Synapt XS SONAR (mzML)

This module does NOT run simulators or search engines (those are external and
host-specific). It reads the JSON emitted by ``groundtruth_eval`` (one per vendor)
and renders a comparison table, so the multi-vendor validation story lives in one
reproducible place. Vendors with no fresh eval JSON can be carried as ``recorded``
entries (a prior validated run) with explicit provenance, kept distinct from
freshly-evaluated rows by the ``source`` column.

Per-vendor generation recipe (each produces one eval JSON consumed here)::

    1. timsim <vendor>.toml                       # -> .d/.raw/.mzML + synthetic_data.db
    2. <engine> on the output                     # DiaNN (.d/.mzML), alphaDIA (.raw/.mzML)
    3. python -m imspy_simulation.timsim.groundtruth_eval \\
         --db <synthetic_data.db> --report <engine report> \\
         --engine <diann|alphadia> --precursor-fdp --out <vendor>.json

A manifest (JSON) lists the vendors to consolidate::

    {
      "q": 0.01,
      "vendors": [
        {"instrument": "sciex_zenotof",  "engine": "alphadia", "eval_json": "/path/sciex.json"},
        {"instrument": "waters_synapt_xs","engine": "diann",   "eval_json": "/path/waters.json"},
        {"instrument": "bruker_timstof", "engine": "diann", "source": "recorded",
         "recorded": {"observable": 0, "ids": 0, "recall_pct": 0.0,
                      "fdp_backbone": 0.0, "fdp_peptidoform": 0.0},
         "note": "prior validated run <ref>"}
      ]
    }

    python -m imspy_simulation.timsim.benchmark.multi_vendor --manifest m.json --out table.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

# Display metadata per supported instrument (output format the render produces).
VENDOR_META = {
    "bruker_timstof": ("Bruker timsTOF", ".d (TDF)"),
    "orbitrap_astral": ("Thermo Astral", ".raw"),
    "orbitrap_exploris": ("Thermo Orbitrap", ".raw"),
    "sciex_zenotof": ("SCIEX ZenoTOF", "mzML"),
    "waters_synapt_xs": ("Waters Synapt XS SONAR", "mzML"),
}


@dataclass
class VendorRow:
    instrument: str
    display: str
    output: str
    engine: str
    observable: Optional[int]
    ids: Optional[int]
    recall_pct: Optional[float]
    fdp_backbone: Optional[float]
    fdp_peptidoform: Optional[float]
    source: str
    note: str = ""


def summarize_eval(eval_json_path: str, q: float = 0.01) -> dict:
    """Extract the recall / FDP summary at q-threshold ``q`` from a groundtruth_eval JSON.

    FDR-threshold semantics: report the curve point with the LARGEST ``q`` that is still
    ``<= q`` (i.e. the most IDs that pass at or below the requested FDR) — never a point
    above the threshold, which would overstate IDs/recall. If the curve starts above
    ``q`` (no point qualifies), fall back to the smallest-``q`` point; the returned ``q``
    reflects which point was actually used.
    """
    with open(eval_json_path) as fh:
        data = json.load(fh)
    curve = data.get("curve") or []
    if not curve:
        raise ValueError(f"{eval_json_path}: no 'curve' in eval JSON")
    at_or_below = [r for r in curve if float(r["q"]) <= q + 1e-12]
    if at_or_below:
        point = max(at_or_below, key=lambda r: float(r["q"]))
    else:
        point = min(curve, key=lambda r: float(r["q"]))
    observable = data.get("rendered_observable_precursors")
    ids = point.get("n_precursors")
    recall = (100.0 * ids / observable) if (observable and ids is not None) else None
    return {
        "observable": observable,
        "ids": ids,
        "recall_pct": recall,
        "fdp_backbone": point.get("fdp_backbone"),
        "fdp_peptidoform": point.get("fdp_peptidoform_genuine"),
        "q": point.get("q"),
        "engine": data.get("engine"),
    }


def build_rows(manifest: dict) -> list[VendorRow]:
    q = float(manifest.get("q", 0.01))
    rows: list[VendorRow] = []
    for v in manifest.get("vendors", []):
        instrument = v["instrument"]
        display, output = VENDOR_META.get(instrument, (instrument, "?"))
        engine = v.get("engine")  # may be None -> fall back to the eval JSON's engine
        if v.get("eval_json"):
            s = summarize_eval(v["eval_json"], q=q)
            rows.append(VendorRow(
                instrument=instrument, display=display, output=output,
                engine=engine or s.get("engine") or "?",
                observable=s["observable"], ids=s["ids"], recall_pct=s["recall_pct"],
                fdp_backbone=s["fdp_backbone"], fdp_peptidoform=s["fdp_peptidoform"],
                source=v.get("source", "live"), note=v.get("note", ""),
            ))
        elif v.get("recorded"):
            r = v["recorded"]
            rows.append(VendorRow(
                instrument=instrument, display=display, output=output, engine=engine or "?",
                observable=r.get("observable"), ids=r.get("ids"),
                recall_pct=r.get("recall_pct"),
                fdp_backbone=r.get("fdp_backbone"), fdp_peptidoform=r.get("fdp_peptidoform"),
                source=v.get("source", "recorded"), note=v.get("note", ""),
            ))
        else:
            raise ValueError(f"vendor {instrument!r} has neither 'eval_json' nor 'recorded'")
    return rows


def _cell(s) -> str:
    """Sanitize a string for a Markdown table cell (escape pipes, flatten newlines)."""
    return str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1f}%"


def _fmt_fdp(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def render_markdown(rows: list[VendorRow], q: float = 0.01) -> str:
    head = (
        f"# TimSim multi-vendor benchmark (q = {q:g})\n\n"
        "| Vendor | Output | Engine | Observable | IDs @FDR | Recall | FDP (backbone) | FDP (peptidoform) | Source |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---|\n"
    )
    body = "".join(
        f"| {_cell(r.display)} | {_cell(r.output)} | {_cell(r.engine)} | "
        f"{'—' if r.observable is None else r.observable} | "
        f"{'—' if r.ids is None else r.ids} | {_fmt_pct(r.recall_pct)} | "
        f"{_fmt_fdp(r.fdp_backbone)} | {_fmt_fdp(r.fdp_peptidoform)} | {_cell(r.source)} |\n"
        for r in rows
    )
    notes = [f"- **{_cell(r.display)}**: {_cell(r.note)}" for r in rows if r.note]
    tail = ("\n" + "\n".join(notes) + "\n") if notes else ""
    return head + body + tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="path to the vendor manifest JSON")
    ap.add_argument("--out", help="optional path to write the Markdown table")
    ap.add_argument("--json-out", help="optional path to write the consolidated JSON")
    args = ap.parse_args(argv)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    q = float(manifest.get("q", 0.01))
    rows = build_rows(manifest)
    table = render_markdown(rows, q=q)
    print(table)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(table)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"q": q, "vendors": [asdict(r) for r in rows]}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
