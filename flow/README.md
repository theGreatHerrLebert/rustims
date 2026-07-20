# timsim2 necroflow pipeline

`timsim_flow.py` defines the timsim2 simulation as a content-addressed necroflow DAG.

**Pipeline factory functions** (what you call to build a `Pipeline`):
- `timsim_pipeline(cfg, sample_id)` — the timsTOF `.d` pipeline (feature space → legacy `simulate`).
- `timsim_thermo_pipeline(cfg, sample_id)` — the Thermo `.raw` pipeline (feature space → `fragments →
  spectra → render_thermo`, authored into a real Astral/Orbitrap template). No IMS; the fragment model
  (`frag_model`, e.g. `koina:Prosit_2020_intensity_HCD`) and the `.raw` template are node parameters, so
  the device×method matrix is a fan-out and the feature space is computed once (content-addressed).

The `@r.command(...)` rules + `NodeType` classes above the factories are the DAG's artifacts/commands.

Run/graph (Rust bins on `TIMSIM_BIN`, Python console-scripts on PATH):
```
export TIMSIM_BIN=../target/release
python timsim_flow.py --dry-run                                   # timsTOF .d, resolve only
python timsim_flow.py --thermo-template <astral_or_orbi>.raw \
    --frag-model koina:Prosit_2020_intensity_HCD --outdir <dir>   # execute Thermo .raw
python timsim_flow.py --graph dag.png ...                         # plot
```
The `.toml` specs (hye/mods/design/v1) carry local paths — adjust `reference_path`/`fasta_path` for your box.
