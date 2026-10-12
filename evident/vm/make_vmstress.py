#!/usr/bin/env python3
"""Predictor-swap stress replays (rustims evident/vm/stress_settings.yaml), one per setting x baseline seed.

Each config is the baseline 250k config.toml with from_existing pointed at the prepared slim DB
(vm_stress_prepare.py) and the replicate-variation knobs removed, so the replay adds no jitter of its own.
Keys in REQUIRED are set explicitly. Everything else, sample_seed included, is the baseline's.
"""
import json, os, sys

ROOT = "/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/TIMSIM-REVISION"
BASE = f"{ROOT}/searches/vm/sim-baseline"
OUT = f"{ROOT}/searches/vm/sim-stress"
SEEDS = {"jitter-250k-r1": 4101, "jitter-250k-r2": 4102, "jitter-250k-r3": 4103}  # stress_settings.yaml
SETTINGS = ("replay-control", "rt-chronologer", "im-std", "joint")
DROP = ("rt_variation_std", "ion_mobility_variation_std", "intensity_variation_std")
# Replay must not re-predict, re-rescale or transform what was prepared, and must log what it does.
REQUIRED = {"re_scale_rt": "false", "phospho_mode": "false", "proteome_mix": "false", "silent_mode": "false",
            "rt_model": '""', "ccs_model": '""', "koina_rt_model": '""', "from_findings": "false",
            "acquisition_type": '"DIA"'}


def main():
    queue = []
    for rid, seed in SEEDS.items():
        plan = json.load(open(f"{BASE}/{rid}/plan.json"))
        if plan["seed"] != seed or plan["n_peptides"] != 250000:
            sys.exit(f"{rid}: plan seed/complexity {plan['seed']}/{plan['n_peptides']} != {seed}/250000")
        lines = open(f"{BASE}/{rid}/config.toml").read().splitlines()
        for s in SETTINGS:
            sid = f"{rid}-{s}"
            d = f"{OUT}/{sid}"
            prep = f"{OUT}/prepared/{rid}/{s}"
            over = {"from_existing": "true", "existing_path": json.dumps(prep),
                    "experiment_name": json.dumps(sid), "save_path": json.dumps(f"{d}/out")}
            seen, cfg, vals = set(), [], {}
            for ln in lines:
                k = ln.split("=", 1)[0].strip()
                if k in DROP or k in REQUIRED:
                    continue
                if k in over:
                    ln, _ = f"{k} = {over[k]}", seen.add(k)
                cfg.append(ln)
                if "=" in ln:
                    vals[k] = ln.split("=", 1)[1].strip()
            cfg += [f"{k} = {v}" for k, v in REQUIRED.items()]
            if vals.get("sample_seed") != str(seed):
                sys.exit(f"{rid}: config sample_seed {vals.get('sample_seed')} != {seed}")
            missing = set(over) - seen
            if missing:
                sys.exit(f"{rid}: baseline config lacks {missing}")
            os.makedirs(d, exist_ok=True)
            open(f"{d}/config.toml", "w").write("\n".join(cfg) + "\n")
            json.dump(dict(plan, run_id=sid, baseline=rid, setting=s, existing_path=prep),
                      open(f"{d}/plan.json", "w"), indent=1)
            queue.append(sid)
    open(f"{OUT}/queue.txt", "w").write("\n".join(queue) + "\n")
    print(f"{len(queue)} replays -> {OUT}/queue.txt")


if __name__ == "__main__":
    main()
