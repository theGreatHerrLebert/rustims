#!/bin/bash
#SBATCH --account=ki-proanagi
#SBATCH --partition=ki-smallcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=vm-stage-c
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/logs/vm-stage-c-%j.log
# Stage C of the validity-matrix analysis on the baseline data: marginals, score distributions and protein-level
# entrapment (claims/marginals.yaml, scores.yaml, protein.yaml), then the descriptive diagnostics (vm_diagnose.py,
# reported only). Reads no stress-row output. Markers: VM_STAGE_C_DONE / VM_STAGE_C_FAIL <step>.
set -uo pipefail
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
R=$M/TIMSIM-REVISION/searches/vm
PY=$M/vmenv/bin/python
EV=${EV:-$M/evident-vm}
OUT=$R/analysis-v1
fail() { echo "VM_STAGE_C_FAIL $1 $(date -u +%FT%TZ)"; exit 1; }
cd $EV || fail code
echo "VM_STAGE_C_START $(date -u +%FT%TZ) code=$(cat $EV/COMMIT 2>/dev/null)"
[ -s $OUT/matched_complexity.json ] || fail "stage A missing"
$PY vm_marginals.py --recipe $R/recipe-v1 --raw $R/raw-v1 --out $OUT || fail marginals
$PY vm_scores.py --recipe $R/recipe-v1 --fragpipe $R/fragpipe-v1 --out $OUT || fail scores
$PY vm_protein.py --recipe $R/recipe-v1 --protein $R/recipe-v1-protein --sim-root $R/sim-baseline --out $OUT || fail protein
$PY vm_diagnose.py --recipe $R/recipe-v1 --sim-root $R/sim-baseline --out $OUT || fail diagnose
echo "VM_STAGE_C_DONE $(date -u +%FT%TZ)"
