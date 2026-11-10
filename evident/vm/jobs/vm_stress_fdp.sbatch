#!/bin/bash
#SBATCH --account=ki-proanagi
#SBATCH --partition=ki-smallcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=vm-stress-fdp
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/logs/vm-stress-fdp-%j.log
# Truth/entrapment FDP of finished stress rows, exactly the evidence commands frozen in claims/stress.yaml.
# ROWS (default "predictor-swap rt-spread") names the rows; each writes analysis-v1/results/stress-<row>.json.
# Each row's searches must be done (VM_SIMSEARCH_DONE) before this runs. Markers VM_STRESS_FDP_DONE <row> /
# VM_STRESS_FDP_FAIL <row> <claim>.
set -uo pipefail
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
T=$M/TIMSIM-REVISION/searches/vm
PY=$M/vmenv/bin/python
EV=${EV:?EV=$M/evident-vm-<commit>}
ROWS=${ROWS:-"predictor-swap rt-spread"}
cd $EV || { echo "VM_STRESS_FDP_FAIL code $(date -u +%FT%TZ)"; exit 1; }
(sha256sum --quiet -c SHA256SUMS) || { echo "VM_STRESS_FDP_FAIL deployed_files_changed $(date -u +%FT%TZ)"; exit 1; }
echo "VM_STRESS_FDP_START $(date -u +%FT%TZ) code=$(cat COMMIT) rows=$ROWS"
rc=0
for ROW in $ROWS; do
  for KIND in fdp-bound fdp-delta; do
    C=vm.$ROW.$KIND.diann
    $PY vm_truth_fdp.py --mode stress --claim $C --recipe $T/recipe-v1 --sim-root $T/sim-baseline \
      --out $T/analysis-v1 --stress-recipe $T/recipe-v1-stress --stress-root $T/sim-stress \
      || { echo "VM_STRESS_FDP_FAIL $ROW $C $(date -u +%FT%TZ)"; rc=1; continue 2; }
  done
  echo "VM_STRESS_FDP_DONE $ROW $(date -u +%FT%TZ)"
done
exit $rc
