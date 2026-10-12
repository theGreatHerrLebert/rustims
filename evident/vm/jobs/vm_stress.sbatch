#!/bin/bash
#SBATCH --account=ki-proanagi
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --job-name=vm-stress
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/logs/vm-stress-%x-%A_%a.log
# Predictor-swap stress row (rustims evident/vm/stress_settings.yaml, frozen at $FROZEN).
# Deployed from rustims evident/vm/ to $EV with COMMIT and SHA256SUMS; every file is checked before use.
#   STAGE=prep : slim + rewritten DBs for the three 250k baseline seeds, replay configs, descriptive RT fit.
#                Markers VM_STRESS_PREP_DONE / VM_STRESS_FAIL prep <step>.
#   STAGE=sim  : array 0-11, one from_existing replay per task (queue.txt line = task id).
#                Markers VM_STRESS_RUN <id> rc=... / VM_STRESS_FAIL sim <id>.
set -uo pipefail
module purge; module load tools/Apptainer/1.3.4-GCCcore-13.3.0
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
R=$M/TIMSIM-REVISION/searches/vm
BASE=$R/sim-baseline
OUT=$R/sim-stress
EV=$M/evident-vm
CHR=$EV/chronologer-e518130
CHR_MODEL=$CHR/models/Chronologer_20220601193755.pt
CHR_SHA=1a500c246b49a1a23643bce7f2df86d5a107359bf0ec34365531c73431b6c0b3
X() { apptainer exec --nv --bind /lustre --env IMSPY_CACHE_DIR=$M/imspy-models,PYTHONPATH=$CHR/src:$EV \
        $M/containers/py312-bookworm.sif "$@"; }
PY=$M/timsim-env/bin/python
: "${STAGE:?STAGE=prep|sim}"; : "${FROZEN:?FROZEN=<commit that froze stress_settings.yaml>}"
fail() { echo "VM_STRESS_FAIL $STAGE $1 $(date -u +%FT%TZ)"; exit 1; }
[ "$(cat $EV/COMMIT)" = "$FROZEN" ] || fail "evident-vm is at $(cat $EV/COMMIT), not $FROZEN"
(cd $EV && sha256sum --quiet -c SHA256SUMS) || fail deployed_files_changed
one_db() { local n; n=$(find "$1" -name synthetic_data.db | wc -l); [ "$n" -eq 1 ] || return 1; find "$1" -name synthetic_data.db; }

if [ "$STAGE" = prep ]; then
  echo "$CHR_SHA  $CHR_MODEL" | sha256sum -c - || fail chronologer_sha
  mkdir -p $OUT
  for RID in jitter-250k-r1 jitter-250k-r2 jitter-250k-r3; do
    [ -f $BASE/$RID/done.flag ] || fail "baseline $RID not done"
    DB=$(one_db $BASE/$RID/out) || fail "not exactly one db for $RID"
    read -r SEED GRAD < <(X $PY -c "import json;p=json.load(open('$BASE/$RID/plan.json'));print(p['seed'],p['gradient_s'])")
    [ -n "${SEED:-}" ] && [ -n "${GRAD:-}" ] || fail "plan $RID"
    EXPECT=$((4100 + ${RID##*-r}))
    X $PY $EV/vm_stress_prepare.py --baseline-db "$DB" --gradient "$GRAD" --seed "$SEED" --expect-seed $EXPECT \
      --chronologer-model $CHR_MODEL --out $OUT/prepared/$RID > $OUT/prepared-$RID.log 2>&1 || fail "prepare $RID"
    echo "PREPARED $RID seed=$SEED $(date -u +%FT%TZ)"
  done
  X $PY $EV/make_vmstress.py || fail make_configs
  [ "$(wc -l < $OUT/queue.txt)" -eq 12 ] || fail queue_length
  X $PY $EV/vm_predictor_fit.py --recipe $R/recipe-v1 --chronologer-model $CHR_MODEL \
    --out $OUT/predictor_fit.json > $OUT/predictor_fit.log 2>&1 || echo "VM_STRESS_WARN predictor_fit failed (descriptive only)"
  echo "VM_STRESS_PREP_DONE $(date -u +%FT%TZ)"
  exit 0
fi

[ "$STAGE" = sim ] || fail "unknown STAGE"
T=${SLURM_ARRAY_TASK_ID:?array task id}; [ "$T" -ge 0 ] && [ "$T" -lt 12 ] || fail "task $T out of range"
SID=$(sed -n "$(( T + 1 ))p" $OUT/queue.txt); [ -n "$SID" ] || fail "no queue line $T"
D=$OUT/$SID
PREP=$(X $PY -c "import json;print(json.load(open('$D/plan.json'))['existing_path'])") || fail "plan $SID"
[ -s "$(dirname $PREP)/prepare.json" ] && [ -s "$PREP/synthetic_data.db" ] || fail "not prepared $SID"
if [ -f $D/done.flag ]; then
  [ "$(cat $D/done.flag)" = "$FROZEN" ] && [ -s $D/verify.json ] && { echo "skip $SID"; exit 0; }
  fail "stale done.flag in $SID"
fi
cd "$D" || fail "cd $SID"
rm -rf "$D/out" "$D/failed.flag" "$D/verify.json"; mkdir -p "$D/out"
echo "VM_STRESS_START $SID $(date -u +%FT%TZ) host=$(hostname)"
START=$(date +%s)
X $M/timsim-env/bin/timsim $D/config.toml > $D/timsim.log 2>&1
RC=$?
# guard: the replay must have loaded the prepared DB and must not have re-predicted RT/IM or added jitter
grep -q "Loading Existing Simulation" $D/timsim.log || { RC=90; echo "guard: no from_existing load"; }
grep -q -E "Applying (RT|ion mobility|intensity) variation|Re-scaling retention times|Simulating Ion Mobilities" \
  $D/timsim.log && { RC=91; echo "guard: replay re-predicted or jittered RT/IM"; }
echo "VM_STRESS_RUN $SID rc=$RC wall=$(( $(date +%s) - START ))s $(date -u +%FT%TZ)"
if [ $RC -eq 0 ]; then
  RDB=$(one_db $D/out) || { RC=92; echo "guard: not exactly one replay db"; }
fi
if [ $RC -eq 0 ]; then
  X $PY $EV/vm_stress_prepare.py --verify "$RDB" --prepared "$PREP" --out $D/verify.json || RC=93
fi
if [ $RC -eq 0 ]; then echo "$FROZEN" > $D/done.flag; else touch $D/failed.flag; tail -5 $D/timsim.log; fail "sim $SID rc=$RC"; fi
