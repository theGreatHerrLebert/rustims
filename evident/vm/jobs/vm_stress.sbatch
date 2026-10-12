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
# Stress rows predictor-swap (evident/vm/stress_settings.yaml) and rt-spread (stress_settings_rt_spread.yaml),
# frozen at $FROZEN.
# Deployed from rustims evident/vm/ to $EV with COMMIT and SHA256SUMS; every file is checked before use.
#   STAGE=prep : slim + rewritten DBs for the three 250k baseline seeds, replay configs, descriptive RT fit.
#                Markers VM_STRESS_PREP_DONE / VM_STRESS_FAIL prep <step>.
#   STAGE=prep-rt-spread : rt-spread DBs (RT apexes quantile-mapped onto real LS10) and queue-rt-spread.txt.
#   STAGE=sim  : one from_existing replay per array task (line of $OUT/${QUEUE:-queue.txt} = task id + 1);
#                predictor-swap array 0-11, rt-spread array 0-2 with QUEUE=queue-rt-spread.txt.
#                Markers VM_STRESS_RUN <id> rc=... / VM_STRESS_FAIL sim <id>.
set -uo pipefail
module purge; module load tools/Apptainer/1.3.4-GCCcore-13.3.0
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
R=$M/TIMSIM-REVISION/searches/vm
BASE=$R/sim-baseline
OUT=$R/sim-stress
EV=${EV:-$M/evident-vm}         # each freeze deploys to its own directory; rt-spread: EV=$M/evident-vm-<commit>
CHR=$EV/chronologer-e518130
CHR_MODEL=$CHR/models/Chronologer_20220601193755.pt
CHR_SHA=1a500c246b49a1a23643bce7f2df86d5a107359bf0ec34365531c73431b6c0b3
X() { apptainer exec --nv --bind /lustre --env IMSPY_CACHE_DIR=$M/imspy-models,PYTHONPATH=$CHR/src:$EV \
        $M/containers/py312-bookworm.sif "$@"; }
PY=$M/timsim-env/bin/python
: "${STAGE:?STAGE=prep|prep-rt-spread|sim}"; : "${FROZEN:?FROZEN=<commit that froze stress_settings.yaml>}"
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

if [ "$STAGE" = prep-rt-spread ]; then
  REAL=$R/recipe-v1/real10/LS10/plain/report.parquet
  [ -s $REAL ] || fail "no real LS10 report"
  mkdir -p $OUT/prepared-rt-spread
  # pinned in stress_settings_rt_spread.yaml: the real sample and the replay environment
  echo "cb22c214104f6ca76c72c456522cbf951115a8b139a8a935d08cc0d5be2e0147  $REAL" | sha256sum -c - || fail real_report_sha
  echo "bd04f116a1fe2e402a17f308a31b00bb1f9721141613375a191fb35bc89107c0  $M/timsim-env.freeze.txt" | sha256sum -c - \
    || fail timsim_env_changed
  for RID in jitter-250k-r1 jitter-250k-r2 jitter-250k-r3; do
    [ -f $BASE/$RID/done.flag ] || fail "baseline $RID not done"
    DB=$(one_db $BASE/$RID/out) || fail "not exactly one db for $RID"
    read -r SEED GRAD < <(X $PY -c "import json;p=json.load(open('$BASE/$RID/plan.json'));print(p['seed'],p['gradient_s'])")
    [ -n "${SEED:-}" ] && [ -n "${GRAD:-}" ] || fail "plan $RID"
    X $PY $EV/vm_stress_prepare.py --row rt-spread --baseline-db "$DB" --gradient "$GRAD" --seed "$SEED" \
      --expect-seed $((4100 + ${RID##*-r})) --real-report $REAL --out $OUT/prepared-rt-spread/$RID \
      > $OUT/prepared-rt-spread-$RID.log 2>&1 || fail "prepare rt-spread $RID"
    echo "PREPARED rt-spread $RID seed=$SEED $(date -u +%FT%TZ)"
  done
  X $PY $EV/make_vmstress.py --row rt-spread || fail make_configs_rt_spread
  [ "$(wc -l < $OUT/queue-rt-spread.txt)" -eq 3 ] || fail queue_length_rt_spread
  echo "VM_STRESS_PREP_DONE rt-spread $(date -u +%FT%TZ)"
  exit 0
fi

[ "$STAGE" = sim ] || fail "unknown STAGE"
QF=$OUT/${QUEUE:-queue.txt}; [ -s "$QF" ] || fail "no queue $QF"
T=${SLURM_ARRAY_TASK_ID:?array task id}; [ "$T" -ge 0 ] && [ "$T" -lt "$(wc -l < $QF)" ] || fail "task $T out of range"
SID=$(sed -n "$(( T + 1 ))p" $QF); [ -n "$SID" ] || fail "no queue line $T"
D=$OUT/$SID
PREP=$(X $PY -c "import json;print(json.load(open('$D/plan.json'))['existing_path'])") || fail "plan $SID"
[ -s "$(dirname $PREP)/prepare.json" ] && [ -s "$PREP/synthetic_data.db" ] || fail "not prepared $SID"
# VERIFY_ONLY=1 re-runs the log guards and --verify on an existing replay (amendment 2); it never re-simulates
if [ "${VERIFY_ONLY:-0}" = 1 ]; then
  [ -s $D/timsim.log ] || fail "no replay to verify in $SID"
  rm -f "$D/done.flag" "$D/failed.flag" "$D/verify.json"
  echo "VM_STRESS_START $SID verify-only $(date -u +%FT%TZ) host=$(hostname)"
  START=$(date +%s); RC=0
else
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
fi
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
