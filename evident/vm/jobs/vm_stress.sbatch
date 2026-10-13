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
#   STAGE=prep-mass-accuracy : mass-accuracy DBs (per-ion m/z bias drawn from real LS10 deltas) and
#                queue-mass-accuracy.txt. Needs TIMSIM_ENV=$M/timsim-env-<freeze> (vm_timsim_env.sbatch).
#   STAGE=sim  : one from_existing replay per array task (line of $OUT/${QUEUE:-queue.txt} = task id + 1);
#                predictor-swap array 0-11, rt-spread array 0-2 with QUEUE=queue-rt-spread.txt,
#                mass-accuracy QUEUE=queue-mass-accuracy.txt: array 0-2 (replay-control-ma) first, then 3-17
#                with --dependency=afterok on it, because their --verify compares with the control replays
#                (a dependent task whose control is not done fails before simulating).
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
TE=${TIMSIM_ENV:-$M/timsim-env}   # mass-accuracy: $M/timsim-env-<freeze commit>, built by vm_timsim_env.sbatch
# A non-default env runs with the baseline env hidden behind an empty directory, so it cannot load code from it.
HIDE=(); [ "$TE" = "$M/timsim-env" ] || { mkdir -p $M/empty-dir; HIDE=(--bind $M/empty-dir:$M/timsim-env); }
X() { apptainer exec --nv --bind /lustre "${HIDE[@]}" --env IMSPY_CACHE_DIR=$M/imspy-models,PYTHONPATH=$CHR/src:$EV \
        $M/containers/py312-bookworm.sif "$@"; }
PY=$TE/bin/python
MA_YAML_SHA=43a1853fadb8379f31285945ad738af989c892fa9f3377bad83399b2504f44db  # stress_settings_mass_accuracy.yaml at fa6b4abf
MA_SO_SHA=208a54985bd37bc6196e4bbf75a4b15d1ee2f18d8bfc525feeea3bcef3984779   # connector .so built from fa6b4abf
: "${STAGE:?STAGE=prep|prep-rt-spread|prep-mass-accuracy|sim}"; : "${FROZEN:?FROZEN=<commit that froze stress_settings.yaml>}"
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

MA_COMMIT=fa6b4abf
ma_env() {  # the mass-accuracy replay environment: built from the freeze, nothing else changed, checked live
  [ "$TE" = "$M/timsim-env-$MA_COMMIT" ] && [ -s $TE/ENV_OK ] || fail "TIMSIM_ENV is not $M/timsim-env-$MA_COMMIT"
  grep -qx "commit $MA_COMMIT" $TE/ENV_OK && grep -qx "connector_so_sha256 $MA_SO_SHA" $TE/ENV_OK || fail env_ok_mismatch
  LIVE=$(X $PY -c "import hashlib,pathlib,imspy_connector as m;s=list(pathlib.Path(m.__file__).parent.glob('*.so'));assert len(s)==1;print(hashlib.sha256(s[0].read_bytes()).hexdigest())") \
    || fail live_connector
  [ "$LIVE" = "$MA_SO_SHA" ] || fail "live connector .so $LIVE != $MA_SO_SHA"
  FRZ=$(X $PY -m pip freeze 2>/dev/null | sha256sum | cut -d' ' -f1)
  grep -qx "freeze_sha256 $FRZ" $TE/ENV_OK || fail timsim_env_changed
  echo "$MA_YAML_SHA  $EV/stress_settings_mass_accuracy.yaml" | sha256sum -c - || fail ma_settings_changed
}

if [ "$STAGE" = prep-mass-accuracy ]; then
  ma_env
  REAL=$R/recipe-v1/real10/LS10/plain/report.parquet
  [ -s $REAL ] || fail "no real LS10 report"
  echo "cb22c214104f6ca76c72c456522cbf951115a8b139a8a935d08cc0d5be2e0147  $REAL" | sha256sum -c - || fail real_report_sha
  mkdir -p $OUT/prepared-mass-accuracy
  for RID in jitter-250k-r1 jitter-250k-r2 jitter-250k-r3; do
    [ -f $BASE/$RID/done.flag ] || fail "baseline $RID not done"
    DB=$(one_db $BASE/$RID/out) || fail "not exactly one db for $RID"
    read -r SEED GRAD < <(X $PY -c "import json;p=json.load(open('$BASE/$RID/plan.json'));print(p['seed'],p['gradient_s'])")
    [ -n "${SEED:-}" ] && [ -n "${GRAD:-}" ] || fail "plan $RID"
    X $PY $EV/vm_stress_prepare.py --row mass-accuracy --baseline-db "$DB" --gradient "$GRAD" --seed "$SEED" \
      --expect-seed $((4100 + ${RID##*-r})) --real-report $REAL --out $OUT/prepared-mass-accuracy/$RID \
      > $OUT/prepared-mass-accuracy-$RID.log 2>&1 || fail "prepare mass-accuracy $RID"
    echo "PREPARED mass-accuracy $RID seed=$SEED $(date -u +%FT%TZ)"
  done
  X $PY $EV/make_vmstress.py --row mass-accuracy || fail make_configs_mass_accuracy
  [ "$(wc -l < $OUT/queue-mass-accuracy.txt)" -eq 18 ] || fail queue_length_mass_accuracy
  [ "$(head -3 $OUT/queue-mass-accuracy.txt | grep -c -- '-replay-control-ma$')" -eq 3 ] || fail queue_order_mass_accuracy
  echo "VM_STRESS_PREP_DONE mass-accuracy $(date -u +%FT%TZ)"
  exit 0
fi

[ "$STAGE" = sim ] || fail "unknown STAGE"
QF=$OUT/${QUEUE:-queue.txt}; [ -s "$QF" ] || fail "no queue $QF"
T=${SLURM_ARRAY_TASK_ID:?array task id}; [ "$T" -ge 0 ] && [ "$T" -lt "$(wc -l < $QF)" ] || fail "task $T out of range"
SID=$(sed -n "$(( T + 1 ))p" $QF); [ -n "$SID" ] || fail "no queue line $T"
D=$OUT/$SID
PREP=$(X $PY -c "import json;print(json.load(open('$D/plan.json'))['existing_path'])") || fail "plan $SID"
[ -s "$(dirname $PREP)/prepare.json" ] && [ -s "$PREP/synthetic_data.db" ] || fail "not prepared $SID"
MA=0; case "$PREP" in */prepared-mass-accuracy/*) MA=1; ma_env ;; esac
[ $MA = 1 ] || [ "$TE" = "$M/timsim-env" ] || fail "only the mass-accuracy row runs in $TE"
# VERIFY_ONLY=1 re-runs the log guards and --verify on an existing replay (amendment 2); it never re-simulates
if [ "${VERIFY_ONLY:-0}" = 1 ]; then
  [ -s $D/timsim.log ] || fail "no replay to verify in $SID"
  rm -f "$D/done.flag" "$D/failed.flag" "$D/verify.json"
  echo "VM_STRESS_START $SID verify-only $(date -u +%FT%TZ) host=$(hostname)"
  START=$(date +%s); RC=0
else
ma_control() {  # dependent mass-accuracy replays need their seed's control replay, done at $FROZEN
  RID=$(basename "$(dirname $PREP)"); C=$OUT/$RID-replay-control-ma
  [ "$(cat $C/done.flag 2>/dev/null)" = "$FROZEN" ] && [ -s $C/verify.json ] && CDB=$(one_db $C/out)
}
if [ $MA = 1 ] && [ "$(basename $PREP)" != replay-control-ma ]; then
  ma_control || fail "$SID: $(basename "$(dirname $PREP)")-replay-control-ma not done at $FROZEN; submit 3-17 afterok on 0-2"
fi
if [ -f $D/done.flag ]; then
  [ "$(cat $D/done.flag)" = "$FROZEN" ] && [ -s $D/verify.json ] && { echo "skip $SID"; exit 0; }
  fail "stale done.flag in $SID"
fi
cd "$D" || fail "cd $SID"
rm -rf "$D/out" "$D/failed.flag" "$D/verify.json"; mkdir -p "$D/out"
echo "VM_STRESS_START $SID $(date -u +%FT%TZ) host=$(hostname)"
START=$(date +%s)
X $TE/bin/timsim $D/config.toml > $D/timsim.log 2>&1
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
CTRL_ARG=()
if [ $RC -eq 0 ] && [ $MA = 1 ] && [ "$(basename $PREP)" != replay-control-ma ]; then
  if ma_control; then CTRL_ARG=(--control-replay "$CDB"); else RC=94; echo "guard: control replay missing"; fi
fi
if [ $RC -eq 0 ]; then
  X $PY $EV/vm_stress_prepare.py --verify "$RDB" --prepared "$PREP" --out $D/verify.json "${CTRL_ARG[@]}" || RC=93
fi
if [ $RC -eq 0 ] && [ $MA = 1 ] && [ "$(basename $PREP)" = replay-control-ma ]; then
  # reported, not a gate: frame data of this control vs predictor-swap's replay-control of the same seed
  RID=$(basename "$(dirname $PREP)")
  A=$(find $D/out -name analysis.tdf_bin | head -1); B=$(find $OUT/$RID-replay-control/out -name analysis.tdf_bin 2>/dev/null | head -1)
  h() { [ -n "$1" ] && echo "\"$(sha256sum "$1" | cut -d' ' -f1)\"" || echo null; }
  printf '{"replay-control-ma": %s, "replay-control": %s}\n' "$(h "$A")" "$(h "$B")" > $D/tdf_bin_sha256.json
  grep -q null $D/tdf_bin_sha256.json && echo "VM_STRESS_WARN $SID tdf_bin hash missing (reported, not a gate)"
fi
if [ $RC -eq 0 ]; then echo "$FROZEN" > $D/done.flag; else touch $D/failed.flag; tail -5 $D/timsim.log; fail "sim $SID rc=$RC"; fi
