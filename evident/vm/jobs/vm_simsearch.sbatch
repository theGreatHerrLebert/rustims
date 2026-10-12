#!/bin/bash
#SBATCH --account=ki-proanagi
#SBATCH --partition=ki-smallcpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=vm-simsearch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/logs/vm-simsearch-%j.log
# Shared recipe (same libraries and flags as vm_real10.sbatch) applied to the baseline sims.
# Markers: VM_SIMSEARCH_DONE / VM_SIMSEARCH_FAIL <step>.
set -uo pipefail
module purge; module load tools/Apptainer/1.3.4-GCCcore-13.3.0 lang/Java/17.0.6
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
E=/lustre/project/ki-proanagi/dateschn/engines
W=$M/TIMSIM-REVISION/searches/vm/recipe-v1
OUT=${SIM_ROOT:-$M/TIMSIM-REVISION/searches/vm/sim-baseline}   # stress: SIM_ROOT=.../sim-stress
SO=${SEARCH_OUT:-$W/sim}                                        # stress: SEARCH_OUT=.../recipe-v1-stress/sim
JAR=$M/primitives/external/fdrbench/fdrbench-1.1.1/fdrbench-1.1.1.jar
PY=$M/vmenv/bin/python
T=$SLURM_CPUS_PER_TASK
fail() { echo "VM_SIMSEARCH_FAIL $1 $(date -u +%FT%TZ)"; exit 1; }
diann() { apptainer exec --bind /lustre $E/diann.sif $M/primitives/external/diann-2.5.1/diann-linux "$@"; }
SEARCH=(--threads $T --verbose 1 --qvalue 0.05 --mass-acc 15 --mass-acc-ms1 15 --no-prot-inf)
LIB_PLAIN=$(ls $W/lib/plain*.speclib | head -1); LIB_ENTRAP=$(ls $W/lib/entrap*.speclib | head -1)
[ -s "$LIB_PLAIN" ] && [ -s "$LIB_ENTRAP" ] || fail lib_missing
echo "VM_SIMSEARCH_START $(date -u +%FT%TZ)"; cat $W/lib/SHA256SUMS
while read -r RID; do
  D=$OUT/$RID; [ -f $D/done.flag ] || { echo "skip $RID (sim not done)"; continue; }
  IN=$(find $D/out -maxdepth 2 -name "*.d" -type d | head -1); [ -n "$IN" ] || fail "no .d $RID"
  for L in plain entrap; do
    O=$SO/$RID/$L; mkdir -p $O; [ -s $O/report.parquet ] && continue
    LIB=$LIB_PLAIN; [ $L = entrap ] && LIB=$LIB_ENTRAP
    diann --f "$IN" --lib "$LIB" --out $O/report.parquet "${SEARCH[@]}" > $O/diann.stdout 2>&1 || fail "search $RID $L"
    echo "SEARCH_DONE $RID $L $(date -u +%FT%TZ)"
  done
  O=$SO/$RID/entrap
  $PY - "$O/report.parquet" "$O/fdrbench_in.tsv" <<'PY' || fail "fdp_input $RID"
import sys, pandas as pd
d = pd.read_parquet(sys.argv[1]); d = d[d["Decoy"] == 0]
pd.DataFrame({"run": d["Run"], "peptide": d["Stripped.Sequence"].str.replace("I", "L"),
              "mod_peptide": d["Modified.Sequence"].str.replace("I", "L"), "charge": d["Precursor.Charge"],
              "q_value": d["Q.Value"], "protein": d["Protein.Ids"], "score": d["Q.Value"]}
             ).to_csv(sys.argv[2], sep="\t", index=False)
PY
  java -Xmx8G -jar $JAR -i $O/fdrbench_in.tsv -fold 1 -pep $W/entrap/human_entrap_pep.txt \
    -level precursor -o $O/fdp_precursor.csv -score 'score:0' > $O/fdrbench_fdp.log 2>&1 || fail "fdp $RID"
done < $OUT/queue.txt
echo "VM_SIMSEARCH_DONE $(date -u +%FT%TZ)"
