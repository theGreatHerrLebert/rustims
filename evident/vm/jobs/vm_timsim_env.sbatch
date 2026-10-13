#!/bin/bash
#SBATCH --account=ki-proanagi
#SBATCH --partition=ki-smallcpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=vm-timsim-env
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3/logs/vm-timsim-env-%j.log
# Replay environment for the mass-accuracy row (evident/vm/stress_settings_mass_accuracy.yaml): a copy of the
# baseline timsim-env with only imspy-connector and imspy-simulation replaced by wheels built from the freeze
# commit fa6b4abf (same versions; adds the opt-in ions.mz_bias_ppm_ms1/ms2 hook). Offline install.
# Checks run with the baseline env hidden behind an empty directory, so the copy cannot load code from it.
# Markers VM_TIMSIM_ENV_DONE / VM_TIMSIM_ENV_FAIL <step>. Writes $TE/ENV_OK, which vm_stress.sbatch re-checks live.
set -uo pipefail
module purge; module load tools/Apptainer/1.3.4-GCCcore-13.3.0
M=/lustre/project/ki-proanagi/dateschn/TIMSIM-REVISION-M3
COMMIT=fa6b4abf
OLD=$M/timsim-env
TE=$M/timsim-env-$COMMIT
WH=$M/wheels-$COMMIT
CONNECTOR_WHL=imspy_connector-0.4.4-cp312-cp312-manylinux_2_34_x86_64.whl
SIM_WHL=imspy_simulation-0.4.5-py3-none-any.whl
SO_SHA=208a54985bd37bc6196e4bbf75a4b15d1ee2f18d8bfc525feeea3bcef3984779   # the .so inside the connector wheel
SIF=$M/containers/py312-bookworm.sif
mkdir -p $M/empty-dir
X() { apptainer exec --bind /lustre $SIF "$@"; }
XH() { apptainer exec --bind /lustre --bind $M/empty-dir:$OLD $SIF "$@"; }   # baseline env hidden
FREEZE() { "$1" "$2" -m pip freeze 2>/dev/null; }   # the env's own pip, as for the archived freeze
fail() { echo "VM_TIMSIM_ENV_FAIL $1 $(date -u +%FT%TZ)"; exit 1; }

printf '%s  %s\n' 09d436b37f799d1d45b952fdd0878879b0bf1f06bf0b5e33123cccc2167bebd5 $WH/$CONNECTOR_WHL \
  80002083f21c1b8c1fe632f36ee46fcee760f97a63bcecb10ac59960a79ff0b1 $WH/$SIM_WHL | sha256sum -c - || fail wheels_sha
echo "bd04f116a1fe2e402a17f308a31b00bb1f9721141613375a191fb35bc89107c0  $M/timsim-env.freeze.txt" | sha256sum -c - \
  || fail archived_freeze_changed
# the live baseline env must still be what the baseline sims ran with
diff <(FREEZE X $OLD/bin/python) $M/timsim-env.freeze.txt > /dev/null || fail baseline_env_drifted
[ -e $TE ] && fail "$TE exists; remove it first"
cp -a $OLD $TE || fail copy
# venv scripts carry the old interpreter path; repoint every text file in bin/
grep -rIl -- "$OLD/" $TE/bin | xargs -r sed -i "s#$OLD/#$TE/#g" || fail repoint
sed -i "s#$OLD\$#$TE#; s#$OLD\"#$TE\"#; s#$OLD'#$TE'#" $TE/bin/activate* 2>/dev/null
# nothing may still point into the baseline env: text in bin/, .pth and editable finders, symlink targets
LEFT=$(grep -rIl -e "$OLD/" -e "$OLD\$" -e "$OLD\"" $TE/bin $TE/pyvenv.cfg 2>/dev/null; \
       find $TE/lib -name "*.pth" -o -name "__editable__*" | xargs -r grep -Il -e "$OLD/" -e "$OLD\$")
[ -z "$LEFT" ] || fail "old env path still in: $LEFT"
LINKS=$(find $TE -type l -exec readlink -f {} \; | grep -e "^$OLD/" -e "^$OLD\$")
[ -z "$LINKS" ] || fail "symlinks resolve into the baseline env: $LINKS"
XH $TE/bin/python -m pip install --no-deps --no-index --force-reinstall \
  $WH/$CONNECTOR_WHL $WH/$SIM_WHL || fail install
CHK=$(XH $TE/bin/python - <<PY
import hashlib, pathlib, sys
te, old = "$TE", "$OLD"
assert not any(p == old or p.startswith(old + "/") for p in sys.path), sys.path
import imspy_connector, imspy_simulation, imspy_core, imspy_predictors, numpy, pandas, torch
for m in (imspy_connector, imspy_simulation, imspy_core, imspy_predictors, numpy, pandas):
    f = str(pathlib.Path(m.__file__).resolve())
    assert f.startswith(te + "/"), (m.__name__, f)
so = list(pathlib.Path(imspy_connector.__file__).parent.glob("*.so"))
assert len(so) == 1
import imspy_simulation.timsim.jobs.simulate_scan_distributions_with_variance as j
assert "mz_bias_ppm_ms2" in pathlib.Path(j.__file__).read_text()
print(hashlib.sha256(so[0].read_bytes()).hexdigest())
PY
) || fail import_check
[ "$CHK" = "$SO_SHA" ] || fail "connector .so sha $CHK != $SO_SHA"
head -1 $TE/bin/timsim | grep -qx "#!$TE/bin/python" || fail timsim_shebang
XH $TE/bin/timsim --help > /dev/null 2>&1 || fail timsim_entry_point
FREEZE XH $TE/bin/python > $TE.freeze.txt || fail freeze
# every package line is the baseline's except the two replaced wheels, which must be exactly the pinned ones
diff <(grep -v "^imspy_connector @\|^imspy-simulation @" $M/timsim-env.freeze.txt) \
     <(grep -v "^imspy_connector @\|^imspy-simulation @" $TE.freeze.txt) || fail other_packages_differ
grep -qx "imspy_connector @ file://$WH/$CONNECTOR_WHL#sha256=09d436b37f799d1d45b952fdd0878879b0bf1f06bf0b5e33123cccc2167bebd5" \
  $TE.freeze.txt || fail connector_not_pinned_wheel
grep -qx "imspy-simulation @ file://$WH/$SIM_WHL#sha256=80002083f21c1b8c1fe632f36ee46fcee760f97a63bcecb10ac59960a79ff0b1" \
  $TE.freeze.txt || fail simulation_not_pinned_wheel
printf "commit %s\nconnector_so_sha256 %s\nfreeze_sha256 %s\n" "$COMMIT" "$CHK" "$(sha256sum < $TE.freeze.txt | cut -d' ' -f1)" \
  > $TE/ENV_OK
cat $TE/ENV_OK
echo "VM_TIMSIM_ENV_DONE $(date -u +%FT%TZ)"
