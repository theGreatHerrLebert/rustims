"""The modform sequence annotation that lets a modified peptide fragment AS MODIFIED.

Without it, the v2 adapter hands v1 bare sequences, so a phosphopeptide is fragmented as its
unmodified self and the site-determining fragments never appear. `annotate_modform` inserts
`[UNIMOD:id]` after each modified residue, in v1's own phospho-path format.
"""

from imspy_simulation.timsim.jobs.load_from_v2 import annotate_modform


def test_single_mod_matches_v1_format():
    # v1 does seq[:p+1] + "[UNIMOD:21]" + seq[p+1:], i.e. the tag follows the modified residue.
    assert annotate_modform("PEPTIDEK", [(3, 21)]) == "PEPT[UNIMOD:21]IDEK"
    assert annotate_modform("PEPTIDEK", [(0, 1)]) == "P[UNIMOD:1]EPTIDEK"
    assert annotate_modform("PEPTIDEK", [(7, 4)]) == "PEPTIDEK[UNIMOD:4]"


def test_multiple_mods_do_not_shift_each_other():
    # Right-to-left insertion: an earlier tag must not move a later residue's index.
    assert annotate_modform("PEPTIDEK", [(3, 21), (6, 35)]) == "PEPT[UNIMOD:21]IDE[UNIMOD:35]K"
    # Order of the input list must not matter.
    assert annotate_modform("PEPTIDEK", [(6, 35), (3, 21)]) == "PEPT[UNIMOD:21]IDE[UNIMOD:35]K"


def test_unmodified_is_a_noop():
    assert annotate_modform("PEPTIDEK", []) == "PEPTIDEK"


def test_positional_isomers_get_distinct_sequences():
    # Same backbone, mod on different residues -> different annotated sequences. This is what makes
    # the two isomers distinct fragmented species in the .d.
    on_s = annotate_modform("AASPEYK", [(2, 21)])
    on_y = annotate_modform("AASPEYK", [(5, 21)])
    assert on_s != on_y
    assert on_s == "AAS[UNIMOD:21]PEYK"
    assert on_y == "AASPEY[UNIMOD:21]K"
