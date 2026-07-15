"""The modform sequence annotation that lets a modified peptide fragment AS MODIFIED.

Without it, the v2 adapter hands v1 bare sequences, so a phosphopeptide is fragmented as its
unmodified self and the site-determining fragments never appear. `annotate_modform` renders the peptidoform in v1's format: a residue tag follows its
residue, an n-terminal tag is a prefix.
"""

from imspy_simulation.timsim.jobs.load_from_v2 import annotate_modform


def test_residue_mod_follows_its_residue():
    # v1 does seq[:p+1] + "[UNIMOD:21]" + seq[p+1:], i.e. the tag follows the modified residue.
    assert annotate_modform("PEPTIDEK", [(3, 21, "residue")]) == "PEPT[UNIMOD:21]IDEK"
    assert annotate_modform("PEPTIDEK", [(0, 21, "residue")]) == "P[UNIMOD:21]EPTIDEK"


def test_n_terminal_mod_is_a_prefix_not_a_first_residue_mod():
    # THE regression for the n-term bug: an n_term mod sits at position 0, same as a residue mod on
    # the first residue, but v1 writes it as a PREFIX. Rendering it as P[UNIMOD:1]... is a different
    # molecule to the fragment chemistry.
    assert annotate_modform("PEPTIDEK", [(0, 1, "n_term")]) == "[UNIMOD:1]PEPTIDEK"
    # n-term acetyl + a residue phospho: prefix, then the residue tag in place.
    assert (
        annotate_modform("PEPTIDEK", [(0, 1, "n_term"), (3, 21, "residue")])
        == "[UNIMOD:1]PEPT[UNIMOD:21]IDEK"
    )


def test_c_terminal_mod_is_a_suffix():
    assert annotate_modform("PEPTIDEK", [(7, 4, "c_term")]) == "PEPTIDEK[UNIMOD:4]"


def test_multiple_mods_do_not_shift_each_other():
    # Right-to-left insertion: an earlier tag must not move a later residue's index.
    assert (
        annotate_modform("PEPTIDEK", [(3, 21, "residue"), (6, 35, "residue")])
        == "PEPT[UNIMOD:21]IDE[UNIMOD:35]K"
    )
    # Order of the input list must not matter.
    assert (
        annotate_modform("PEPTIDEK", [(6, 35, "residue"), (3, 21, "residue")])
        == "PEPT[UNIMOD:21]IDE[UNIMOD:35]K"
    )


def test_unmodified_is_a_noop():
    assert annotate_modform("PEPTIDEK", []) == "PEPTIDEK"


def test_positional_isomers_get_distinct_sequences():
    on_s = annotate_modform("AASPEYK", [(2, 21, "residue")])
    on_y = annotate_modform("AASPEYK", [(5, 21, "residue")])
    assert on_s != on_y
    assert on_s == "AAS[UNIMOD:21]PEYK"
    assert on_y == "AASPEY[UNIMOD:21]K"
