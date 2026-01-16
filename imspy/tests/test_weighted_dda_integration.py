"""
Unit tests for weighted scoring integration in DDA database search.

These tests validate the intensity prediction integration by testing:
1. Database hash computation for caching
2. Score type detection

Run tests:
    pytest tests/test_weighted_dda_integration.py -v
"""

import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_fasta():
    """Create a small FASTA for quick testing."""
    return """>sp|P00533|EGFR_HUMAN Epidermal growth factor receptor
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEV
>sp|P04406|G3P_HUMAN Glyceraldehyde-3-phosphate dehydrogenase
MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTV
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
"""


# ----------------------------------------------------------------------
# Test: Intensity Store Caching
# ----------------------------------------------------------------------

class TestIntensityStoreCaching:
    """Test intensity store caching behavior."""

    def test_database_hash_deterministic(self, small_fasta):
        """Test that database hash is deterministic."""
        from imspy.timstof.dbsearch.search_dda import compute_database_hash

        class MockConfig:
            static_modifications = {"C": "[UNIMOD:4]"}
            variable_modifications = {"M": ["[UNIMOD:35]"]}
            missed_cleavages = 2
            min_peptide_len = 7
            max_peptide_len = 30
            cleave_at = "KR"
            restrict = "P"
            c_terminal = True
            generate_decoys = True
            shuffle_decoys = False
            keep_ends = True

        config = MockConfig()

        hash1 = compute_database_hash(small_fasta, config)
        hash2 = compute_database_hash(small_fasta, config)

        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 16, "Hash should be 16 characters"

    def test_database_hash_changes_with_config(self, small_fasta):
        """Test that database hash changes when config changes."""
        from imspy.timstof.dbsearch.search_dda import compute_database_hash

        class MockConfig1:
            static_modifications = {"C": "[UNIMOD:4]"}
            variable_modifications = {"M": ["[UNIMOD:35]"]}
            missed_cleavages = 1
            min_peptide_len = 7
            max_peptide_len = 30
            cleave_at = "KR"
            restrict = "P"
            c_terminal = True
            generate_decoys = True
            shuffle_decoys = False
            keep_ends = True

        class MockConfig2:
            static_modifications = {"C": "[UNIMOD:4]"}
            variable_modifications = {"M": ["[UNIMOD:35]"]}
            missed_cleavages = 2  # Changed!
            min_peptide_len = 7
            max_peptide_len = 30
            cleave_at = "KR"
            restrict = "P"
            c_terminal = True
            generate_decoys = True
            shuffle_decoys = False
            keep_ends = True

        hash1 = compute_database_hash(small_fasta, MockConfig1())
        hash2 = compute_database_hash(small_fasta, MockConfig2())

        assert hash1 != hash2, "Hash should change when config changes"

    def test_is_weighted_score_type(self):
        """Test score type detection."""
        from imspy.timstof.dbsearch.search_dda import is_weighted_score_type

        # Weighted types
        assert is_weighted_score_type("weightedhyperscore") is True
        assert is_weighted_score_type("weightedopenmshyperscore") is True
        assert is_weighted_score_type("WeightedHyperscore") is True  # Case insensitive

        # Non-weighted types
        assert is_weighted_score_type("hyperscore") is False
        assert is_weighted_score_type("openmshyperscore") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
