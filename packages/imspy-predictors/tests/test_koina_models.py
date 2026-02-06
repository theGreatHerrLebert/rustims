"""Tests for Koina model access, format conversion, and input filtering.

This module tests the koina_models package functionality including:
- Format conversion between UNIMOD bracket notation and AlphaBase format
- Model type detection and supported model listing
- Input filtering based on model-specific requirements
- Model compatibility validation

Models tested (based on Koina server availability):
- Prosit family (intensity, RT, TMT variants)
- AlphaPeptDeep/AlphaPept family (MS2, RT, CCS)
- MS2PIP family (HCD, timsTOF, etc.)
- DeepLC (RT)
- Chronologer (RT)
- IM2Deep (CCS)
- pFly (flyability)
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_peptides():
    """Sample peptide sequences for testing."""
    return [
        "PEPTIDE",
        "SEQUENCE",
        "AAAAAAAAAA",  # 10 AA
        "PEPTIDEPEPTIDEPEPTIDEPEPTIDEK",  # 29 AA (at Prosit limit)
        "PEPTIDEPEPTIDEPEPTIDEPEPTIDEKK",  # 30 AA (at Prosit limit)
        "PEPTIDEPEPTIDEPEPTIDEPEPTIDEKKK",  # 31 AA (over Prosit limit)
    ]


@pytest.fixture
def phospho_peptides():
    """Phosphorylated peptide sequences."""
    return [
        "AS[UNIMOD:21]DFK",              # Phospho-Ser
        "AT[UNIMOD:21]DFK",              # Phospho-Thr
        "AY[UNIMOD:21]DFK",              # Phospho-Tyr
        "AS[UNIMOD:21]DT[UNIMOD:21]FK",  # Double phospho
    ]


@pytest.fixture
def modified_peptides():
    """Peptides with various modifications."""
    return [
        "C[UNIMOD:4]PEPTIDE",            # Carbamidomethyl-Cys
        "M[UNIMOD:35]PEPTIDE",           # Oxidation-Met
        "C[UNIMOD:4]PEPTM[UNIMOD:35]E",  # Multiple mods (Prosit-compatible)
        "S[UNIMOD:21]PEPTIDE",           # Phospho (not Prosit-compatible)
        "K[UNIMOD:121]PEPTIDE",          # Ubiquitin (exotic mod)
    ]


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing filtering functions."""
    return pd.DataFrame({
        "peptide_sequences": [
            "PEPTIDE",
            "SEQUENCE",
            "C[UNIMOD:4]PEPTIDE",
            "M[UNIMOD:35]PEPTIDE",
            "S[UNIMOD:21]PEPTIDE",
            "VERYLONGPEPTIDESEQUENCETHATEXCEEDSLIMIT",  # >30 AA
        ],
        "precursor_charges": [2, 3, 2, 2, 2, 2],
        "collision_energies": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    })


# =============================================================================
# Format Conversion Tests
# =============================================================================

class TestUnimodFormatConversion:
    """Tests for UNIMOD format conversion to AlphaBase format."""

    def test_convert_single_phospho(self):
        """Test conversion of single phosphorylation."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("AS[UNIMOD:21]DFK")
        assert result == "AS(UniMod:21)DFK"

    def test_convert_multiple_mods(self):
        """Test conversion of multiple modifications."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("M[UNIMOD:35]AS[UNIMOD:21]DFKC[UNIMOD:4]")
        assert result == "M(UniMod:35)AS(UniMod:21)DFKC(UniMod:4)"

    def test_convert_no_mods(self):
        """Test that unmodified sequences are unchanged."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("PEPTIDE")
        assert result == "PEPTIDE"

    def test_convert_carbamidomethyl(self):
        """Test conversion of carbamidomethyl cysteine."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("C[UNIMOD:4]PEPTIDE")
        assert result == "C(UniMod:4)PEPTIDE"

    def test_convert_oxidation(self):
        """Test conversion of methionine oxidation."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("PEPTM[UNIMOD:35]IDE")
        assert result == "PEPTM(UniMod:35)IDE"

    def test_convert_tmt_label(self):
        """Test conversion of TMT labels."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        result = convert_unimod_to_alphabase_sequence("K[UNIMOD:312]PEPTIDE")
        assert result == "K(UniMod:312)PEPTIDE"

    def test_convert_silac_labels(self):
        """Test conversion of SILAC heavy labels."""
        from imspy_predictors.koina_models import convert_unimod_to_alphabase_sequence

        # Heavy Lys
        result = convert_unimod_to_alphabase_sequence("PEPTIDEK[UNIMOD:259]")
        assert result == "PEPTIDEK(UniMod:259)"

        # Heavy Arg
        result = convert_unimod_to_alphabase_sequence("PEPTIDER[UNIMOD:267]")
        assert result == "PEPTIDER(UniMod:267)"


class TestUnimodToNamedMods:
    """Tests for UNIMOD to named modification conversion."""

    def test_convert_to_named_phospho(self):
        """Test conversion to named phosphorylation format."""
        from imspy_predictors.koina_models import convert_unimod_to_named_mods

        bare, mods, sites = convert_unimod_to_named_mods("AS[UNIMOD:21]DFK")
        assert bare == "ASDFK"
        assert mods == "Phospho@S"
        assert sites == "2"

    def test_convert_to_named_multiple(self):
        """Test conversion with multiple modifications."""
        from imspy_predictors.koina_models import convert_unimod_to_named_mods

        bare, mods, sites = convert_unimod_to_named_mods("M[UNIMOD:35]AS[UNIMOD:21]DFK")
        assert bare == "MASDFK"
        assert "Oxidation@M" in mods
        assert "Phospho@S" in mods

    def test_convert_to_named_no_mods(self):
        """Test conversion of unmodified sequence."""
        from imspy_predictors.koina_models import convert_unimod_to_named_mods

        bare, mods, sites = convert_unimod_to_named_mods("PEPTIDE")
        assert bare == "PEPTIDE"
        assert mods == ""
        assert sites == ""

    def test_convert_to_named_unknown_mod(self):
        """Test conversion of unknown UNIMOD ID."""
        from imspy_predictors.koina_models import convert_unimod_to_named_mods

        bare, mods, sites = convert_unimod_to_named_mods("K[UNIMOD:9999]PEPTIDE")
        assert bare == "KPEPTIDE"
        assert "UNIMOD:9999" in mods


class TestDataFrameConversion:
    """Tests for DataFrame-level format conversion."""

    def test_convert_dataframe_alphapeptdeep(self, sample_dataframe):
        """Test DataFrame conversion for AlphaPeptDeep models."""
        from imspy_predictors.koina_models import convert_dataframe_for_model

        result = convert_dataframe_for_model(
            sample_dataframe, "AlphaPeptDeep_ms2_generic"
        )

        # Check phospho sequence converted
        assert "S(UniMod:21)PEPTIDE" in result["peptide_sequences"].values

    def test_convert_dataframe_prosit_unchanged(self, sample_dataframe):
        """Test DataFrame NOT converted for Prosit models."""
        from imspy_predictors.koina_models import convert_dataframe_for_model

        result = convert_dataframe_for_model(
            sample_dataframe, "Prosit_2023_intensity_timsTOF"
        )

        # Prosit format should be unchanged
        assert "S[UNIMOD:21]PEPTIDE" in result["peptide_sequences"].values

    def test_convert_dataframe_ms2pip_unchanged(self, sample_dataframe):
        """Test DataFrame NOT converted for MS2PIP models."""
        from imspy_predictors.koina_models import convert_dataframe_for_model

        result = convert_dataframe_for_model(
            sample_dataframe, "ms2pip_timsTOF2024"
        )

        # MS2PIP format should be unchanged
        assert "S[UNIMOD:21]PEPTIDE" in result["peptide_sequences"].values


# =============================================================================
# Model Type Detection Tests
# =============================================================================

class TestModelTypeDetection:
    """Tests for model type detection from model names."""

    @pytest.mark.parametrize("model_name,expected_type", [
        # Prosit family
        ("Prosit_2019_intensity", "prosit"),
        ("Prosit_2020_intensity_HCD", "prosit"),
        ("Prosit_2020_intensity_CID", "prosit"),
        ("Prosit_2020_intensity_TMT", "prosit"),
        ("Prosit_2023_intensity_timsTOF", "prosit"),
        ("Prosit_2024_intensity_cit", "prosit"),
        ("Prosit_2019_irt", "prosit"),
        ("Prosit_2020_irt_TMT", "prosit"),
        # AlphaPeptDeep family
        ("AlphaPeptDeep_ms2_generic", "alphapeptdeep"),
        ("AlphaPeptDeep_rt_generic", "alphapeptdeep"),
        ("AlphaPeptDeep_ccs_generic", "alphapeptdeep"),
        ("AlphaPept_ms2_generic", "alphapept"),
        ("AlphaPept_rt_generic", "alphapept"),
        ("AlphaPept_ccs_generic", "alphapept"),
        # MS2PIP family
        ("ms2pip_2021_HCD", "ms2pip"),
        ("ms2pip_Immuno_HCD", "ms2pip"),
        ("ms2pip_TTOF5600", "ms2pip"),
        ("ms2pip_timsTOF2024", "ms2pip"),
        ("ms2pip_CID_TMT", "ms2pip"),
        ("ms2pip_timsTOF2023", "ms2pip"),
        # Other models
        ("Deeplc_hela_hf", "deeplc"),
        ("Chronologer_RT", "chronologer"),
        ("IM2Deep", "im2deep"),
        ("pfly_2024_fine_tuned", "pfly"),
    ])
    def test_get_model_type(self, model_name, expected_type):
        """Test model type extraction from model names."""
        from imspy_predictors.koina_models import get_model_type

        result = get_model_type(model_name)
        assert result == expected_type


class TestSupportedModels:
    """Tests for supported model listing."""

    def test_get_supported_models(self):
        """Test that supported models list includes expected types."""
        from imspy_predictors.koina_models import get_supported_models

        models = get_supported_models()
        assert "prosit" in models
        assert "alphapeptdeep" in models
        assert "ms2pip" in models
        assert "deeplc" in models
        assert "chronologer" in models
        assert "im2deep" in models
        assert "pfly" in models

    def test_model_filters_defined(self):
        """Test that MODEL_FILTERS is properly defined."""
        from imspy_predictors.koina_models import MODEL_FILTERS

        assert isinstance(MODEL_FILTERS, dict)
        assert "prosit" in MODEL_FILTERS
        assert "alphapeptdeep" in MODEL_FILTERS

    def test_model_descriptions_defined(self):
        """Test that MODEL_DESCRIPTIONS is properly defined."""
        from imspy_predictors.koina_models import MODEL_DESCRIPTIONS

        assert isinstance(MODEL_DESCRIPTIONS, dict)
        assert "prosit" in MODEL_DESCRIPTIONS
        assert "Prosit" in MODEL_DESCRIPTIONS["prosit"]


# =============================================================================
# Model Restrictions Tests
# =============================================================================

class TestModelRestrictions:
    """Tests for model-specific restriction retrieval."""

    def test_prosit_restrictions(self):
        """Test Prosit model restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("Prosit_2023_intensity_timsTOF")
        assert restrictions is not None
        assert len(restrictions) > 0

        # Check for length restriction
        length_filter = next((f for f in restrictions if "length" in f), None)
        assert length_filter is not None
        assert length_filter["length"][1] == 30  # Max 30 AA

        # Check for modification restriction
        mod_filter = next((f for f in restrictions if "modifications" in f), None)
        assert mod_filter is not None
        assert "C[UNIMOD:4]" in mod_filter["modifications"]
        assert "M[UNIMOD:35]" in mod_filter["modifications"]

        # Check for charge restriction
        charge_filter = next((f for f in restrictions if "precursor_charges" in f), None)
        assert charge_filter is not None
        assert 1 in charge_filter["precursor_charges"]
        assert 6 in charge_filter["precursor_charges"]

    def test_alphapeptdeep_no_restrictions(self):
        """Test AlphaPeptDeep has no restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("AlphaPeptDeep_ms2_generic")
        assert restrictions is None

    def test_ms2pip_length_restriction(self):
        """Test MS2PIP model restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("ms2pip_timsTOF2024")
        assert restrictions is not None

        length_filter = next((f for f in restrictions if "length" in f), None)
        assert length_filter is not None
        assert length_filter["length"][1] == 30

    def test_deeplc_length_restriction(self):
        """Test DeepLC model restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("Deeplc_hela_hf")
        assert restrictions is not None

        length_filter = next((f for f in restrictions if "length" in f), None)
        assert length_filter is not None
        assert length_filter["length"][1] == 60

    def test_im2deep_length_restriction(self):
        """Test IM2Deep model restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("IM2Deep")
        assert restrictions is not None

        length_filter = next((f for f in restrictions if "length" in f), None)
        assert length_filter is not None
        assert length_filter["length"][1] == 60

    def test_chronologer_no_restrictions(self):
        """Test Chronologer has no restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("Chronologer_RT")
        assert restrictions is None

    def test_pfly_no_restrictions(self):
        """Test pFly has no restrictions."""
        from imspy_predictors.koina_models import get_model_restrictions

        restrictions = get_model_restrictions("pfly_2024_fine_tuned")
        assert restrictions is None


# =============================================================================
# Input Filtering Tests
# =============================================================================

class TestPeptideLengthFilter:
    """Tests for peptide length filtering."""

    def test_filter_by_length_prosit(self):
        """Test filtering peptides by length for Prosit (max 30)."""
        from imspy_predictors.koina_models import filter_peptide_length

        df = pd.DataFrame({
            "peptide_sequences": [
                "PEPTIDE",           # 7 AA
                "A" * 30,            # 30 AA (at limit)
                "A" * 31,            # 31 AA (over limit)
                "A" * 50,            # 50 AA (way over)
            ],
            "precursor_charges": [2, 2, 2, 2],
        })

        result = filter_peptide_length(df, min_len=0, max_len=30)
        assert len(result) == 2
        assert "A" * 30 in result["peptide_sequences"].values
        assert "A" * 31 not in result["peptide_sequences"].values

    def test_filter_length_with_mods(self):
        """Test that modification annotations don't count toward length."""
        from imspy_predictors.koina_models import filter_peptide_length

        df = pd.DataFrame({
            "peptide_sequences": [
                "C[UNIMOD:4]" + "A" * 29,  # 30 AA bare (at limit)
                "C[UNIMOD:4]" + "A" * 30,  # 31 AA bare (over limit)
            ],
            "precursor_charges": [2, 2],
        })

        result = filter_peptide_length(df, min_len=0, max_len=30)
        assert len(result) == 1

    def test_filter_length_deeplc(self):
        """Test filtering for DeepLC (max 60)."""
        from imspy_predictors.koina_models import filter_peptide_length

        df = pd.DataFrame({
            "peptide_sequences": [
                "A" * 60,  # 60 AA (at limit)
                "A" * 61,  # 61 AA (over limit)
            ],
            "precursor_charges": [2, 2],
        })

        result = filter_peptide_length(df, min_len=0, max_len=60)
        assert len(result) == 1


class TestModificationFilter:
    """Tests for peptide modification filtering."""

    def test_filter_prosit_allowed_mods(self):
        """Test filtering for Prosit-compatible modifications."""
        from imspy_predictors.koina_models import filter_peptide_modifications

        df = pd.DataFrame({
            "peptide_sequences": [
                "PEPTIDE",                    # No mods (OK)
                "C[UNIMOD:4]PEPTIDE",         # Carbamidomethyl (OK)
                "M[UNIMOD:35]PEPTIDE",        # Oxidation (OK)
                "S[UNIMOD:21]PEPTIDE",        # Phospho (NOT OK for Prosit)
                "K[UNIMOD:121]PEPTIDE",       # Other mod (NOT OK)
            ],
            "precursor_charges": [2, 2, 2, 2, 2],
        })

        allowed = ["C[UNIMOD:4]", "M[UNIMOD:35]"]
        result = filter_peptide_modifications(df, allowed_mods=allowed)

        assert len(result) == 3
        assert "PEPTIDE" in result["peptide_sequences"].values
        assert "C[UNIMOD:4]PEPTIDE" in result["peptide_sequences"].values
        assert "M[UNIMOD:35]PEPTIDE" in result["peptide_sequences"].values
        assert "S[UNIMOD:21]PEPTIDE" not in result["peptide_sequences"].values

    def test_filter_no_restriction(self):
        """Test that None allowed_mods passes all."""
        from imspy_predictors.koina_models import filter_peptide_modifications

        df = pd.DataFrame({
            "peptide_sequences": [
                "PEPTIDE",
                "S[UNIMOD:21]PEPTIDE",
                "K[UNIMOD:121]PEPTIDE",
            ],
            "precursor_charges": [2, 2, 2],
        })

        result = filter_peptide_modifications(df, allowed_mods=None)
        assert len(result) == 3


class TestChargeFilter:
    """Tests for precursor charge filtering."""

    def test_filter_prosit_charges(self):
        """Test filtering for Prosit-compatible charges (1-6)."""
        from imspy_predictors.koina_models import filter_precursor_charges

        df = pd.DataFrame({
            "peptide_sequences": ["PEPTIDE"] * 8,
            "precursor_charges": [0, 1, 2, 3, 4, 5, 6, 7],
        })

        allowed = [1, 2, 3, 4, 5, 6]
        result = filter_precursor_charges(df, allowed_charges=allowed)

        assert len(result) == 6
        assert 0 not in result["precursor_charges"].values
        assert 7 not in result["precursor_charges"].values

    def test_filter_charge_no_restriction(self):
        """Test that None allowed_charges passes all."""
        from imspy_predictors.koina_models import filter_precursor_charges

        df = pd.DataFrame({
            "peptide_sequences": ["PEPTIDE"] * 3,
            "precursor_charges": [1, 5, 10],
        })

        result = filter_precursor_charges(df, allowed_charges=None)
        assert len(result) == 3


class TestModelBasedFiltering:
    """Tests for model-based input filtering."""

    def test_filter_for_prosit(self, sample_dataframe):
        """Test full filtering pipeline for Prosit model."""
        from imspy_predictors.koina_models import filter_input_by_model

        result = filter_input_by_model("Prosit_2023_intensity_timsTOF", sample_dataframe)

        # Phospho peptide should be filtered out
        assert "S[UNIMOD:21]PEPTIDE" not in result["peptide_sequences"].values

        # Long peptide should be filtered out
        assert "VERYLONGPEPTIDESEQUENCETHATEXCEEDSLIMIT" not in result["peptide_sequences"].values

        # Standard peptides should remain
        assert "PEPTIDE" in result["peptide_sequences"].values
        assert "C[UNIMOD:4]PEPTIDE" in result["peptide_sequences"].values

    def test_filter_for_alphapeptdeep(self, sample_dataframe):
        """Test that AlphaPeptDeep doesn't filter (no restrictions)."""
        from imspy_predictors.koina_models import filter_input_by_model

        result = filter_input_by_model("AlphaPeptDeep_ms2_generic", sample_dataframe)

        # All peptides should pass (no restrictions)
        assert len(result) == len(sample_dataframe)

    def test_filter_for_ms2pip(self, sample_dataframe):
        """Test filtering for MS2PIP (length only)."""
        from imspy_predictors.koina_models import filter_input_by_model

        result = filter_input_by_model("ms2pip_timsTOF2024", sample_dataframe)

        # Long peptide should be filtered
        assert "VERYLONGPEPTIDESEQUENCETHATEXCEEDSLIMIT" not in result["peptide_sequences"].values

        # Others should pass
        assert "S[UNIMOD:21]PEPTIDE" in result["peptide_sequences"].values

    def test_filter_unsupported_model_raises(self):
        """Test that unsupported model raises ValueError."""
        from imspy_predictors.koina_models import filter_input_by_model

        df = pd.DataFrame({
            "peptide_sequences": ["PEPTIDE"],
            "precursor_charges": [2],
        })

        with pytest.raises(ValueError, match="not supported"):
            filter_input_by_model("UnsupportedModel_v1", df)


# =============================================================================
# Model Compatibility Validation Tests
# =============================================================================

class TestModelCompatibilityValidation:
    """Tests for model compatibility validation."""

    def test_validate_prosit_compatible(self):
        """Test validation of Prosit-compatible peptides."""
        from imspy_predictors.koina_models import validate_model_compatibility

        result = validate_model_compatibility(
            "Prosit_2023_intensity_timsTOF",
            ["PEPTIDE", "C[UNIMOD:4]PEPTIDE", "M[UNIMOD:35]PEPTIDE"],
            charges=[2, 2, 2],
        )

        assert result["compatible"] is True
        assert result["incompatible_count"] == 0
        assert len(result["reasons"]) == 0

    def test_validate_prosit_incompatible_length(self):
        """Test validation catches length violations."""
        from imspy_predictors.koina_models import validate_model_compatibility

        result = validate_model_compatibility(
            "Prosit_2023_intensity_timsTOF",
            ["A" * 31],  # Over 30 AA limit
            charges=[2],
        )

        assert result["compatible"] is False
        assert result["incompatible_count"] > 0
        assert any("Length" in r for r in result["reasons"])

    def test_validate_prosit_incompatible_mod(self):
        """Test validation catches unsupported modifications."""
        from imspy_predictors.koina_models import validate_model_compatibility

        result = validate_model_compatibility(
            "Prosit_2023_intensity_timsTOF",
            ["S[UNIMOD:21]PEPTIDE"],  # Phospho not supported
            charges=[2],
        )

        assert result["compatible"] is False
        assert any("modification" in r.lower() for r in result["reasons"])

    def test_validate_prosit_incompatible_charge(self):
        """Test validation catches unsupported charges."""
        from imspy_predictors.koina_models import validate_model_compatibility

        result = validate_model_compatibility(
            "Prosit_2023_intensity_timsTOF",
            ["PEPTIDE"],
            charges=[7],  # Charge 7 not supported
        )

        assert result["compatible"] is False
        assert any("Charge" in r for r in result["reasons"])

    def test_validate_alphapeptdeep_all_compatible(self):
        """Test AlphaPeptDeep accepts all peptides."""
        from imspy_predictors.koina_models import validate_model_compatibility

        result = validate_model_compatibility(
            "AlphaPeptDeep_ms2_generic",
            ["S[UNIMOD:21]PEPTIDE", "K[UNIMOD:121]PEPTIDE", "A" * 100],
            charges=[2, 10, 2],
        )

        assert result["compatible"] is True


# =============================================================================
# Exception Classes Tests
# =============================================================================

class TestKoinaExceptions:
    """Tests for Koina exception classes."""

    def test_koina_error_base(self):
        """Test KoinaError base class."""
        from imspy_predictors.koina_models import KoinaError

        error = KoinaError("Test error")
        assert str(error) == "Test error"

    def test_koina_disabled_error(self):
        """Test KoinaDisabledError."""
        from imspy_predictors.koina_models import KoinaDisabledError

        error = KoinaDisabledError("Koina is disabled")
        assert "disabled" in str(error).lower()

    def test_koina_connection_error(self):
        """Test KoinaConnectionError."""
        from imspy_predictors.koina_models import KoinaConnectionError

        error = KoinaConnectionError("Cannot connect")
        assert str(error) == "Cannot connect"

    def test_koina_timeout_error(self):
        """Test KoinaTimeoutError."""
        from imspy_predictors.koina_models import KoinaTimeoutError

        error = KoinaTimeoutError("Request timed out")
        assert "timed out" in str(error)

    def test_koina_prediction_error_details(self):
        """Test KoinaPredictionError with details."""
        from imspy_predictors.koina_models import KoinaPredictionError

        error = KoinaPredictionError(
            "Prediction failed",
            model_name="Prosit_2023_intensity_timsTOF",
            batch_size=100,
            sample_peptides=["PEPTIDE", "SEQUENCE"],
        )

        error_str = str(error)
        assert "Prediction failed" in error_str
        assert "Prosit_2023_intensity_timsTOF" in error_str
        assert "100" in error_str


# =============================================================================
# Global Enable/Disable Tests
# =============================================================================

class TestKoinaEnableDisable:
    """Tests for global Koina enable/disable functionality."""

    def test_disable_and_enable(self):
        """Test disabling and re-enabling Koina."""
        from imspy_predictors.koina_models import (
            disable_koina,
            enable_koina,
            is_koina_disabled,
        )

        # Initially should be enabled
        enable_koina()  # Ensure clean state
        assert is_koina_disabled() is False

        # Disable
        disable_koina()
        assert is_koina_disabled() is True

        # Re-enable
        enable_koina()
        assert is_koina_disabled() is False


# =============================================================================
# Server Check Tests (without network)
# =============================================================================

class TestServerCheck:
    """Tests for server availability checking."""

    def test_check_koina_server_function_exists(self):
        """Test that check_koina_server function is available."""
        from imspy_predictors.koina_models import check_koina_server

        assert callable(check_koina_server)

    def test_check_invalid_host(self):
        """Test checking invalid host returns failure."""
        from imspy_predictors.koina_models import check_koina_server

        available, error = check_koina_server("invalid.host.that.does.not.exist:443", timeout=1.0)
        assert available is False
        assert error is not None


# =============================================================================
# Configuration Constants Tests
# =============================================================================

class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_default_host(self):
        """Test default Koina host is set."""
        from imspy_predictors.koina_models import DEFAULT_KOINA_HOST

        assert DEFAULT_KOINA_HOST == "koina.wilhelmlab.org:443"

    def test_default_timeout(self):
        """Test default timeout is reasonable."""
        from imspy_predictors.koina_models import DEFAULT_TIMEOUT_SECONDS

        assert DEFAULT_TIMEOUT_SECONDS >= 30
        assert DEFAULT_TIMEOUT_SECONDS <= 300

    def test_default_retries(self):
        """Test default retries is reasonable."""
        from imspy_predictors.koina_models import DEFAULT_MAX_RETRIES

        assert DEFAULT_MAX_RETRIES >= 1
        assert DEFAULT_MAX_RETRIES <= 10

    def test_default_retry_delay(self):
        """Test default retry delay is reasonable."""
        from imspy_predictors.koina_models import DEFAULT_RETRY_DELAY

        assert DEFAULT_RETRY_DELAY >= 0.5
        assert DEFAULT_RETRY_DELAY <= 10


# =============================================================================
# UNIMOD Mapping Tests
# =============================================================================

class TestUnimodMapping:
    """Tests for UNIMOD to AlphaBase mapping."""

    def test_common_mods_mapped(self):
        """Test common modifications are mapped."""
        from imspy_predictors.koina_models import UNIMOD_TO_ALPHABASE

        # Carbamidomethyl (UNIMOD:4)
        assert 4 in UNIMOD_TO_ALPHABASE
        assert UNIMOD_TO_ALPHABASE[4] == "Carbamidomethyl"

        # Phospho (UNIMOD:21)
        assert 21 in UNIMOD_TO_ALPHABASE
        assert UNIMOD_TO_ALPHABASE[21] == "Phospho"

        # Oxidation (UNIMOD:35)
        assert 35 in UNIMOD_TO_ALPHABASE
        assert UNIMOD_TO_ALPHABASE[35] == "Oxidation"

    def test_tmt_mapped(self):
        """Test TMT labels are mapped."""
        from imspy_predictors.koina_models import UNIMOD_TO_ALPHABASE

        assert 312 in UNIMOD_TO_ALPHABASE
        assert "TMT" in UNIMOD_TO_ALPHABASE[312]

    def test_silac_mapped(self):
        """Test SILAC labels are mapped."""
        from imspy_predictors.koina_models import UNIMOD_TO_ALPHABASE

        # Heavy Lys (UNIMOD:259)
        assert 259 in UNIMOD_TO_ALPHABASE
        assert "13C" in UNIMOD_TO_ALPHABASE[259]

        # Heavy Arg (UNIMOD:267)
        assert 267 in UNIMOD_TO_ALPHABASE
        assert "13C" in UNIMOD_TO_ALPHABASE[267]


# =============================================================================
# Integration Tests (require network, marked for optional execution)
# =============================================================================

@pytest.mark.skipif(
    True,  # Set to False to run network tests
    reason="Network tests disabled by default"
)
class TestKoinaIntegration:
    """Integration tests that require network access to Koina server.

    These tests are disabled by default. To run them:
    1. Ensure network access to koina.wilhelmlab.org
    2. Set the skipif condition to False
    """

    def test_server_is_reachable(self):
        """Test that Koina server is reachable."""
        from imspy_predictors.koina_models import check_koina_server

        available, error = check_koina_server()
        assert available is True, f"Koina server not available: {error}"

    def test_prosit_prediction(self):
        """Test Prosit intensity prediction via Koina."""
        from imspy_predictors.koina_models import ModelFromKoina

        model = ModelFromKoina("Prosit_2023_intensity_timsTOF")

        df = pd.DataFrame({
            "peptide_sequences": ["PEPTIDE", "SEQUENCE"],
            "precursor_charges": [2, 2],
            "collision_energies": [0.30, 0.30],
            "instrument_types": ["TIMSTOF", "TIMSTOF"],
        })

        result = model.predict(df)
        assert len(result) == 2

    def test_alphapeptdeep_phospho_prediction(self):
        """Test AlphaPeptDeep prediction with phospho peptides."""
        from imspy_predictors.koina_models import ModelFromKoina

        model = ModelFromKoina("AlphaPeptDeep_ms2_generic")

        df = pd.DataFrame({
            "peptide_sequences": [
                "AS[UNIMOD:21]DFK",  # Will be converted to AS(UniMod:21)DFK
                "PEPTIDE",
            ],
            "precursor_charges": [2, 2],
            "collision_energies": [0.30, 0.30],
        })

        result = model.predict(df)
        assert len(result) == 2
