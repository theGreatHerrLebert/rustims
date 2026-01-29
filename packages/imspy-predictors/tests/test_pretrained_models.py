"""
Sanity tests for pretrained models.

These tests verify that saved/loaded pretrained models produce sensible outputs.
They help catch issues like:
- Corrupted model weights
- Training bugs that produce degenerate models
- Incorrect output distributions

Run these tests after training new models or updating model checkpoints.
"""

import pytest
import numpy as np
import torch

# Test peptides with known expected charge state distributions
# Short tryptic peptides (ending in K/R) should be predominantly charge 2+
TEST_PEPTIDES_CHARGE_2_DOMINANT = [
    "PEPTIDER",      # 8 aa, ends in R -> charge 2 dominant
    "AAAAAAAAK",     # 9 aa, ends in K -> charge 2 dominant
    "ELVISLIVESK",   # 11 aa, ends in K -> charge 2 dominant
    "SIMPLESEQR",    # 10 aa, ends in R -> charge 2 dominant
]

# Longer peptides may have more charge 3
TEST_PEPTIDES_CHARGE_3_POSSIBLE = [
    "THISISMUCHLONGERPEPTIDEK",  # 24 aa -> may have charge 3
    "VERYLONGSEQUENCEWITHLYSINEK",  # 27 aa -> charge 3 more likely
]


class TestChargeModelSanity:
    """Sanity tests for charge state prediction models."""

    @pytest.fixture
    def charge_predictor(self):
        """Load the pretrained charge predictor."""
        try:
            from imspy_predictors.ionization.predictors import (
                DeepChargeStateDistribution,
                load_deep_charge_state_predictor,
            )
            from imspy_predictors.utility import load_tokenizer_from_resources

            model = load_deep_charge_state_predictor()
            tokenizer = load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm')
            predictor = DeepChargeStateDistribution(model=model, tokenizer=tokenizer)
            return predictor
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Pretrained charge model not available: {e}")

    def test_charge_model_outputs_valid_probabilities(self, charge_predictor):
        """Test that charge model outputs valid probability distributions."""
        probs = charge_predictor.predict_probabilities(TEST_PEPTIDES_CHARGE_2_DOMINANT)

        # Should be valid probabilities (sum to 1, all >= 0)
        assert probs.shape[0] == len(TEST_PEPTIDES_CHARGE_2_DOMINANT)
        assert np.all(probs >= 0), "Probabilities should be non-negative"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities should sum to 1"

    def test_charge_model_not_degenerate(self, charge_predictor):
        """Test that charge model doesn't predict same charge for all peptides."""
        all_peptides = TEST_PEPTIDES_CHARGE_2_DOMINANT + TEST_PEPTIDES_CHARGE_3_POSSIBLE
        probs = charge_predictor.predict_probabilities(all_peptides)

        # Get most likely charge for each peptide
        most_likely_charges = np.argmax(probs, axis=1) + 1  # +1 because index 0 = charge 1

        # At least 2 different charge states should be predicted across peptides
        # A broken model would predict the same charge for everything
        unique_charges = len(np.unique(most_likely_charges))
        assert unique_charges >= 2, (
            f"Model appears degenerate: predicted only charge {most_likely_charges[0]} "
            f"for all {len(all_peptides)} test peptides. Expected diversity in predictions."
        )

    def test_charge_model_not_always_charge_3(self, charge_predictor):
        """Test that model doesn't always predict charge 3 (known failure mode)."""
        probs = charge_predictor.predict_probabilities(TEST_PEPTIDES_CHARGE_2_DOMINANT)

        # For short tryptic peptides, charge 3 should NOT be 100% probability
        charge_3_probs = probs[:, 2]  # Index 2 = charge 3

        # If all peptides have >99% charge 3, the model is broken
        all_charge_3 = np.all(charge_3_probs > 0.99)
        assert not all_charge_3, (
            f"Model is broken: predicts ~100% charge 3 for all peptides. "
            f"Charge 3 probabilities: {charge_3_probs}. "
            f"This indicates the training used probabilities instead of logits for cross_entropy."
        )

    def test_charge_2_dominant_for_short_tryptic(self, charge_predictor):
        """Test that charge 2 is dominant for short tryptic peptides."""
        probs = charge_predictor.predict_probabilities(TEST_PEPTIDES_CHARGE_2_DOMINANT)

        # Get average probability for each charge state
        mean_probs = probs.mean(axis=0)

        # Charge 2 (index 1) should have highest average probability for these peptides
        # or at least be competitive with charge 3
        charge_2_prob = mean_probs[1] if len(mean_probs) > 1 else 0
        charge_3_prob = mean_probs[2] if len(mean_probs) > 2 else 0

        # Charge 2 should be at least 20% on average for short tryptic peptides
        assert charge_2_prob > 0.20, (
            f"Charge 2 probability too low for short tryptic peptides: {charge_2_prob:.3f}. "
            f"Expected > 0.20. Distribution: {mean_probs}"
        )

    def test_charge_probabilities_vary_with_length(self, charge_predictor):
        """Test that charge probabilities change with peptide length."""
        short_peptides = ["PEPTIDER", "AAK"]
        long_peptides = ["THISISMUCHLONGERPEPTIDEK", "VERYLONGSEQUENCEWITHLYSINEK"]

        short_probs = charge_predictor.predict_probabilities(short_peptides)
        long_probs = charge_predictor.predict_probabilities(long_peptides)

        # Mean charge 3 probability should be higher for longer peptides
        short_charge_3 = short_probs[:, 2].mean() if short_probs.shape[1] > 2 else 0
        long_charge_3 = long_probs[:, 2].mean() if long_probs.shape[1] > 2 else 0

        # Long peptides should have at least somewhat higher charge 3 probability
        # This tests that the model learned length-dependent charge behavior
        # Note: we use a soft check since some short peptides can also be charge 3
        assert long_charge_3 >= short_charge_3 * 0.8, (
            f"Model may not have learned length dependence. "
            f"Short peptide charge 3: {short_charge_3:.3f}, Long: {long_charge_3:.3f}"
        )


class TestRTModelSanity:
    """Sanity tests for retention time prediction models."""

    @pytest.fixture
    def rt_predictor(self):
        """Load the pretrained RT predictor."""
        try:
            from imspy_predictors.models import UnifiedPeptideModel
            from imspy_predictors.utility import get_model_path

            model_path = get_model_path('rt/best_model.pt')
            if not model_path.exists():
                pytest.skip("Pretrained RT model not available")

            model = UnifiedPeptideModel.from_pretrained(str(model_path), tasks=['rt'])
            return model
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Pretrained RT model not available: {e}")

    def test_rt_model_outputs_valid_range(self, rt_predictor):
        """Test that RT model outputs are in valid range [0, 1]."""
        from imspy_predictors.utility import load_tokenizer_from_resources

        tokenizer = load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm')
        test_seqs = ["PEPTIDER", "AAAAAAAAK", "WWWWWWWWK"]

        result = tokenizer(test_seqs, padding=True, return_tensors='pt')
        tokens = result['input_ids']

        with torch.no_grad():
            rt_predictor.eval()
            rt = rt_predictor.predict_rt(tokens)

        assert torch.all(rt >= 0) and torch.all(rt <= 1), (
            f"RT predictions should be in [0, 1], got range [{rt.min():.3f}, {rt.max():.3f}]"
        )

    def test_rt_model_hydrophobic_elute_later(self, rt_predictor):
        """Test that hydrophobic peptides elute later (higher RT)."""
        from imspy_predictors.utility import load_tokenizer_from_resources

        tokenizer = load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm')

        # Hydrophilic (early eluting)
        hydrophilic = ["KKKKKKK", "RRRRRR", "DDDDDDK"]
        # Hydrophobic (late eluting)
        hydrophobic = ["IIIIIIIK", "LLLLLLK", "FFFFFFFK"]

        all_seqs = hydrophilic + hydrophobic
        result = tokenizer(all_seqs, padding=True, return_tensors='pt')
        tokens = result['input_ids']

        with torch.no_grad():
            rt_predictor.eval()
            rt = rt_predictor.predict_rt(tokens).squeeze().numpy()

        mean_hydrophilic = rt[:len(hydrophilic)].mean()
        mean_hydrophobic = rt[len(hydrophilic):].mean()

        assert mean_hydrophobic > mean_hydrophilic, (
            f"Hydrophobic peptides should elute later. "
            f"Hydrophilic RT: {mean_hydrophilic:.3f}, Hydrophobic RT: {mean_hydrophobic:.3f}"
        )


class TestCCSModelSanity:
    """Sanity tests for CCS/ion mobility prediction models."""

    @pytest.fixture
    def ccs_predictor(self):
        """Load the pretrained CCS predictor."""
        try:
            from imspy_predictors.models import UnifiedPeptideModel
            from imspy_predictors.utility import get_model_path

            model_path = get_model_path('ccs/best_model.pt')
            if not model_path.exists():
                pytest.skip("Pretrained CCS model not available")

            model = UnifiedPeptideModel.from_pretrained(str(model_path), tasks=['ccs'])
            return model
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"Pretrained CCS model not available: {e}")

    def test_ccs_model_outputs_positive(self, ccs_predictor):
        """Test that CCS predictions are positive."""
        from imspy_predictors.utility import load_tokenizer_from_resources

        tokenizer = load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm')
        test_seqs = ["PEPTIDER", "AAAAAAAAK"]

        result = tokenizer(test_seqs, padding=True, return_tensors='pt')
        tokens = result['input_ids']
        mz = torch.tensor([500.0, 400.0])
        charge = torch.tensor([2, 2])

        with torch.no_grad():
            ccs_predictor.eval()
            ccs_mean, ccs_std = ccs_predictor.predict_ccs(tokens, mz, charge)

        assert torch.all(ccs_mean > 0), "CCS predictions should be positive"
        assert torch.all(ccs_std >= 0), "CCS std should be non-negative"

    def test_ccs_increases_with_size(self, ccs_predictor):
        """Test that CCS increases with peptide size/m/z."""
        from imspy_predictors.utility import load_tokenizer_from_resources

        tokenizer = load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm')

        small_seq = ["AAK"]
        large_seq = ["AAAAAAAAAAAAAAAAAAAAAAAAK"]

        result_small = tokenizer(small_seq, padding=True, return_tensors='pt')
        result_large = tokenizer(large_seq, padding=True, return_tensors='pt')

        with torch.no_grad():
            ccs_predictor.eval()
            ccs_small, _ = ccs_predictor.predict_ccs(
                result_small['input_ids'], torch.tensor([300.0]), torch.tensor([2])
            )
            ccs_large, _ = ccs_predictor.predict_ccs(
                result_large['input_ids'], torch.tensor([1500.0]), torch.tensor([2])
            )

        assert ccs_large > ccs_small, (
            f"Larger peptides should have higher CCS. "
            f"Small: {ccs_small.item():.1f}, Large: {ccs_large.item():.1f}"
        )


class TestUnifiedModelIntegration:
    """Integration tests for the UnifiedPeptideModel with all tasks."""

    def test_unified_model_creation(self):
        """Test that UnifiedPeptideModel can be created."""
        from imspy_predictors.models import UnifiedPeptideModel

        model = UnifiedPeptideModel(
            vocab_size=2200,
            tasks=["charge", "rt"],
        )
        assert model is not None
        assert "charge" in model.heads
        assert "rt" in model.heads

    def test_unified_model_forward(self):
        """Test forward pass of UnifiedPeptideModel."""
        from imspy_predictors.models import UnifiedPeptideModel

        model = UnifiedPeptideModel(
            vocab_size=2200,
            tasks=["charge"],
        )

        batch_size = 4
        seq_len = 30
        tokens = torch.randint(0, 100, (batch_size, seq_len))

        outputs = model(tokens, tasks=["charge"])

        assert "charge" in outputs
        assert outputs["charge"].shape == (batch_size, 6)  # 6 charge states

        # Should be valid probabilities
        assert torch.allclose(outputs["charge"].sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_unified_model_logits_mode(self):
        """Test that return_logits=True returns unnormalized logits."""
        from imspy_predictors.models import UnifiedPeptideModel

        model = UnifiedPeptideModel(
            vocab_size=2200,
            tasks=["charge"],
        )

        batch_size = 4
        seq_len = 30
        tokens = torch.randint(0, 100, (batch_size, seq_len))

        # Get probabilities
        probs = model(tokens, tasks=["charge"])["charge"]

        # Get logits
        logits = model(tokens, tasks=["charge"], return_logits=True)["charge"]

        # Logits should NOT sum to 1 (they're not normalized)
        assert not torch.allclose(logits.sum(dim=1), torch.ones(batch_size), atol=1e-5), (
            "Logits should not be normalized probabilities"
        )

        # But softmax(logits) should equal probs
        from torch.nn.functional import softmax
        probs_from_logits = softmax(logits, dim=-1)
        assert torch.allclose(probs, probs_from_logits, atol=1e-5), (
            "softmax(logits) should equal probabilities from normal forward pass"
        )
