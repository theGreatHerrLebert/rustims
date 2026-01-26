"""Tests for PyTorch model architectures."""

import pytest
import torch
import numpy as np


class TestPeptideTransformer:
    """Test suite for PeptideTransformer encoder."""

    @pytest.fixture
    def transformer(self):
        """Create a small transformer for testing."""
        from imspy_predictors.models.transformer import PeptideTransformer
        return PeptideTransformer(
            vocab_size=100,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_seq_len=50,
            dropout=0.1,
        )

    def test_transformer_creation(self, transformer):
        """Test transformer can be created."""
        assert transformer is not None

    def test_forward_pass(self, transformer):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        output = transformer(tokens)

        assert output.shape == (batch_size, seq_len, 64)

    def test_forward_with_padding_mask(self, transformer):
        """Test forward pass with padding mask."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, 15:] = True  # Mask last 5 tokens

        output = transformer(tokens, padding_mask=padding_mask)
        assert output.shape == (batch_size, seq_len, 64)

    def test_cls_token_output(self, transformer):
        """Test extracting CLS token representation."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        output = transformer(tokens)

        # CLS token is first position
        cls_output = output[:, 0, :]
        assert cls_output.shape == (batch_size, 64)

    def test_gradient_flow(self, transformer):
        """Test that gradients flow through the model."""
        tokens = torch.randint(0, 100, (2, 10))
        output = transformer(tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in transformer.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCCSHead:
    """Test suite for CCS prediction head."""

    @pytest.fixture
    def ccs_head(self):
        """Create a CCS head for testing."""
        from imspy_predictors.models.heads import CCSHead
        return CCSHead(d_model=64, max_charge=4)

    def test_ccs_head_creation(self, ccs_head):
        """Test CCS head can be created."""
        assert ccs_head is not None

    def test_forward_pass(self, ccs_head):
        """Test forward pass produces correct output shape."""
        batch_size = 4

        encoder_out = torch.randn(batch_size, 10, 64)
        mz = torch.rand(batch_size) * 1000 + 400
        charge_onehot = torch.zeros(batch_size, 4)
        charge_onehot[:, 1] = 1  # Charge 2

        ccs, std = ccs_head(encoder_out, mz, charge_onehot)

        assert ccs.shape == (batch_size, 1)
        assert std.shape == (batch_size, 1)
        assert (std > 0).all()  # Std should be positive


class TestRTHead:
    """Test suite for RT prediction head."""

    @pytest.fixture
    def rt_head(self):
        """Create an RT head for testing."""
        from imspy_predictors.models.heads import RTHead
        return RTHead(d_model=64)

    def test_rt_head_creation(self, rt_head):
        """Test RT head can be created."""
        assert rt_head is not None

    def test_forward_pass(self, rt_head):
        """Test forward pass produces correct output shape."""
        batch_size = 4

        encoder_out = torch.randn(batch_size, 10, 64)
        rt = rt_head(encoder_out)

        assert rt.shape == (batch_size, 1)


class TestChargeHead:
    """Test suite for charge state prediction head."""

    @pytest.fixture
    def charge_head(self):
        """Create a charge head for testing."""
        from imspy_predictors.models.heads import ChargeHead
        return ChargeHead(d_model=64, max_charge=6)

    def test_charge_head_creation(self, charge_head):
        """Test charge head can be created."""
        assert charge_head is not None

    def test_forward_pass(self, charge_head):
        """Test forward pass produces correct output shape."""
        batch_size = 4

        encoder_out = torch.randn(batch_size, 10, 64)
        probs = charge_head(encoder_out)

        assert probs.shape == (batch_size, 6)
        # Should be valid probabilities
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestIntensityHead:
    """Test suite for intensity prediction head."""

    @pytest.fixture
    def intensity_head(self):
        """Create an intensity head for testing."""
        from imspy_predictors.models.heads import IntensityHead
        return IntensityHead(d_model=64, num_outputs=174, max_charge=6)

    def test_intensity_head_creation(self, intensity_head):
        """Test intensity head can be created."""
        assert intensity_head is not None

    def test_forward_pass(self, intensity_head):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        seq_len = 20

        encoder_out = torch.randn(batch_size, seq_len, 64)
        charge_onehot = torch.zeros(batch_size, 6)
        charge_onehot[:, 1] = 1  # Charge 2
        collision_energy = torch.rand(batch_size, 1) * 0.3 + 0.2

        intensities = intensity_head(encoder_out, charge_onehot, collision_energy)

        assert intensities.shape == (batch_size, 174)


class TestUnifiedModel:
    """Test suite for UnifiedPeptideModel."""

    @pytest.fixture
    def unified_model(self):
        """Create a unified model for testing."""
        from imspy_predictors.models.unified import UnifiedPeptideModel
        return UnifiedPeptideModel(
            vocab_size=100,
            encoder_config="small",
            tasks=["ccs", "rt", "charge"],
            max_charge=4,
            use_instrument=False,
        )

    def test_unified_model_creation(self, unified_model):
        """Test unified model can be created."""
        assert unified_model is not None
        assert hasattr(unified_model, "encoder")
        assert hasattr(unified_model, "ccs_head")
        assert hasattr(unified_model, "rt_head")
        assert hasattr(unified_model, "charge_head")

    def test_forward_ccs(self, unified_model):
        """Test CCS prediction."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        mz = torch.rand(batch_size) * 1000 + 400
        charge = torch.randint(1, 5, (batch_size,))

        outputs = unified_model(tokens, mz=mz, charge=charge, task="ccs")

        assert "ccs" in outputs
        assert "ccs_std" in outputs
        assert outputs["ccs"].shape == (batch_size, 1)

    def test_forward_rt(self, unified_model):
        """Test RT prediction."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        outputs = unified_model(tokens, task="rt")

        assert "rt" in outputs
        assert outputs["rt"].shape == (batch_size, 1)

    def test_forward_charge(self, unified_model):
        """Test charge state prediction."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        outputs = unified_model(tokens, task="charge")

        assert "charge_probs" in outputs
        assert outputs["charge_probs"].shape == (batch_size, 4)

    def test_forward_all_tasks(self, unified_model):
        """Test predicting all tasks at once."""
        batch_size = 4
        seq_len = 20

        tokens = torch.randint(0, 100, (batch_size, seq_len))
        mz = torch.rand(batch_size) * 1000 + 400
        charge = torch.randint(1, 5, (batch_size,))

        outputs = unified_model(tokens, mz=mz, charge=charge, task="all")

        assert "ccs" in outputs
        assert "rt" in outputs
        assert "charge_probs" in outputs


class TestSquareRootProjectionLayer:
    """Test suite for physics-based CCS initialization."""

    def test_sqrt_projection(self):
        """Test square root projection layer."""
        from imspy_predictors.ccs.predictors import SquareRootProjectionLayerPyTorch

        layer = SquareRootProjectionLayerPyTorch()

        batch_size = 4
        mz = torch.rand(batch_size) * 1000 + 400
        charge_onehot = torch.zeros(batch_size, 5)
        charge_onehot[:, 2] = 1  # Charge 2 (0-indexed as position 2)

        output = layer(mz, charge_onehot)

        assert output.shape == (batch_size, 1)
        # Output should be positive CCS values
        assert (output > 0).all()
