"""Tests for utility functions and classes."""

import pytest
import numpy as np
import tempfile
import os

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TestInMemoryCheckpoint:
    """Test suite for InMemoryCheckpoint."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def checkpoint(self):
        """Create a checkpoint for testing."""
        from imspy_predictors.utility import InMemoryCheckpoint
        return InMemoryCheckpoint(validation_target='val_loss', mode='min')

    def test_checkpoint_creation(self, checkpoint):
        """Test checkpoint can be created."""
        assert checkpoint is not None
        assert checkpoint.mode == 'min'
        assert checkpoint.best_weights is None

    def test_checkpoint_min_mode(self, model):
        """Test checkpoint in minimize mode."""
        from imspy_predictors.utility import InMemoryCheckpoint
        checkpoint = InMemoryCheckpoint(mode='min')

        # First step should always save
        assert not checkpoint.step(1.0, model)
        assert checkpoint.best_value == 1.0

        # Lower value should update
        assert not checkpoint.step(0.5, model)
        assert checkpoint.best_value == 0.5

        # Higher value should not update
        assert not checkpoint.step(0.8, model)
        assert checkpoint.best_value == 0.5

    def test_checkpoint_max_mode(self, model):
        """Test checkpoint in maximize mode."""
        from imspy_predictors.utility import InMemoryCheckpoint
        checkpoint = InMemoryCheckpoint(mode='max')

        # First step should always save
        assert not checkpoint.step(0.5, model)
        assert checkpoint.best_value == 0.5

        # Higher value should update
        assert not checkpoint.step(0.8, model)
        assert checkpoint.best_value == 0.8

        # Lower value should not update
        assert not checkpoint.step(0.6, model)
        assert checkpoint.best_value == 0.8

    def test_checkpoint_patience(self, model):
        """Test checkpoint early stopping with patience."""
        from imspy_predictors.utility import InMemoryCheckpoint
        checkpoint = InMemoryCheckpoint(mode='min', patience=3)

        # Improvement
        assert not checkpoint.step(1.0, model)
        assert checkpoint.wait == 0

        # No improvement
        assert not checkpoint.step(1.1, model)
        assert checkpoint.wait == 1

        assert not checkpoint.step(1.2, model)
        assert checkpoint.wait == 2

        # Should stop now
        assert checkpoint.step(1.3, model)
        assert checkpoint.wait == 3

    def test_checkpoint_restore(self, model):
        """Test restoring best weights."""
        from imspy_predictors.utility import InMemoryCheckpoint
        checkpoint = InMemoryCheckpoint(mode='min')

        # Store initial weights
        initial_weight = model.fc.weight.clone()

        # First step saves
        checkpoint.step(1.0, model)

        # Modify model weights
        with torch.no_grad():
            model.fc.weight.fill_(999.0)

        # Restore should bring back best weights
        assert checkpoint.restore(model)
        assert torch.allclose(model.fc.weight, initial_weight)

    def test_checkpoint_reset(self, checkpoint):
        """Test resetting checkpoint state."""
        checkpoint.best_value = 0.5
        checkpoint.wait = 5

        checkpoint.reset()

        assert checkpoint.best_value == float('inf')
        assert checkpoint.wait == 0
        assert checkpoint.best_weights is None

    def test_checkpoint_verbose(self, model, capsys):
        """Test verbose output."""
        from imspy_predictors.utility import InMemoryCheckpoint
        checkpoint = InMemoryCheckpoint(mode='min', verbose=True)

        checkpoint.step(1.0, model)
        captured = capsys.readouterr()
        assert "Improved" in captured.out


class TestSaveLoadCheckpoint:
    """Test suite for save/load checkpoint functions."""

    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return SimpleModel()

    def test_save_load_model(self, model):
        """Test saving and loading model checkpoint."""
        from imspy_predictors.utility import save_model_checkpoint, load_model_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')

            # Save model
            save_model_checkpoint(model, path, epoch=10, metadata={'test': 'value'})

            # Create new model and load
            model2 = SimpleModel()
            checkpoint = load_model_checkpoint(path, model=model2)

            assert 'model_state_dict' in checkpoint
            assert checkpoint['epoch'] == 10
            assert checkpoint['metadata']['test'] == 'value'

            # Weights should match
            assert torch.allclose(model.fc.weight, model2.fc.weight)

    def test_save_load_with_optimizer(self, model):
        """Test saving and loading with optimizer state."""
        from imspy_predictors.utility import save_model_checkpoint, load_model_checkpoint

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Do some training steps to modify optimizer state
        for _ in range(5):
            x = torch.randn(4, 10)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')

            save_model_checkpoint(model, path, optimizer=optimizer, epoch=5)

            # Load into new model and optimizer
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

            checkpoint = load_model_checkpoint(path, model=model2, optimizer=optimizer2)

            assert 'optimizer_state_dict' in checkpoint


class TestCountParameters:
    """Test suite for count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        from imspy_predictors.utility import count_parameters

        model = SimpleModel(input_dim=10, output_dim=1)
        total = count_parameters(model, trainable_only=False)

        # Linear layer: 10 weights + 1 bias = 11
        assert total == 11

    def test_count_trainable_only(self):
        """Test counting trainable parameters only."""
        from imspy_predictors.utility import count_parameters

        model = SimpleModel(input_dim=10, output_dim=1)

        # Freeze some parameters
        model.fc.weight.requires_grad = False

        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)

        assert trainable == 1  # Only bias
        assert total == 11  # All parameters


class TestGetDevice:
    """Test suite for get_device function."""

    def test_get_device_cpu(self):
        """Test getting CPU device."""
        from imspy_predictors.utility import get_device

        device = get_device(prefer_cuda=False)
        # Should return 'cpu' or 'mps' if available
        assert device in ['cpu', 'mps']

    def test_get_device_prefer_cuda(self):
        """Test device selection with CUDA preference."""
        from imspy_predictors.utility import get_device

        device = get_device(prefer_cuda=True)

        if torch.cuda.is_available():
            assert device == 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert device == 'mps'
        else:
            assert device == 'cpu'


class TestResourcePaths:
    """Test suite for resource path functions."""

    def test_get_model_path(self, monkeypatch):
        """A real model name resolves, and does so without downloading anything.

        This used to pass `'test_model'` and assert the name appeared in the returned string, from
        when `get_model_path` merely composed a path and never checked anything. It resolves against
        a registry now and falls through to `ensure_model`, so a made-up name is a `ValueError` —
        the test had been red ever since, asserting behaviour the function deliberately dropped.

        `ensure_model` is stubbed rather than trusted to be unreachable. The bundled `.pt` files
        make step 1 succeed in a source checkout, but pyproject excludes
        `src/imspy_predictors/pretrained/**/*.pt` from the wheel, so in a wheel or clean CI install
        the same call falls through and fetches from GitHub Releases. Asserting "this never hits the
        network" while depending on a file the wheel omits would be true here and false in CI.
        """
        import pathlib

        from imspy_predictors.utility import get_model_path

        calls = []

        def _stub_ensure_model(model_name):
            calls.append(model_name)
            return pathlib.Path('/nonexistent/cache') / model_name

        monkeypatch.setattr(
            'imspy_predictors.pretrained.hub.ensure_model', _stub_ensure_model
        )

        path = get_model_path('ccs/best_model.pt')

        # Either step 1 found the bundled copy (source checkout) or it delegated to the stub
        # (wheel). Both are correct; neither touches the network.
        assert 'best_model.pt' in str(path)
        assert calls in ([], ['ccs/best_model.pt'])

    def test_get_model_path_falls_through_to_hub_when_not_bundled(self, monkeypatch):
        """The wheel case: `.pt` files are excluded from the wheel, so step 1 misses and the hub
        does the work. That is the branch CI takes and the one a source checkout never exercises,
        so the bundled lookup is forced to miss here rather than left to chance."""
        import pathlib

        from imspy_predictors.utility import get_model_path

        class _Missing:
            def joinpath(self, *_):
                return self

            def is_file(self):
                return False

        monkeypatch.setattr(
            'imspy_predictors.utility.resources.files', lambda _pkg: _Missing()
        )

        calls = []

        def _stub_ensure_model(model_name):
            calls.append(model_name)
            return pathlib.Path('/nonexistent/cache') / model_name

        monkeypatch.setattr(
            'imspy_predictors.pretrained.hub.ensure_model', _stub_ensure_model
        )

        path = get_model_path('ccs/best_model.pt')

        assert calls == ['ccs/best_model.pt'], "should have delegated to the hub"
        assert 'best_model.pt' in str(path)

    def test_get_model_path_rejects_unknown_model(self):
        """An unknown name must name the known ones, not return an unusable path or 404 later."""
        import pytest

        from imspy_predictors.utility import get_model_path

        with pytest.raises(ValueError) as ei:
            get_model_path('test_model')

        msg = str(ei.value)
        assert 'test_model' in msg
        assert 'ccs/best_model.pt' in msg, "the error should list what IS available"

    def test_get_tokenizer_path(self):
        """Test getting tokenizer path."""
        from imspy_predictors.utility import get_tokenizer_path

        path = get_tokenizer_path('tokenizer')
        assert 'tokenizer.json' in str(path)
