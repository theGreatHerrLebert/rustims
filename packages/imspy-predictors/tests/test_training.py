"""Tests for training utilities."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import torch
import torch.nn as nn

# Import directly to avoid imspy_core dependency
from imspy_predictors.training import (
    TrainingConfig,
    EarlyStopping,
    MetricTracker,
    PerformanceLogger,
    Trainer,
    compute_regression_metrics,
    compute_spectral_angle,
    compute_intensity_metrics,
    compute_classification_metrics,
)


class SimpleModel(nn.Module):
    """Simple model for testing training utilities."""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, **kwargs):
        x = torch.relu(self.fc1(x))
        return {"output": self.fc2(x)}


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.early_stopping_patience == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            epochs=50,
            batch_size=128,
            learning_rate=1e-3,
        )

        assert config.epochs == 50
        assert config.batch_size == 128
        assert config.learning_rate == 1e-3


class TestEarlyStopping:
    """Test suite for EarlyStopping."""

    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return SimpleModel()

    def test_early_stopping_improvement(self, model):
        """Test early stopping detects improvement."""
        es = EarlyStopping(patience=3, mode="min")

        # First call sets baseline
        assert not es(1.0, model)
        assert es.best_score == 1.0

        # Improvement
        assert not es(0.9, model)
        assert es.best_score == 0.9
        assert es.counter == 0

    def test_early_stopping_no_improvement(self, model):
        """Test early stopping counts non-improvements."""
        es = EarlyStopping(patience=3, mode="min")

        es(1.0, model)
        es(1.1, model)  # No improvement
        assert es.counter == 1

        es(1.2, model)  # No improvement
        assert es.counter == 2

    def test_early_stopping_triggers(self, model):
        """Test early stopping triggers after patience exceeded."""
        es = EarlyStopping(patience=2, mode="min")

        es(1.0, model)
        es(1.1, model)
        es(1.2, model)
        assert es(1.3, model)  # Should trigger
        assert es.should_stop

    def test_early_stopping_max_mode(self, model):
        """Test early stopping in max mode."""
        es = EarlyStopping(patience=2, mode="max")

        assert not es(0.5, model)
        assert not es(0.8, model)  # Improvement
        assert es.best_score == 0.8

        es(0.7, model)  # No improvement
        assert es.counter == 1

    def test_restore_best_weights(self, model):
        """Test restoring best weights."""
        es = EarlyStopping(patience=3, mode="min")

        # Record initial weight
        initial_weight = model.fc1.weight.clone()

        # First call saves weights
        es(1.0, model)

        # Modify model
        with torch.no_grad():
            model.fc1.weight.fill_(999.0)

        # Restore
        es.restore_best_weights(model)

        assert torch.allclose(model.fc1.weight, initial_weight)


class TestMetricTracker:
    """Test suite for MetricTracker."""

    def test_update_and_average(self):
        """Test metric tracking and averaging."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0, "accuracy": 0.8}, count=10)
        tracker.update({"loss": 2.0, "accuracy": 0.9}, count=10)

        avg = tracker.average()
        assert avg["loss"] == 1.5
        assert avg["accuracy"] == 0.85

    def test_reset(self):
        """Test resetting metrics."""
        tracker = MetricTracker()
        tracker.update({"loss": 1.0})

        tracker.reset()

        assert len(tracker.metrics) == 0
        assert len(tracker.counts) == 0


class TestPerformanceLogger:
    """Test suite for PerformanceLogger."""

    def test_logger_creation(self):
        """Test logger can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PerformanceLogger(
                log_dir=tmpdir,
                experiment_name="test_experiment",
                use_tensorboard=False,
            )

            assert logger.experiment_name == "test_experiment"
            assert logger.experiment_dir.exists()

            logger.close()

    def test_log_epoch(self):
        """Test logging epoch metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PerformanceLogger(
                log_dir=tmpdir,
                experiment_name="test",
                use_tensorboard=False,
            )

            logger.log_epoch(
                epoch=1,
                train_metrics={"loss": 1.0},
                val_metrics={"loss": 0.8},
                lr=1e-4,
            )

            # Check log file was written
            assert logger.log_file.exists()

            # Check epoch was recorded
            assert len(logger.log_data["epochs"]) == 1
            assert logger.log_data["epochs"][0]["epoch"] == 1

            logger.close()

    def test_log_hyperparameters(self):
        """Test logging hyperparameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PerformanceLogger(
                log_dir=tmpdir,
                experiment_name="test",
                use_tensorboard=False,
            )

            config = TrainingConfig(epochs=50, learning_rate=1e-3)
            logger.log_hyperparameters(config)

            assert "hyperparameters" in logger.log_data
            assert logger.log_data["hyperparameters"]["epochs"] == 50

            logger.close()

    def test_log_model_info(self):
        """Test logging model information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PerformanceLogger(
                log_dir=tmpdir,
                experiment_name="test",
                use_tensorboard=False,
            )

            model = SimpleModel()
            logger.log_model_info(model)

            assert "model_info" in logger.log_data
            assert logger.log_data["model_info"]["architecture"] == "SimpleModel"
            assert logger.log_data["model_info"]["total_parameters"] > 0

            logger.close()


class TestRegressionMetrics:
    """Test suite for regression metrics."""

    def test_compute_regression_metrics(self):
        """Test computing regression metrics."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.1, 4.9])

        metrics = compute_regression_metrics(predictions, targets)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "medae" in metrics
        assert "pearson_r" in metrics

        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["r2"] <= 1.0

    def test_perfect_prediction(self):
        """Test metrics for perfect prediction."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert abs(metrics["r2"] - 1.0) < 0.01


class TestSpectralAngle:
    """Test suite for spectral angle computation."""

    def test_spectral_angle_identical(self):
        """Test spectral angle for identical vectors."""
        pred = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        target = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])

        angle = compute_spectral_angle(pred, target)
        assert abs(angle - 1.0) < 0.01  # Perfect similarity

    def test_spectral_angle_orthogonal(self):
        """Test spectral angle for orthogonal vectors."""
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])

        angle = compute_spectral_angle(pred, target)
        assert abs(angle) < 0.01  # No similarity


class TestIntensityMetrics:
    """Test suite for intensity metrics."""

    def test_compute_intensity_metrics(self):
        """Test computing intensity metrics."""
        predictions = torch.rand(10, 174)
        targets = torch.rand(10, 174)

        metrics = compute_intensity_metrics(predictions, targets)

        assert "spectral_angle" in metrics
        assert "cosine_similarity" in metrics
        assert "pearson_correlation" in metrics


class TestClassificationMetrics:
    """Test suite for classification metrics."""

    def test_compute_classification_metrics(self):
        """Test computing classification metrics."""
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ])
        targets = torch.tensor([0, 1, 2])

        metrics = compute_classification_metrics(predictions, targets)

        assert "accuracy" in metrics
        assert "top2_accuracy" in metrics
        assert "cross_entropy" in metrics

        assert metrics["accuracy"] == 1.0  # All correct
        assert metrics["top2_accuracy"] == 1.0

    def test_classification_metrics_with_distributions(self):
        """Test classification metrics with distribution targets."""
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ])
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        metrics = compute_classification_metrics(predictions, targets)
        assert metrics["accuracy"] == 1.0


class TestTrainerCheckpoints:
    """Test suite for Trainer checkpointing."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    def test_save_and_load_checkpoint(self, model):
        """Test saving and loading checkpoints."""
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig(checkpoint_dir=None)

        # Create trainer (without actually training)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=lambda b: torch.tensor(0.0),
            config=config,
        )

        # Manually add to history
        trainer.history["train_loss"] = [1.0, 0.9, 0.8]
        trainer.history["val_loss"] = [1.1, 0.95, 0.85]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            trainer._save_model(path, epoch=3, val_loss=0.85)

            # Load into new model
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters())

            model2, checkpoint = Trainer.load_checkpoint(
                str(path),
                model=model2,
                optimizer=optimizer2,
            )

            assert checkpoint["epoch"] == 3
            assert checkpoint["val_loss"] == 0.85
            assert "history" in checkpoint
