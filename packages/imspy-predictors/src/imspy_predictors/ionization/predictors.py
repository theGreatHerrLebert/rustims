"""
Charge State Distribution Predictors.

This module provides deep learning models for predicting charge state
distributions for peptide sequences using PyTorch.

Classes:
    - DeepChargeStateDistribution: High-level wrapper for charge state prediction
    - PyTorchChargePredictor: PyTorch transformer/GRU-based model
    - BinomialChargeStateDistributionModel: Simple binomial-based model
"""

from typing import List, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from imspy_predictors.utility import get_model_path, get_device

import imspy_connector
ims = imspy_connector.py_chemistry


def charge_state_distribution_from_sequence_rust(
    sequence: str,
    max_charge: Optional[int] = None,
    charge_probability: Optional[float] = None,
) -> NDArray:
    """Calculate charge state distribution using Rust implementation."""
    return np.array(ims.simulate_charge_state_for_sequence(sequence, max_charge, charge_probability))


def charge_state_distributions_from_sequences_rust(
    sequences: List[str],
    n_threads: int = 4,
    max_charge: Optional[int] = None,
    charge_probability: Optional[float] = None,
) -> NDArray:
    """Calculate charge state distributions for multiple sequences using Rust."""
    return np.array(ims.simulate_charge_states_for_sequences(
        sequences, n_threads, max_charge, charge_probability
    ))


class PeptideChargeStateDistribution(ABC):
    """Abstract base class for charge state distribution prediction."""

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ionizations(self, sequences: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def simulate_charge_state_distribution_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class BinomialChargeStateDistributionModel(PeptideChargeStateDistribution):
    """
    Simple binomial-based charge state distribution model.

    Uses the Rust implementation for fast calculation.
    """

    def __init__(
        self,
        charged_probability: float = 0.8,
        max_charge: int = 4,
        normalize: bool = True,
    ):
        super().__init__()
        self.charged_probability = charged_probability
        self.max_charge = max_charge
        self.normalize = normalize

    def simulate_ionizations(self, sequences: List[str]) -> np.ndarray:
        return charge_state_distributions_from_sequences_rust(
            sequences,
            max_charge=self.max_charge,
            charge_probability=self.charged_probability,
        )

    def simulate_charge_state_distribution_pandas(
        self,
        data: pd.DataFrame,
        min_charge_contrib: float = 0.005,
    ) -> pd.DataFrame:
        probabilities = charge_state_distributions_from_sequences_rust(
            data.sequence.values.tolist(),
            max_charge=self.max_charge,
            charge_probability=self.charged_probability,
        )

        r_table = []
        for charges, (_, row) in tqdm(
            zip(probabilities, data.iterrows()),
            desc='flatmap charges',
            ncols=80,
            total=len(probabilities),
        ):
            if self.normalize:
                kept_prob_mass = np.sum(np.where(charges[1:] >= min_charge_contrib, charges[1:], 0))
                if kept_prob_mass > 0:
                    charges[1:] = charges[1:] / kept_prob_mass

            for i, charge in enumerate(charges[1:], start=1):
                if charge >= min_charge_contrib:
                    r_table.append({
                        'peptide_id': row.peptide_id,
                        'charge': i,
                        'relative_abundance': charge,
                    })

        return pd.DataFrame(r_table)


# =============================================================================
# PyTorch Implementation
# =============================================================================

if TORCH_AVAILABLE:

    class PyTorchChargePredictor(nn.Module):
        """
        PyTorch charge state predictor using GRU or Transformer encoder.

        Architecture options:
        - 'gru': Bidirectional GRU (similar to legacy TensorFlow model)
        - 'transformer': Transformer encoder (modern architecture)
        """

        def __init__(
            self,
            vocab_size: int,
            max_charge: int = 4,
            max_seq_len: int = 50,
            emb_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            architecture: str = 'gru',
        ):
            super().__init__()

            self.vocab_size = vocab_size
            self.max_charge = max_charge
            self.max_seq_len = max_seq_len
            self.architecture = architecture

            # Sequence embedding
            self.embedding = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)

            # Encoder
            if architecture == 'gru':
                self.encoder = nn.GRU(
                    emb_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                encoder_out_dim = hidden_dim * 2
            else:  # transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                encoder_out_dim = emb_dim

            # Output layers
            self.fc1 = nn.Linear(encoder_out_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(dropout)
            self.out = nn.Linear(64, max_charge)

        def forward(
            self,
            sequences: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                sequences: Token IDs (batch, seq_len)
                padding_mask: Boolean mask for padding (batch, seq_len)

            Returns:
                Charge state probabilities (batch, max_charge)
            """
            x = self.embedding(sequences)

            if self.architecture == 'gru':
                _, hidden = self.encoder(x)
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                if padding_mask is not None:
                    mask_expanded = padding_mask.unsqueeze(-1).float()
                    x = self.encoder(x, src_key_padding_mask=padding_mask)
                    x = (x * (1 - mask_expanded)).sum(dim=1) / (1 - mask_expanded).sum(dim=1)
                else:
                    x = self.encoder(x)
                    x = x.mean(dim=1)
                hidden = x

            x = self.dropout(F.relu(self.fc1(hidden)))
            x = F.relu(self.fc2(x))
            return F.softmax(self.out(x), dim=-1)


def load_deep_charge_state_predictor(backend: Optional[str] = None):
    """
    Load a pretrained charge state predictor model.

    Args:
        backend: Kept for backward compatibility, ignored (always uses PyTorch)

    Returns:
        Loaded PyTorch model
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for charge state prediction. "
            "Install with: pip install torch"
        )

    # Try to load UnifiedPeptideModel first (new architecture)
    try:
        from imspy_predictors.models import UnifiedPeptideModel
        model_path = get_model_path('charge/best_model.pt')
        if model_path.exists():
            model = UnifiedPeptideModel.from_pretrained(str(model_path), tasks=['charge'])
            return model
    except (ImportError, FileNotFoundError, AttributeError):
        pass

    # Try legacy model path
    try:
        model_path = get_model_path('charge/charge-model.pt')
        from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        tokenizer = ProformaTokenizer.with_defaults()
        vocab_size = tokenizer.vocab_size

        model = PyTorchChargePredictor(vocab_size=vocab_size)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "No pretrained charge model found. Please ensure model files are available."
    )


class DeepChargeStateDistribution(PeptideChargeStateDistribution):
    """
    High-level wrapper for charge state distribution prediction using PyTorch.

    Args:
        model: Pre-loaded model (optional, will load default if None)
        tokenizer: Tokenizer for sequences (optional, will load default if None)
        allowed_charges: Array of allowed charge states
        name: Model name for output columns
        verbose: Whether to print progress
        backend: Kept for backward compatibility, ignored (always uses PyTorch)

    Example:
        >>> predictor = DeepChargeStateDistribution()
        >>> charges = predictor.simulate_ionizations(["PEPTIDE", "SEQUENCE"])
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        allowed_charges: NDArray = np.array([1, 2, 3, 4]),
        name: str = 'gru_predictor',
        verbose: bool = True,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeepChargeStateDistribution. "
                "Install with: pip install torch"
            )

        self.backend = 'torch'  # Always torch

        # Load tokenizer
        if tokenizer is None:
            from imspy_predictors.utilities.tokenizers import ProformaTokenizer
            self.tokenizer = ProformaTokenizer.with_defaults()
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            self.model = load_deep_charge_state_predictor()
        else:
            self.model = model

        self.allowed_charges = allowed_charges
        self.name = name
        self.verbose = verbose
        self._device = get_device()
        self.model = self.model.to(self._device)

    def _preprocess_sequences(
        self, sequences: List[str], pad_len: int = 50
    ) -> torch.Tensor:
        """Tokenize and pad sequences."""
        result = self.tokenizer(sequences, padding=True, return_tensors='pt')
        tokens = result['input_ids']

        if tokens.shape[1] < pad_len:
            padding = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
            tokens = torch.cat([tokens, padding], dim=1)
        elif tokens.shape[1] > pad_len:
            tokens = tokens[:, :pad_len]

        return tokens.to(self._device)

    def simulate_ionizations(
        self,
        sequences: List[str],
        batch_size: int = 1024,
    ) -> NDArray:
        """
        Predict most likely charge state for each peptide.

        Args:
            sequences: List of peptide sequences
            batch_size: Batch size for prediction

        Returns:
            Most likely charge state for each peptide
        """
        self.model.eval()
        tokens = self._preprocess_sequences(sequences)

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                probs = self.model(batch_tokens)
                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)

        # Sample from probability distribution
        c_list = []
        for p in probabilities:
            c_list.append(np.random.choice(range(1, len(p) + 1), 1, p=p)[0])

        return np.array(c_list)

    def predict_probabilities(
        self,
        sequences: List[str],
        batch_size: int = 1024,
    ) -> NDArray:
        """
        Predict charge state probability distributions.

        Args:
            sequences: List of peptide sequences
            batch_size: Batch size for prediction

        Returns:
            Probability distributions of shape (n_samples, max_charge)
        """
        self.model.eval()
        tokens = self._preprocess_sequences(sequences)

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                probs = self.model(batch_tokens)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def simulate_charge_state_distribution_pandas(
        self,
        data: pd.DataFrame,
        charge_state_one_probability: float = 0.1,
        batch_size: int = 1024,
        min_charge_contrib: float = 0.005,
    ) -> pd.DataFrame:
        """
        Simulate charge state distribution for a DataFrame.

        Args:
            data: DataFrame with 'sequence' column
            charge_state_one_probability: Probability to add for charge state 1
            batch_size: Batch size for prediction
            min_charge_contrib: Minimum relative abundance to include

        Returns:
            DataFrame with charge state distributions
        """
        assert 0 <= charge_state_one_probability <= 1, (
            f'charge_state_one_probability must be in [0, 1], was: {charge_state_one_probability}'
        )

        sequences = data.sequence.values.tolist()
        probabilities = self.predict_probabilities(sequences, batch_size)

        # Add charge state 1 probability and normalize
        probabilities[:, 0] = probabilities[:, 0] + charge_state_one_probability
        probabilities = probabilities / np.expand_dims(np.sum(probabilities, axis=1), axis=1)

        r_table = []
        for charges, (_, row) in tqdm(
            zip(probabilities, data.iterrows()),
            desc='flatmap charges',
            ncols=80,
            total=len(probabilities),
        ):
            for i, charge in enumerate(charges, start=1):
                if charge >= min_charge_contrib:
                    r_table.append({
                        'peptide_id': row.peptide_id,
                        'charge': i,
                        'relative_abundance': charge,
                    })

        return pd.DataFrame(r_table)

    def to(self, device: str) -> "DeepChargeStateDistribution":
        """Move model to device."""
        self._device = device
        self.model = self.model.to(device)
        return self

    def __repr__(self):
        return f'DeepChargeStateDistribution(name={self.name}, backend={self.backend})'


def predict_peptide_flyability_with_koina(
    data: pd.DataFrame,
    model_name: str = "pfly_2024_fine_tuned",
    seq_col: str = "sequence",
    verbose: bool = False,
) -> np.ndarray:
    """
    Predict peptide flyability using Koina model.

    Args:
        data: DataFrame containing peptide sequences
        model_name: Name of the Koina model
        seq_col: Column name for peptide sequences
        verbose: Verbosity flag

    Returns:
        Predicted probability of being a flyer
    """
    from imspy_predictors.koina_models import ModelFromKoina

    flyability_model = ModelFromKoina(model_name=model_name)
    inputs = data.copy()
    inputs.rename(columns={seq_col: "peptide_sequences"}, inplace=True)
    flyability = flyability_model.predict(inputs[['peptide_sequences']])

    flyer_labels = ['non_flyer', 'weak_flyer', 'intermediate_flyer', 'strong_flyer']
    flyability['flyer_type'] = flyability.groupby('peptide_sequences').cumcount().map(
        dict(enumerate(flyer_labels))
    )

    flyability_wide = flyability.pivot(
        index='peptide_sequences', columns='flyer_type', values='output_1'
    ).reset_index()
    efficiency = 1 - flyability_wide['non_flyer']

    if verbose:
        print(f"Koina model {model_name} predicted flyability for {len(efficiency)} peptides.")

    return efficiency.values
