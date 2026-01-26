"""
Retention Time Predictors.

This module provides deep learning models for predicting chromatographic
retention times for peptide sequences using PyTorch.

Classes:
    - DeepChromatographyApex: High-level wrapper for RT prediction
    - PyTorchRTPredictor: PyTorch transformer/GRU-based model
"""

from typing import List, Union, Optional
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

from imspy_predictors.utility import (
    get_model_path,
    InMemoryCheckpoint,
    get_device,
)


# Lazy import for sagepy (optional dependency, requires imspy-search)
def _get_sagepy_utils():
    """Lazy import of sagepy utilities. Requires imspy-search package."""
    try:
        from sagepy.core.scoring import Psm
        from sagepy.utility import psm_collection_to_pandas
        return Psm, psm_collection_to_pandas
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


# Lazy import for dbsearch utility (optional, requires imspy-search)
def _get_dbsearch_utils():
    """Lazy import of dbsearch utilities. Requires imspy-search package."""
    try:
        from imspy_search.utility import linear_map, generate_balanced_rt_dataset
        return linear_map, generate_balanced_rt_dataset
    except ImportError:
        raise ImportError(
            "dbsearch utilities require imspy-search package."
        )


def linear_map(x, old_min, old_max, new_min, new_max):
    """Map values from one range to another linearly."""
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def predict_retention_time(
    psm_collection: List,
    refine_model: bool = True,
    verbose: bool = False,
) -> None:
    """
    Predict retention times for a collection of peptide-spectrum matches.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: A list of peptide-spectrum matches (sagepy Psm objects)
        refine_model: Whether to refine the model
        verbose: Whether to print verbose output

    Returns:
        None, retention times are set in the peptide-spectrum matches
    """
    Psm, psm_collection_to_pandas = _get_sagepy_utils()
    _linear_map, generate_balanced_rt_dataset = _get_dbsearch_utils()

    rt_predictor = DeepChromatographyApex(verbose=verbose)

    rt_min = np.min([x.retention_time for x in psm_collection])
    rt_max = np.max([x.retention_time for x in psm_collection])

    for psm in psm_collection:
        psm.retention_time_projected = _linear_map(
            psm.retention_time, old_min=rt_min, old_max=rt_max, new_min=0, new_max=60
        )

    if refine_model:
        rt_predictor.fine_tune_model(
            psm_collection_to_pandas(generate_balanced_rt_dataset(psm_collection)),
            batch_size=128,
            verbose=verbose,
        )

    # Predict retention times
    rt_predicted = rt_predictor.simulate_separation_times(
        sequences=[
            x.sequence_modified if not x.decoy else x.sequence_decoy_modified
            for x in psm_collection
        ],
    )

    # Set the predicted retention times
    for rt, ps in zip(rt_predicted, psm_collection):
        ps.retention_time_predicted = rt


class PeptideChromatographyApex(ABC):
    """Abstract base class for chromatographic separation prediction."""

    def __init__(self):
        pass

    @abstractmethod
    def simulate_separation_times(self, sequences: List[str]) -> NDArray:
        pass

    @abstractmethod
    def simulate_separation_times_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


# =============================================================================
# PyTorch Implementation
# =============================================================================

if TORCH_AVAILABLE:

    class PyTorchRTPredictor(nn.Module):
        """
        PyTorch retention time predictor using GRU or Transformer encoder.

        Architecture options:
        - 'gru': Bidirectional GRU (similar to legacy TensorFlow model)
        - 'transformer': Transformer encoder (modern architecture)
        """

        def __init__(
            self,
            vocab_size: int,
            max_seq_len: int = 50,
            emb_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            architecture: str = 'gru',
        ):
            super().__init__()

            self.vocab_size = vocab_size
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
            self.out = nn.Linear(64, 1)

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
                Predicted retention times (batch, 1)
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
            return self.out(x)


def load_deep_retention_time_predictor(backend: Optional[str] = None):
    """
    Load a pretrained retention time predictor model.

    Args:
        backend: Kept for backward compatibility, ignored (always uses PyTorch)

    Returns:
        Loaded PyTorch model
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for retention time prediction. "
            "Install with: pip install torch"
        )

    # Try to load UnifiedPeptideModel first (new architecture)
    try:
        from imspy_predictors.models import UnifiedPeptideModel
        model_path = get_model_path('rt/best_model.pt')
        if model_path.exists():
            model = UnifiedPeptideModel.from_pretrained(str(model_path), tasks=['rt'])
            return model
    except (ImportError, FileNotFoundError, AttributeError):
        pass

    # Try legacy model path
    try:
        model_path = get_model_path('rt/rt-model.pt')
        from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        tokenizer = ProformaTokenizer.with_defaults()
        vocab_size = tokenizer.vocab_size

        model = PyTorchRTPredictor(vocab_size=vocab_size)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        "No pretrained RT model found. Please ensure model files are available."
    )


class DeepChromatographyApex(PeptideChromatographyApex):
    """
    High-level wrapper for retention time prediction using PyTorch.

    Args:
        model: Pre-loaded model (optional, will load default if None)
        tokenizer: Tokenizer for sequences (optional, will load default if None)
        name: Model name for output columns
        verbose: Whether to print progress
        backend: Kept for backward compatibility, ignored (always uses PyTorch)

    Example:
        >>> predictor = DeepChromatographyApex()
        >>> rt = predictor.simulate_separation_times(["PEPTIDE", "SEQUENCE"])
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        name: str = 'gru_predictor',
        verbose: bool = False,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeepChromatographyApex. "
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
            self.model = load_deep_retention_time_predictor()
        else:
            self.model = model

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

    def simulate_separation_times(
        self,
        sequences: List[str],
        batch_size: int = 1024,
    ) -> NDArray:
        """
        Predict retention times for peptide sequences.

        Args:
            sequences: List of peptide sequences
            batch_size: Batch size for prediction

        Returns:
            Predicted retention times
        """
        self.model.eval()
        tokens = self._preprocess_sequences(sequences)

        all_rt = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                rt = self.model(batch_tokens)
                all_rt.append(rt.cpu().numpy())

        return np.concatenate(all_rt, axis=0).flatten()

    def fine_tune_model(
        self,
        data: pd.DataFrame,
        batch_size: int = 64,
        epochs: int = 150,
        learning_rate: float = 1e-3,
        patience: int = 6,
        verbose: bool = False,
        decoys_separate: bool = True,
    ) -> None:
        """
        Fine-tune the model on new data.

        Args:
            data: DataFrame with columns: sequence, retention_time_projected
            batch_size: Training batch size
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to print progress
            decoys_separate: Whether to handle decoys separately
        """
        assert 'sequence' in data.columns, 'Data must contain column "sequence"'
        assert 'retention_time_projected' in data.columns, 'Data must contain column "retention_time_projected"'

        if decoys_separate:
            sequences = []
            for _, row in data.iterrows():
                if not row.decoy:
                    sequences.append(row.sequence)
                else:
                    sequences.append(row.sequence_decoy_modified)
        else:
            sequences = list(data.sequence.values)

        rts = data.retention_time_projected.values
        self._fine_tune(sequences, rts, batch_size, epochs, learning_rate, patience, verbose)

    def _fine_tune(
        self,
        sequences: List[str],
        rts: NDArray,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        patience: int,
        verbose: bool,
    ) -> None:
        """PyTorch fine-tuning."""
        from torch.utils.data import DataLoader, TensorDataset

        tokens = self._preprocess_sequences(sequences)
        rt_tensor = torch.tensor(rts, dtype=torch.float32, device=self._device).unsqueeze(1)

        n = len(sequences)
        n_train = int(0.8 * n)
        indices = torch.randperm(n)

        train_dataset = TensorDataset(tokens[indices[:n_train]], rt_tensor[indices[:n_train]])
        val_dataset = TensorDataset(tokens[indices[n_train:]], rt_tensor[indices[n_train:]])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, min_lr=1e-6)
        checkpoint = InMemoryCheckpoint(patience=patience)

        for epoch in range(epochs):
            self.model.train()
            for tokens_b, rt_b in train_loader:
                optimizer.zero_grad()
                pred = self.model(tokens_b)
                loss = F.l1_loss(pred, rt_b)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for tokens_b, rt_b in val_loader:
                    pred = self.model(tokens_b)
                    val_loss += F.l1_loss(pred, rt_b).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

            if checkpoint.step(val_loss, self.model):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        checkpoint.restore(self.model)
        self.model.eval()

    def simulate_separation_times_pandas(
        self,
        data: pd.DataFrame,
        batch_size: int = 1024,
        gradient_length: Optional[float] = None,
        decoys_separate: bool = True,
    ) -> pd.DataFrame:
        """
        Predict retention times for a DataFrame.

        Args:
            data: DataFrame with sequence column
            batch_size: Batch size for prediction
            gradient_length: If provided, scale predictions to this gradient length
            decoys_separate: Handle decoys separately

        Returns:
            DataFrame with predicted retention times
        """
        assert 'sequence' in data.columns, 'Data must contain column "sequence"'

        if decoys_separate:
            sequences = []
            for _, row in data.iterrows():
                if not row.decoy:
                    sequences.append(getattr(row, 'sequence_modified', row.sequence))
                else:
                    sequences.append(getattr(row, 'sequence_decoy_modified', getattr(row, 'sequence_decoy', row.sequence)))
        else:
            sequences = list(data.sequence.values)

        rts = self.simulate_separation_times(sequences, batch_size)

        if gradient_length is not None:
            rts = linear_map(rts, old_min=rts.min(), old_max=rts.max(), new_min=0, new_max=gradient_length)

        data[f"retention_time_{self.name}"] = rts
        return data

    def to(self, device: str) -> "DeepChromatographyApex":
        """Move model to device."""
        self._device = device
        self.model = self.model.to(device)
        return self

    def __repr__(self):
        return f'DeepChromatographyApex(name={self.name}, backend={self.backend})'


def predict_retention_time_with_koina(
    model_name: str,
    data: pd.DataFrame,
    seq_col: str = "sequence",
    gradient_length: Optional[float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Predict retention times using Koina.

    Args:
        model_name: Name of the model.
        data: DataFrame with peptide sequences.
        seq_col: Column name for sequences.
        gradient_length: Gradient length for scaling.
        verbose: Verbosity.

    Returns:
        DataFrame with predicted retention times.
    """
    from imspy_predictors.koina_models import ModelFromKoina

    rt_model = ModelFromKoina(model_name=model_name)
    inputs = data.copy()
    inputs.rename(columns={seq_col: "peptide_sequences"}, inplace=True)
    rts = rt_model.predict(inputs[["peptide_sequences"]])

    if verbose:
        print(f"Koina model {model_name} predicted RT for {len(rts)} peptides.")

    if gradient_length is not None:
        mapped_rt = linear_map(
            rts.iloc[:, 1].values,
            old_min=rts.iloc[:, 1].min(),
            old_max=rts.iloc[:, 1].max(),
            new_min=0,
            new_max=gradient_length,
        )
        data["retention_time_gru_predictor"] = mapped_rt
    else:
        data["retention_time_gru_predictor"] = rts.iloc[:, 1].values

    return data
