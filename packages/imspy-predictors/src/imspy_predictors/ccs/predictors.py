"""
CCS (Collision Cross Section) and Ion Mobility Predictors.

This module provides deep learning models for predicting collision cross sections
and ion mobilities for peptide ions using PyTorch.

Classes:
    - DeepPeptideIonMobilityApex: High-level wrapper for CCS/ion mobility prediction
    - PyTorchCCSPredictor: PyTorch transformer-based model
"""

from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit

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
from imspy_core.chemistry import ccs_to_one_over_k0, one_over_k0_to_ccs, calculate_mz
from imspy_core.utility import tokenize_unimod_sequence


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
        from imspy_search.utility import generate_balanced_im_dataset
        return generate_balanced_im_dataset
    except ImportError:
        raise ImportError(
            "generate_balanced_im_dataset requires imspy-search package."
        )


def predict_inverse_ion_mobility(
    psm_collection: List,
    refine_model: bool = True,
    verbose: bool = False,
) -> None:
    """
    Predict inverse ion mobility for a collection of peptide spectrum matches.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: A list of peptide spectrum matches (sagepy Psm objects).
        refine_model: Whether to refine the model by fine-tuning it on the provided data.
        verbose: Whether to print additional information during the prediction.

    Returns:
        None, the inverse ion mobility is set in the peptide spectrum matches in place.
    """
    Psm, psm_collection_to_pandas = _get_sagepy_utils()
    generate_balanced_im_dataset = _get_dbsearch_utils()

    im_predictor = DeepPeptideIonMobilityApex(verbose=verbose)

    if refine_model:
        im_predictor.fine_tune_model(
            psm_collection_to_pandas(generate_balanced_im_dataset(psm_collection)),
            batch_size=128,
            verbose=verbose,
        )

    # predict ion mobilities
    inv_mob = im_predictor.simulate_ion_mobilities(
        sequences=[
            x.sequence_modified if not x.decoy else x.sequence_decoy_modified
            for x in psm_collection
        ],
        charges=[x.charge for x in psm_collection],
        mz=[x.mono_mz_calculated for x in psm_collection],
    )

    # set ion mobilities
    for mob, ps in zip(inv_mob, psm_collection):
        ps.inverse_ion_mobility_predicted = mob


def get_sqrt_slopes_and_intercepts(
    mz: NDArray,
    charge: NDArray,
    ccs: NDArray,
    fit_charge_state_one: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Fit the square root model to the provided data.

    Args:
        mz: The m/z values.
        charge: The charge states.
        ccs: The collision cross sections.
        fit_charge_state_one: Whether to fit charge state one.

    Returns:
        The slopes and intercepts of the square root model fit.
    """
    if fit_charge_state_one:
        slopes, intercepts = [], []
    else:
        slopes, intercepts = [0.0], [0.0]

    c_begin = 1 if fit_charge_state_one else 2

    for c in range(c_begin, 5):
        def fit_func(x, a, b):
            return a * np.sqrt(x) + b

        triples = list(filter(lambda x: x[1] == c, zip(mz, charge, ccs)))
        mz_tmp = np.array([x[0] for x in triples])
        ccs_tmp = np.array([x[2] for x in triples])

        popt, _ = curve_fit(fit_func, mz_tmp, ccs_tmp)
        slopes.append(popt[0])
        intercepts.append(popt[1])

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)


# Default physics-based initialization for CCS prediction
DEFAULT_CCS_SLOPES = np.array([0.0, 11.72, 15.06, 18.95, 21.87], dtype=np.float32)
DEFAULT_CCS_INTERCEPTS = np.array([0.0, 134.79, 60.49, -18.30, -86.35], dtype=np.float32)


class PeptideIonMobilityApex(ABC):
    """Abstract base class for ion mobility prediction."""

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ion_mobilities(
        self, sequences: List[str], charges: List[int], mz: List[float]
    ) -> NDArray:
        pass

    @abstractmethod
    def simulate_ion_mobilities_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


# =============================================================================
# PyTorch Implementation
# =============================================================================

if TORCH_AVAILABLE:

    class SquareRootProjectionLayerPyTorch(nn.Module):
        """
        Physics-based CCS initialization layer using square root of m/z.

        Projects m/z to initial CCS estimate using the relation:
        CCS ≈ a * sqrt(m/z) + b, weighted by charge state.
        """

        def __init__(
            self,
            slopes: Union[np.ndarray, torch.Tensor] = DEFAULT_CCS_SLOPES,
            intercepts: Union[np.ndarray, torch.Tensor] = DEFAULT_CCS_INTERCEPTS,
            trainable: bool = True,
        ):
            super().__init__()

            if isinstance(slopes, np.ndarray):
                slopes = torch.from_numpy(slopes)
            if isinstance(intercepts, np.ndarray):
                intercepts = torch.from_numpy(intercepts)

            if trainable:
                self.slopes = nn.Parameter(slopes.float())
                self.intercepts = nn.Parameter(intercepts.float())
            else:
                self.register_buffer('slopes', slopes.float())
                self.register_buffer('intercepts', intercepts.float())

        def forward(
            self, mz: torch.Tensor, charge_onehot: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                mz: m/z values of shape (batch,) or (batch, 1)
                charge_onehot: One-hot charge of shape (batch, num_charges)

            Returns:
                Initial CCS estimate of shape (batch, 1)
            """
            if mz.dim() == 1:
                mz = mz.unsqueeze(1)

            # sqrt(m/z) * slopes + intercepts, weighted by charge
            sqrt_mz = torch.sqrt(mz)  # (batch, 1)
            projection = (self.slopes * sqrt_mz + self.intercepts) * charge_onehot
            return projection.sum(dim=-1, keepdim=True)

    class PyTorchCCSPredictor(nn.Module):
        """
        PyTorch CCS predictor using GRU or Transformer encoder.

        Architecture options:
        - 'gru': Bidirectional GRU (similar to legacy TensorFlow model)
        - 'transformer': Transformer encoder (modern architecture)
        """

        def __init__(
            self,
            vocab_size: int,
            slopes: np.ndarray = DEFAULT_CCS_SLOPES,
            intercepts: np.ndarray = DEFAULT_CCS_INTERCEPTS,
            max_seq_len: int = 50,
            emb_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            max_charge: int = 4,
            architecture: str = 'gru',
            predict_std: bool = False,
        ):
            super().__init__()

            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len
            self.max_charge = max_charge
            self.architecture = architecture
            self.predict_std = predict_std

            # Physics-based initialization
            self.sqrt_proj = SquareRootProjectionLayerPyTorch(slopes, intercepts)

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
            self.fc1 = nn.Linear(encoder_out_dim + max_charge, 128)
            self.fc2 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(dropout)

            # Output: mean and optionally std
            output_dim = 2 if predict_std else 1
            self.out = nn.Linear(64, output_dim)

        def forward(
            self,
            mz: torch.Tensor,
            charge_onehot: torch.Tensor,
            sequences: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                mz: m/z values (batch,) or (batch, 1)
                charge_onehot: One-hot charge (batch, max_charge)
                sequences: Token IDs (batch, seq_len)
                padding_mask: Boolean mask for padding (batch, seq_len)

            Returns:
                Tuple of (total_ccs, residual_ccs)
                If predict_std=True, total_ccs is (mean, std)
            """
            # Initial physics-based estimate
            initial_ccs = self.sqrt_proj(mz, charge_onehot)

            # Sequence encoding
            x = self.embedding(sequences)

            if self.architecture == 'gru':
                # GRU encoding - take final hidden state
                _, hidden = self.encoder(x)
                # Concatenate forward and backward final states
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                # Transformer encoding - use mean pooling
                if padding_mask is not None:
                    # Expand mask for broadcasting
                    mask_expanded = padding_mask.unsqueeze(-1).float()
                    x = self.encoder(x, src_key_padding_mask=padding_mask)
                    x = (x * (1 - mask_expanded)).sum(dim=1) / (1 - mask_expanded).sum(dim=1)
                else:
                    x = self.encoder(x)
                    x = x.mean(dim=1)
                hidden = x

            # Combine with charge information
            concat = torch.cat([charge_onehot, hidden], dim=-1)

            # Dense layers
            x = self.dropout(F.relu(self.fc1(concat)))
            x = F.relu(self.fc2(x))

            # Output
            residual = self.out(x)

            if self.predict_std:
                mean_residual = residual[:, :1]
                log_std = residual[:, 1:]
                std = F.softplus(log_std)
                total_ccs = initial_ccs + mean_residual
                return (total_ccs, std), mean_residual
            else:
                total_ccs = initial_ccs + residual
                return total_ccs, residual

# Backwards compatibility alias
SquareRootProjectionLayer = SquareRootProjectionLayerPyTorch if TORCH_AVAILABLE else None


def load_deep_ccs_predictor(backend: Optional[str] = None):
    """
    Load a pretrained CCS predictor model.

    Args:
        backend: Ignored (kept for backward compatibility). Always uses PyTorch.

    Returns:
        Loaded PyTorch model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for CCS prediction. Install with: pip install torch")

    # Try to load UnifiedPeptideModel first (new architecture)
    try:
        from imspy_predictors.models import UnifiedPeptideModel
        model_path = get_model_path('ccs/best_model.pt')
        if model_path.exists():
            model = UnifiedPeptideModel.from_pretrained(str(model_path), tasks=['ccs'])
            return model
    except (ImportError, FileNotFoundError):
        pass

    # Fall back to legacy PyTorchCCSPredictor
    model_path = get_model_path('ccs/ccs-model.pt')
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer
    tokenizer = ProformaTokenizer.with_defaults()
    vocab_size = tokenizer.vocab_size

    model = PyTorchCCSPredictor(vocab_size=vocab_size)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


class DeepPeptideIonMobilityApex(PeptideIonMobilityApex):
    """
    High-level wrapper for CCS and ion mobility prediction using PyTorch.

    Args:
        model: Pre-loaded model (optional, will load default if None)
        tokenizer: Tokenizer for sequences (optional, will load default if None)
        verbose: Whether to print progress
        name: Model name for output columns

    Example:
        >>> predictor = DeepPeptideIonMobilityApex()
        >>> inv_mob = predictor.simulate_ion_mobilities(
        ...     sequences=["PEPTIDE", "SEQUENCE"],
        ...     charges=[2, 3],
        ...     mz=[400.0, 350.0]
        ... )
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        verbose: bool = False,
        name: str = 'gru_predictor',
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepPeptideIonMobilityApex. Install with: pip install torch")

        super().__init__()

        # Load tokenizer
        if tokenizer is None:
            from imspy_predictors.utilities.tokenizers import ProformaTokenizer
            self.tokenizer = ProformaTokenizer.with_defaults()
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            self.model = load_deep_ccs_predictor()
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

        # Pad to fixed length
        if tokens.shape[1] < pad_len:
            padding = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
            tokens = torch.cat([tokens, padding], dim=1)
        elif tokens.shape[1] > pad_len:
            tokens = tokens[:, :pad_len]

        return tokens.to(self._device)

    def simulate_ion_mobilities(
        self,
        sequences: List[str],
        charges: List[int],
        mz: List[float],
        batch_size: int = 1024,
        return_uncertainty: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict inverse ion mobilities for peptide sequences.

        Args:
            sequences: List of peptide sequences
            charges: List of charge states
            mz: List of m/z values
            batch_size: Batch size for prediction
            return_uncertainty: If True, also return predicted uncertainty (std)

        Returns:
            Inverse ion mobilities (1/K0), or tuple of (1/K0, std) if return_uncertainty=True
        """
        self.model.eval()

        # Prepare data
        tokens = self._preprocess_sequences(sequences)
        mz_tensor = torch.tensor(mz, dtype=torch.float32, device=self._device)
        charges_onehot = F.one_hot(
            torch.tensor(charges, device=self._device) - 1, num_classes=4
        ).float()

        # Predict in batches
        all_ccs = []
        all_ccs_std = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                batch_mz = mz_tensor[i:i + batch_size]
                batch_charges = charges_onehot[i:i + batch_size]

                # Handle different model types
                if hasattr(self.model, 'predict_ccs'):
                    # UnifiedPeptideModel
                    ccs, ccs_std = self.model.predict_ccs(
                        batch_tokens,
                        batch_mz,
                        torch.argmax(batch_charges, dim=1) + 1,
                    )
                else:
                    # Legacy PyTorchCCSPredictor
                    result = self.model(batch_mz, batch_charges, batch_tokens)
                    if isinstance(result, tuple):
                        ccs, ccs_std = result[0], result[1] if len(result) > 1 else None
                    else:
                        ccs, ccs_std = result, None

                all_ccs.append(ccs.cpu().numpy())
                if ccs_std is not None:
                    all_ccs_std.append(ccs_std.cpu().numpy())

        ccs = np.concatenate(all_ccs, axis=0).flatten()

        # Convert CCS to inverse mobility
        inverse_mobility = np.array([
            ccs_to_one_over_k0(c, m, z)
            for c, m, z in zip(ccs, mz, charges)
        ])

        if return_uncertainty and all_ccs_std:
            ccs_std = np.concatenate(all_ccs_std, axis=0).flatten()
            inverse_mobility_std = np.array([
                ccs_to_one_over_k0(s, m, z)
                for s, m, z in zip(ccs_std, mz, charges)
            ])
            return inverse_mobility, inverse_mobility_std

        return inverse_mobility

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
            data: DataFrame with columns: sequence, charge, calcmass, ims
            batch_size: Training batch size
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to print progress
            decoys_separate: Whether to handle decoys separately
        """
        from torch.utils.data import DataLoader, TensorDataset

        assert 'sequence' in data.columns, 'Data must contain column "sequence"'
        assert 'charge' in data.columns, 'Data must contain column "charge"'
        assert 'calcmass' in data.columns, 'Data must contain column "calcmass"'
        assert 'ims' in data.columns, 'Data must contain column "ims"'

        mz = [calculate_mz(m, z) for m, z in zip(data.calcmass.values, data.charge.values.astype(np.int32))]
        charges = data.charge.values.astype(np.int32)

        if decoys_separate:
            sequences = []
            for _, row in data.iterrows():
                if not row.decoy:
                    sequences.append(row.sequence_modified)
                else:
                    sequences.append(row.sequence_decoy_modified)
        else:
            sequences = list(data.sequence_modified.values)

        inv_mob = data.ims.values
        ccs = np.array([
            one_over_k0_to_ccs(i, m, z)
            for i, m, z in zip(inv_mob, mz, charges)
        ])

        # Prepare data
        tokens = self._preprocess_sequences(sequences)
        mz_tensor = torch.tensor(mz, dtype=torch.float32, device=self._device)
        charges_onehot = F.one_hot(
            torch.tensor(charges, device=self._device) - 1, num_classes=4
        ).float()
        ccs_tensor = torch.tensor(ccs, dtype=torch.float32, device=self._device).unsqueeze(1)

        # Split data
        n = len(sequences)
        n_train = int(0.8 * n)
        indices = torch.randperm(n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        # Create datasets
        train_dataset = TensorDataset(
            mz_tensor[train_idx],
            charges_onehot[train_idx],
            tokens[train_idx],
            ccs_tensor[train_idx],
        )
        val_dataset = TensorDataset(
            mz_tensor[val_idx],
            charges_onehot[val_idx],
            tokens[val_idx],
            ccs_tensor[val_idx],
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training setup
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, min_lr=1e-6
        )
        checkpoint = InMemoryCheckpoint(patience=patience)

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                mz_b, charge_b, tokens_b, ccs_b = batch
                optimizer.zero_grad()

                # Handle different model types
                if hasattr(self.model, 'predict_ccs'):
                    pred, _ = self.model.predict_ccs(
                        tokens_b,
                        mz_b,
                        torch.argmax(charge_b, dim=1) + 1,
                    )
                else:
                    pred, _ = self.model(mz_b, charge_b, tokens_b)
                    if isinstance(pred, tuple):
                        pred = pred[0]

                loss = F.l1_loss(pred, ccs_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    mz_b, charge_b, tokens_b, ccs_b = batch

                    if hasattr(self.model, 'predict_ccs'):
                        pred, _ = self.model.predict_ccs(
                            tokens_b,
                            mz_b,
                            torch.argmax(charge_b, dim=1) + 1,
                        )
                    else:
                        pred, _ = self.model(mz_b, charge_b, tokens_b)
                        if isinstance(pred, tuple):
                            pred = pred[0]

                    val_loss += F.l1_loss(pred, ccs_b).item()

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

    def simulate_ion_mobilities_pandas(
        self,
        data: pd.DataFrame,
        batch_size: int = 1024,
        return_ccs: bool = False,
        decoys_separate: bool = True,
    ) -> pd.DataFrame:
        """
        Predict ion mobilities for a DataFrame.

        Args:
            data: DataFrame with sequence, mz, charge columns
            batch_size: Batch size for prediction
            return_ccs: If True, return CCS instead of inverse mobility
            decoys_separate: Handle decoys separately

        Returns:
            DataFrame with predicted mobilities/CCS
        """
        assert 'sequence' in data.columns, 'Data must contain column "sequence"'

        if decoys_separate:
            sequences = []
            for _, row in data.iterrows():
                if not row.decoy:
                    sequences.append(getattr(row, 'sequence_modified', row.sequence))
                else:
                    sequences.append(getattr(row, 'sequence_decoy_modified', row.sequence))
        else:
            sequences = list(data.sequence_modified.values)

        mz = data.mz.values.tolist()
        charges = data.charge.values.astype(np.int32).tolist()

        inv_mob = self.simulate_ion_mobilities(sequences, charges, mz, batch_size)

        if not return_ccs:
            data[f'inv_mobility_{self.name}'] = inv_mob
        else:
            # Convert back to CCS
            ccs = [
                one_over_k0_to_ccs(i, m, z)
                for i, m, z in zip(inv_mob, mz, charges)
            ]
            data[f'ccs_{self.name}'] = ccs

        return data

    def to(self, device: str) -> "DeepPeptideIonMobilityApex":
        """Move model to device."""
        self._device = device
        self.model = self.model.to(device)
        return self

    def __repr__(self):
        return f'DeepPeptideIonMobilityApex(name={self.name})'


def predict_inverse_ion_mobility_with_koina(
    model_name: str,
    data: pd.DataFrame,
    seq_col: str = "sequence",
    charge_col: str = "charge",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Predict ion mobility using Koina model.

    Args:
        model_name: Name of the Koina model.
        data: Input data for the model.
        seq_col: Column name for peptide sequences.
        charge_col: Column name for precursor charges.
        verbose: Verbosity.

    Returns:
        DataFrame with predicted CCS, mz, and inverse mobility.
    """
    assert 'calcmass' in data.columns, 'Data must contain column "calcmass"'

    from imspy_predictors.koina_models import ModelFromKoina

    ccs_model = ModelFromKoina(model_name=model_name)
    inputs = data.copy()
    inputs.rename(
        columns={seq_col: "peptide_sequences", charge_col: "precursor_charges"},
        inplace=True,
    )
    ccs = ccs_model.predict(inputs)

    if verbose:
        print(f"Koina model {model_name} predicted ccs for {len(ccs)} peptides.")

    data[f'ccs_{model_name}'] = ccs['ccs']
    data['mz'] = data['calcmass'] / data[charge_col]
    data[f'inv_mobility_{model_name}'] = [
        ccs_to_one_over_k0(c, m, z)
        for c, m, z in zip(data[f'ccs_{model_name}'].values, data['mz'], data[charge_col])
    ]

    return data
