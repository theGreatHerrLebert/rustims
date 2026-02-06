"""
Data utilities for training peptide property prediction models.

Provides PyTorch Dataset and DataLoader utilities for working with
peptide sequences and their associated properties (CCS, RT, charge, intensity).

Includes support for instrument type encoding to enable instrument-specific
predictions.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from imspy_predictors.models.heads import get_instrument_id


class PeptideDataset(Dataset):
    """
    PyTorch Dataset for peptide property prediction.

    Handles tokenization and encoding of peptide sequences along with
    associated properties like m/z, charge, CCS, RT, instrument type, etc.

    Args:
        sequences: List of peptide sequences in UNIMOD format
        tokenizer: Tokenizer instance for encoding sequences
        mz: Optional m/z values
        charge: Optional charge states
        collision_energy: Optional collision energies (for intensity prediction)
        instrument: Optional instrument types (names or IDs)
        ccs: Optional CCS values (target)
        rt: Optional retention time values (target)
        intensity: Optional fragment intensities (target)
        max_length: Maximum sequence length (will pad/truncate)

    Example:
        >>> from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        >>> tokenizer = ProformaTokenizer.with_defaults()
        >>> dataset = PeptideDataset(
        ...     sequences=["PEPTIDE", "MC[UNIMOD:4]PEPTIDE"],
        ...     tokenizer=tokenizer,
        ...     mz=[500.0, 600.0],
        ...     charge=[2, 3],
        ...     ccs=[350.0, 420.0],
        ...     instrument=["timstof", "timstof"],
        ... )
        >>> batch = dataset[0]
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer,
        mz: Optional[List[float]] = None,
        charge: Optional[List[int]] = None,
        collision_energy: Optional[List[float]] = None,
        instrument: Optional[Union[List[str], List[int]]] = None,
        ccs: Optional[List[float]] = None,
        rt: Optional[List[float]] = None,
        intensity: Optional[List[np.ndarray]] = None,
        charge_dist: Optional[List[np.ndarray]] = None,
        max_length: int = 100,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Features
        self.mz = mz
        self.charge = charge
        self.collision_energy = collision_energy

        # Convert instrument names to IDs if needed
        if instrument is not None and len(instrument) > 0:
            if isinstance(instrument[0], str):
                self.instrument = [get_instrument_id(inst) for inst in instrument]
            else:
                self.instrument = instrument
        else:
            self.instrument = None

        # Targets
        self.ccs = ccs
        self.rt = rt
        self.intensity = intensity
        self.charge_dist = charge_dist

        # Pre-tokenize and encode sequences for efficiency
        self._tokenize_all()

    def _tokenize_all(self):
        """Tokenize and encode all sequences."""
        tokens = self.tokenizer.tokenize_batch(self.sequences)

        # Truncate and pad
        encoded_list = []
        mask_list = []

        for seq_tokens in tokens:
            # Truncate if needed
            if len(seq_tokens) > self.max_length:
                seq_tokens = seq_tokens[: self.max_length - 1] + ["[SEP]"]

            # Encode
            encoded = self.tokenizer.encode(seq_tokens)

            # Pad if needed
            pad_len = self.max_length - len(encoded)
            mask = [1] * len(encoded) + [0] * pad_len
            encoded = encoded + [self.tokenizer.pad_token_id] * pad_len

            encoded_list.append(encoded)
            mask_list.append(mask)

        self.token_ids = torch.tensor(encoded_list, dtype=torch.long)
        self.attention_mask = torch.tensor(mask_list, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - mz, charge, collision_energy, instrument: Features (if provided)
                - ccs, rt, intensity, charge_dist: Targets (if provided)
        """
        item = {
            "input_ids": self.token_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

        # Add features
        if self.mz is not None:
            item["mz"] = torch.tensor(self.mz[idx], dtype=torch.float32)
        if self.charge is not None:
            item["charge"] = torch.tensor(self.charge[idx], dtype=torch.long)
        if self.collision_energy is not None:
            item["collision_energy"] = torch.tensor(
                self.collision_energy[idx], dtype=torch.float32
            )
        if self.instrument is not None:
            item["instrument"] = torch.tensor(self.instrument[idx], dtype=torch.long)

        # Add targets
        if self.ccs is not None:
            item["ccs"] = torch.tensor(self.ccs[idx], dtype=torch.float32)
        if self.rt is not None:
            item["rt"] = torch.tensor(self.rt[idx], dtype=torch.float32)
        if self.intensity is not None:
            item["intensity"] = torch.tensor(self.intensity[idx], dtype=torch.float32)
        if self.charge_dist is not None:
            item["charge_dist"] = torch.tensor(self.charge_dist[idx], dtype=torch.float32)

        return item


class HuggingFaceDatasetWrapper(Dataset):
    """
    Wrapper for Hugging Face datasets to work with our tokenizer.

    Lazily tokenizes sequences on access for memory efficiency
    with large datasets.

    Args:
        hf_dataset: Hugging Face dataset
        tokenizer: ProformaTokenizer instance
        sequence_column: Name of the column containing sequences
        max_length: Maximum sequence length
        column_mapping: Optional dict mapping our column names to dataset columns
        default_instrument: Default instrument type for all samples
                           (string name like "timstof" or integer ID)
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        sequence_column: str = "modified_sequence",
        max_length: int = 100,
        column_mapping: Optional[Dict[str, str]] = None,
        default_instrument: Optional[Union[str, int]] = None,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.max_length = max_length
        self.column_mapping = column_mapping or {}

        # Convert instrument name to ID if needed
        if default_instrument is not None:
            if isinstance(default_instrument, str):
                self.default_instrument = get_instrument_id(default_instrument)
            else:
                self.default_instrument = default_instrument
        else:
            self.default_instrument = 0  # "unknown"

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_column(self, item: dict, name: str):
        """Get column value with optional mapping."""
        mapped_name = self.column_mapping.get(name, name)
        return item.get(mapped_name)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get and tokenize a single sample."""
        item = self.dataset[idx]

        # Tokenize sequence
        sequence = item[self.sequence_column]
        tokens = self.tokenizer.tokenize(sequence)

        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + ["[SEP]"]

        # Encode
        encoded = self.tokenizer.encode(tokens)

        # Pad if needed
        pad_len = self.max_length - len(encoded)
        mask = [1] * len(encoded) + [0] * pad_len
        encoded = encoded + [self.tokenizer.pad_token_id] * pad_len

        result = {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

        # Add optional fields
        if "mz" in item or self._get_column(item, "mz") is not None:
            result["mz"] = torch.tensor(
                self._get_column(item, "mz") or item.get("mz"), dtype=torch.float32
            )

        if "charge" in item or self._get_column(item, "charge") is not None:
            result["charge"] = torch.tensor(
                self._get_column(item, "charge") or item.get("charge"), dtype=torch.long
            )
        elif "precursor_charge_onehot" in item:
            # Convert one-hot to integer (1-indexed)
            charge_onehot = item["precursor_charge_onehot"]
            charge_int = charge_onehot.index(1) + 1 if 1 in charge_onehot else 2
            result["charge"] = torch.tensor(charge_int, dtype=torch.long)

        if "collision_energy" in item or "collision_energy_aligned_normed" in item:
            ce = item.get("collision_energy_aligned_normed", item.get("collision_energy"))
            if ce is not None:
                result["collision_energy"] = torch.tensor(ce, dtype=torch.float32)

        # Add instrument type (from data or default)
        if "instrument" in item:
            inst = item["instrument"]
            if isinstance(inst, str):
                inst = get_instrument_id(inst)
            result["instrument"] = torch.tensor(inst, dtype=torch.long)
        else:
            result["instrument"] = torch.tensor(self.default_instrument, dtype=torch.long)

        # Targets
        if "ccs" in item:
            result["ccs"] = torch.tensor(item["ccs"], dtype=torch.float32)

        # CCS standard deviation (for supervised uncertainty training)
        # -1 indicates missing values that should be masked during training
        if "ccs_std" in item:
            result["ccs_std"] = torch.tensor(item["ccs_std"], dtype=torch.float32)

        if "indexed_retention_time" in item:
            result["rt"] = torch.tensor(item["indexed_retention_time"], dtype=torch.float32)
        elif "retention_time" in item:
            result["rt"] = torch.tensor(item["retention_time"], dtype=torch.float32)

        if "intensities_raw" in item:
            result["intensity"] = torch.tensor(item["intensities_raw"], dtype=torch.float32)

        if "charge_state_dist" in item:
            result["charge_dist"] = torch.tensor(item["charge_state_dist"], dtype=torch.float32)
        elif "precursor_charge_onehot" in item:
            result["charge_dist"] = torch.tensor(item["precursor_charge_onehot"], dtype=torch.float32)

        return result


def collate_peptide_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for peptide datasets.

    Stacks tensors from individual samples into batched tensors.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    result = {}

    # Get all keys from the first sample
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch if key in sample]
        if values:
            result[key] = torch.stack(values)

    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop the last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_peptide_batch,
    )


def load_ionmob_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
    streaming: bool = False,
    default_instrument: str = "timstof",
):
    """
    Load the ionmob CCS dataset from Hugging Face.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split ("train", "validation", "test")
        max_length: Maximum sequence length
        streaming: Whether to use streaming mode (for large datasets)
        default_instrument: Default instrument type (default: "timstof")

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Map split names (ionmob uses 'val' instead of 'validation')
    split_name = "val" if split == "validation" else split

    dataset = load_dataset(
        "theGreatHerrLebert/ionmob",
        split=split_name,
        streaming=streaming,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="sequence_modified",
        max_length=max_length,
        column_mapping={
            "charge": "charge",
            "mz": "mz",
        },
        default_instrument=default_instrument,
    )


def load_prospect_rt_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
):
    """
    Load the PROSPECT RT dataset from Hugging Face.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split
        max_length: Maximum sequence length

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Map split names (dataset uses 'val' instead of 'validation')
    split_name = "val" if split == "validation" else split

    dataset = load_dataset(
        "Wilhelmlab/prospect-ptms-irt",
        split=split_name,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="modified_sequence",
        max_length=max_length,
    )


def load_prospect_charge_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
):
    """
    Load the PROSPECT charge dataset from Hugging Face.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split
        max_length: Maximum sequence length

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Map split names (dataset uses 'val' instead of 'validation')
    split_name = "val" if split == "validation" else split

    dataset = load_dataset(
        "Wilhelmlab/prospect-ptms-charge",
        split=split_name,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="modified_sequence",
        max_length=max_length,
    )


def load_prospect_ms2_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
    streaming: bool = True,
):
    """
    Load the PROSPECT MS2 intensity dataset from Hugging Face.

    Note: This is a large dataset (31M samples). Consider using streaming=True.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split
        max_length: Maximum sequence length
        streaming: Whether to use streaming mode (recommended for this large dataset)

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    dataset = load_dataset(
        "Wilhelmlab/prospect-ptms-ms2",
        split=split,
        streaming=streaming,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="modified_sequence",
        max_length=max_length,
    )


def load_timstof_ms2_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
):
    """
    Load the timsTOF MS2 intensity dataset from Hugging Face.

    This is the fine-tuning dataset for timsTOF-specific intensity prediction.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split
        max_length: Maximum sequence length

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Map split names (dataset uses 'val' instead of 'validation')
    split_name = "val" if split == "validation" else split

    dataset = load_dataset(
        "Wilhelmlab/timsTOF-ms2",
        split=split_name,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="modified_sequence",
        max_length=max_length,
        default_instrument="timstof",  # This dataset is specifically timsTOF
    )


def load_orbitrap_ms2_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 100,
    instrument: str = "orbitrap",
):
    """
    Load an Orbitrap MS2 intensity dataset from Hugging Face.

    Args:
        tokenizer: ProformaTokenizer instance
        split: Dataset split
        max_length: Maximum sequence length
        instrument: Specific Orbitrap variant (default: "orbitrap")
                   Options: "orbitrap", "orbitrap_fusion", "orbitrap_eclipse", "exploris", "astral"

    Returns:
        HuggingFaceDatasetWrapper instance
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # The PROSPECT MS2 dataset contains mostly Orbitrap data
    dataset = load_dataset(
        "Wilhelmlab/prospect-ptms-ms2",
        split=split,
    )

    return HuggingFaceDatasetWrapper(
        dataset,
        tokenizer,
        sequence_column="modified_sequence",
        max_length=max_length,
        default_instrument=instrument,
    )
