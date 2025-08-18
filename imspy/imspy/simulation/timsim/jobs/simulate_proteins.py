from typing import Union

import numpy as np
import pandas as pd

from collections import Counter

from sagepy.core.database import PeptideIx
from sagepy.core import EnzymeBuilder, SageSearchConfiguration


def parse_fasta_to_dataframe(file_path):
    """
    Parses a FASTA file and returns a pandas DataFrame with columns 'Name' and 'Sequence'.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        pandas.DataFrame: A DataFrame with two columns: 'Name' and 'Sequence'.
    """
    proteins = []
    with open(file_path, 'r') as fasta_file:
        current_name = None
        current_sequence = []

        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):  # Indicates the start of a new protein
                if current_name:  # Save the previous protein if one exists
                    proteins.append({"protein": current_name, "sequence": ''.join(current_sequence)})
                current_name = line[1:]  # Exclude the '>' character
                current_sequence = []  # Reset sequence for the new protein
            else:
                current_sequence.append(line)  # Add the line to the current sequence

        # Add the last protein to the list
        if current_name:
            proteins.append({"protein": current_name, "sequence": ''.join(current_sequence)})

    # Convert to pandas DataFrame
    return pd.DataFrame(proteins)


def generate_single_fasta(name, sequence):
    """
    Generates a FASTA-formatted string for a single protein.

    Args:
        name (str): The name of the protein.
        sequence (str): The sequence of the protein.

    Returns:
        str: FASTA-formatted string.
    """
    # Create the name line with '>'
    fasta_lines = [f">{name}"]
    # Split the sequence into lines of 60 characters (standard FASTA format)
    fasta_lines.extend(sequence[i:i + 60] for i in range(0, len(sequence), 60))
    return "\n".join(fasta_lines)


def protein_to_peptides(fasta,
                        missed_cleavages=2,
                        min_len=7,
                        max_len=30,
                        cleave_at='KR',
                        restrict='P',
                        generate_decoys=False,
                        c_terminal=True,
                        variable_mods={},
                        static_mods={"C": "[UNIMOD:4]"},
                        digest: bool = True,
                        ) -> Union[set, None]:
    """
    Generates a set of unique peptides from a protein sequence using digestion logic, including PTMs.

    Args:
        sequence (str): The input protein sequence.
        missed_cleavages (int): Number of allowed missed cleavages.
        min_len (int): Minimum length of peptides.
        max_len (int): Maximum length of peptides.
        cleave_at (str): Residues at which cleavage occurs.
        restrict (str): Residues after which cleavage is restricted.
        generate_decoys (bool): Whether to generate decoy peptides.
        c_terminal (bool): Whether cleavage occurs at the C-terminal.
        verbose (bool): Whether to display progress.
        variable_mods (dict): Variable modifications to apply.
        static_mods (dict): Static modifications to apply.

    Returns:
        set: A set of unique peptide sequences with PTMs.
    """

    # if digest is set to False, we will take every protein and put it into a set
    if not digest:
        # Remove '>' from the FASTA header and return the sequence as a set
        return {fasta.split('\n', 1)[1].replace('\n', '')}

    # Simulate enzyme builder and digestion logic
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        cleave_at=cleave_at,
        restrict=restrict,
        c_terminal=c_terminal,
    )

    # Simulate SageSearch configuration
    sage_config = SageSearchConfiguration(
        fasta=fasta,  # Wrap sequence as FASTA format
        enzyme_builder=enzyme_builder,
        static_mods=static_mods,
        variable_mods=variable_mods,
        generate_decoys=generate_decoys,
        bucket_size=int(np.power(2, 6))
    )

    try:
        indexed_db = sage_config.generate_indexed_database()

        peptide_set = set()

        # Generate peptides using SageSearch's indexed database
        for i in range(indexed_db.num_peptides):
            idx = PeptideIx(idx=i)
            peptide = indexed_db[idx]
            peptide_set.add(peptide.to_unimod_sequence())

        return peptide_set

    except Exception as e:
        # print(f"Error generating peptides: {e}")
        # print("This error can usually be safely ignored when using TIMSIM ...")
        return None


def get_tenzer_hokey():
    # Tenzer constants
    delta_exp = 0.005
    delta = 0.0011
    exp_red = 600
    target_start = 1e4

    rank = np.linspace(1, 1e4, int(1e4))

    exp_values = 1 / np.exp(rank / exp_red)

    cumulative_exp_sum = np.cumsum(exp_values)

    targets_closed_form = target_start * (2 ** (-delta * rank)) * (2 ** (-delta_exp * cumulative_exp_sum))

    return targets_closed_form


def assign_events(df, upscale_factor=int(1e5)):
    num_samples = len(df)
    indices = np.random.uniform(1, 1e4, num_samples).astype(np.int32)
    num_events = upscale_factor * get_tenzer_hokey()[indices]
    df["events"] = num_events
    df["total_events"] = df["events"]
    return df


def simulate_proteins(
        fasta_file_path: str,
        n_proteins: int = 20_000,
        upscale_factor: int = 1e5,
        cleave_at: str = 'KR',
        restrict: str = 'P',
        missed_cleavages: int = 2,
        min_len: int = 7,
        max_len: int = 30,
        generate_decoys: bool = False,
        variable_mods: dict = {},
        static_mods: dict = {"C": "[UNIMOD:4]"},
        verbose: bool = True,
        sample_proteins: bool = True,
        digest: bool = True,
        remove_degenerate_peptides: bool = False,
) -> pd.DataFrame:
    """
    Simulate proteins.

    Args:
        fasta_file_path (str): Path to the FASTA file.
        n_proteins (int): Number of proteins to sample.
        upscale_factor (int): Upscale factor.
        variable_mods (dict): Variable modifications.
        static_mods (dict): Static modifications.
        cleave_at (str): Cleavage sites.
        restrict (str): Restrict to specific proteins.
        missed_cleavages (int): Number of missed cleavages.
        min_len (int): Minimum peptide length.
        max_len (int): Maximum peptide length.
        generate_decoys (bool): Generate decoys.
        verbose (bool): Verbosity.
        sample_proteins (bool): Whether to sample proteins from the FASTA file.
        digest (bool): Whether to digest proteins into peptides.
        remove_degenerate_peptides (bool): Whether to remove peptides that map to multiple proteins.

    Returns:
        pd.DataFrame: Proteins DataFrame.
    """

    tbl = parse_fasta_to_dataframe(
        fasta_file_path,
    )

    # Sample proteins
    if n_proteins > len(tbl):
        n_proteins = len(tbl)

        if verbose:
            print(f"Number of proteins requested exceeds the number of proteins in the FASTA file. Using {n_proteins} available proteins.")

    else:
        if sample_proteins:
            if verbose:
                print(f"Sampling {n_proteins} proteins from the FASTA file.")

    if sample_proteins:
        sample = tbl.sample(n=n_proteins)
    else:
        print("Sampling all proteins from the FASTA file.")
        sample = tbl

    if verbose and not digest:
        print("Skipping digestion, returning raw protein sequences as peptides.")

    # Generate peptides
    sample["peptides"] = sample.apply(lambda f: protein_to_peptides(
        generate_single_fasta(
            f.name,
            f.sequence,
        ),
        generate_decoys=generate_decoys,
        variable_mods=variable_mods,
        static_mods=static_mods,
        cleave_at=cleave_at,
        restrict=restrict,
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        digest=digest,
    ), axis=1)

    # Remove None values
    sample = sample[sample.peptides.notnull()]

    # Remove peptides that have the letter 'U' or 'X' in them
    sample = sample[sample.peptides.apply(lambda x: not any(c in x for c in ['U', 'X']))]

    # Assign protein IDs and events
    sample["protein_id"] = list(range(0, len(sample)))
    sample = assign_events(sample, int(upscale_factor))

    if remove_degenerate_peptides:
        # Pass 1: Count occurrences of each item
        all_items = (item for row_set in sample['peptides'] for item in row_set)
        item_counts = Counter(all_items)

        # Identify items that appear more than once
        shared_items = {item for item, count in item_counts.items() if count > 1}

        # Pass 2: Remove shared items from each set
        sample['peptides'] = sample['peptides'].apply(lambda row_set: list(row_set - shared_items))

    sample["num_peptides"] = sample.peptides.apply(lambda s: len(s))

    sample = sample[sample.num_peptides > 0]

    return sample
