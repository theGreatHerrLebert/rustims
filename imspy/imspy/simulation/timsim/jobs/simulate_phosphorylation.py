import pandas as pd
import numpy as np

from imspy.data.peptide import PeptideSequence
from imspy.simulation.timsim.jobs.utility import phosphorylation_sizes

def simulate_phosphorylation(
        peptides: pd.DataFrame,
        min_phospho_sizes: int = 2,
        pick_phospho_sites: int = 2,
        template: bool = True,
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Simulate phosphorylation for peptides.

    Args:
        peptides: Peptides DataFrame.
        min_phospho_sizes: Minimum number of phosphorylation sizes.
        pick_phospho_sites: Number of phosphorylation sites to pick for each peptide.
        template: Generate template
        verbose: Verbosity.

    Returns:
        pd.DataFrame: Peptides DataFrame.
    """

    assert min_phospho_sizes >= pick_phospho_sites, "min_phospho_sizes must be greater or equal to pick_phospho_sites"

    if template:
        # Filter peptides by phosphorylation sizes
        peptides_filtered = peptides[peptides["sequence"].apply(lambda seq: phosphorylation_sizes(seq)[0] > min_phospho_sizes)].copy()

        # Calculate phosphorylation sites
        peptides_filtered["sites"] = peptides_filtered["sequence"].apply(lambda seq: phosphorylation_sizes(seq)[1])

        # Pick random indices for phosphorylation sites
        peptides_filtered["picked_sites_indices"] = peptides_filtered["sites"].apply(
            lambda sites: np.random.choice(range(len(sites)), pick_phospho_sites, replace=False)
        )

        # Map indices to actual site values
        peptides_filtered["picked_sites"] = peptides_filtered.apply(
            lambda r: [r["sites"][i] for i in r["picked_sites_indices"]], axis=1
        )

        # Assign picked sites to separate columns
        peptides_filtered["phospho_site_a"] = peptides_filtered["picked_sites"].apply(lambda x: x[0])
        peptides_filtered["phospho_site_b"] = peptides_filtered["picked_sites"].apply(lambda x: x[1])

        # Remove unnecessary columns
        peptides_filtered.drop(columns=["sites", "picked_sites"], inplace=True)

        # Save the original sequence
        peptides_filtered["sequence_original"] = peptides_filtered["sequence"]

        # Add UNIMOD:21 to the A site
        peptides_filtered["sequence"] = peptides_filtered.apply(
            lambda r: r["sequence"][:r["phospho_site_a"] + 1] + "[UNIMOD:21]" + r["sequence"][r["phospho_site_a"] + 2:], axis=1
        )

        # Since we changed the sequence, we need to recalculate the monoisotopic mass
        peptides_filtered["monoisotopic-mass"] = peptides_filtered.sequence.apply(lambda p: PeptideSequence(p).mono_isotopic_mass)

        return peptides_filtered

    else:
        if verbose:
            print("Simulating phosphorylation from existing template...")

        # Add UNIMOD:21 to the B site if not a template
        peptides = peptides.copy()
        peptides["sequence"] = peptides.apply(
            lambda r: r["sequence_original"][:r["phospho_site_b"] + 1] + "[UNIMOD:21]" + r["sequence_original"][r["phospho_site_b"] + 2:], axis=1
        )

        # Since we changed the sequence, we need to recalculate the monoisotopic mass
        peptides["monoisotopic-mass"] = peptides.sequence.apply(lambda p: PeptideSequence(p).mono_isotopic_mass)

        return peptides
