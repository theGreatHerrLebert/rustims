from .utility import check_path
from imspy.simulation.proteome import PeptideDigest


def digest_fasta(
        fasta_file_path: str,
        missed_cleavages: int = 2,
        min_len: int = 6,
        max_len: int = 30,
        cleave_at: str = 'KR',
        restrict: str = None,
        decoys: bool = False,
        verbose: bool = False,
        job_name: str = "digest_fasta",
) -> PeptideDigest:
    """Digest a fasta file.

    Args:
        fasta_file_path: Path to the fasta file.
        missed_cleavages: Number of missed cleavages.
        min_len: Minimum peptide length.
        max_len: Maximum peptide length.
        cleave_at: Cleavage sites.
        restrict: Restrict to specific proteins.
        decoys: Generate decoys.
        verbose: Verbosity.
        job_name: Job name.

    Returns:
        PeptideDigest: Peptide digest object.
    """
    if verbose:
        print("Digesting peptides...")

    return PeptideDigest(
        check_path(fasta_file_path),
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        cleave_at=cleave_at,
        restrict=restrict,
        generate_decoys=decoys,
        verbose=verbose,
    )
