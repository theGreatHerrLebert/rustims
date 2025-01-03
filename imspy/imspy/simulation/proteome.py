import numpy as np
import pandas as pd

from sagepy.core.database import PeptideIx
from sagepy.core import EnzymeBuilder, SAGE_KNOWN_MODS, validate_mods, validate_var_mods, SageSearchConfiguration
from tqdm import tqdm


class PeptideDigest:
    def __init__(self,
                 fasta_path: str,
                 missed_cleavages: int = 2,
                 min_len: int = 7,
                 max_len: int = 50,
                 cleave_at: str = 'KR',
                 restrict: str = 'P',
                 generate_decoys: bool = False,
                 c_terminal: bool = True,
                 verbose: bool = True,
                 variable_mods: dict[str, list[str]] = { "M": ["[UNIMOD:35]"], "[": ["[UNIMOD:1]"] },
                 static_mods: dict[str, str] = {"C": "[UNIMOD:4]"}
                 ):

        self.verbose = verbose
        self.peptides = None
        self.min_len = min_len
        self.max_len = max_len
        self.missed_cleavages = missed_cleavages
        self.cleave_at = cleave_at
        self.restrict = restrict
        self.generate_decoys = generate_decoys
        self.c_terminal = c_terminal
        self.fasta_path = fasta_path
        self.variable_mods = variable_mods
        self.static_mods = static_mods

        self._setup()

    def _setup(self):
        try:
            fasta = open(self.fasta_path, 'r').read()
        except FileNotFoundError:
            print(f"Could not find fasta file at {self.fasta_path}")
            raise FileNotFoundError

        enzyme_builder = EnzymeBuilder(
            missed_cleavages=self.missed_cleavages,
            min_len=self.min_len,
            max_len=self.max_len,
            cleave_at=self.cleave_at,
            restrict=self.restrict,
            c_terminal=self.c_terminal
        )

        sage_config = SageSearchConfiguration(
            fasta=fasta,
            enzyme_builder=enzyme_builder,
            static_mods=self.static_mods,
            variable_mods=self.variable_mods,
            generate_decoys=self.generate_decoys,
            bucket_size=int(np.power(2, 6))
        )

        indexed_db = sage_config.generate_indexed_database()

        peptide_list = []

        for i in tqdm(range(indexed_db.num_peptides), desc='Digesting peptides', disable=not self.verbose, ncols=100):
            idx = PeptideIx(idx=i)
            peptide = indexed_db[idx]

            peptide_list.append({'peptide_id': i,
                                 'sequence': peptide.to_unimod_sequence(),
                                 'proteins': '\t'.join(peptide.proteins),
                                 'decoy': peptide.decoy,
                                 'missed_cleavages': peptide.missed_cleavages,
                                 'n_term': peptide.n_term,
                                 'c_term': peptide.c_term,
                                 'monoisotopic-mass': peptide.mono_isotopic,
                                 })

        self.peptides = pd.DataFrame(peptide_list)
