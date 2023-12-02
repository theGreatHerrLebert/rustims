import numpy as np
import pandas as pd

from sagepy.core.database import PeptideIx
from sagepy.core import EnzymeBuilder, SAGE_KNOWN_MODS, validate_mods, validate_var_mods, SageSearchConfiguration
from tqdm import tqdm

from imspy.simulation.exp import ExperimentDataHandle


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

        # generate static cysteine modification
        static_mods = {k: v for k, v in [SAGE_KNOWN_MODS.cysteine_static()]}

        # generate variable methionine modification
        variable_mods = {k: v for k, v in [SAGE_KNOWN_MODS.methionine_variable()]}

        # generate SAGE compatible mod representations
        static = validate_mods(static_mods)
        variable = validate_var_mods(variable_mods)

        sage_config = SageSearchConfiguration(
            fasta=fasta,
            enzyme_builder=enzyme_builder,
            static_mods=static,
            variable_mods=variable,
            generate_decoys=self.generate_decoys,
            bucket_size=int(np.power(2, 6))
        )

        indexed_db = sage_config.generate_indexed_database()

        peptide_list = []

        for i in tqdm(range(indexed_db.num_peptides), desc='Digesting peptides', disable=not self.verbose, ncols=80):
            idx = PeptideIx(idx=i)
            peptide = indexed_db[idx]

            peptide_list.append({'peptide_id': i,
                                 'sequence': peptide.to_unimod_sequence(),
                                 'proteins': '\t'.join(peptide.proteins),
                                 'decoy': peptide.decoy,
                                 'missed_cleavages': peptide.missed_cleavages,
                                 'n_term': peptide.n_term,
                                 'c_term': peptide.c_term,
                                 'monoisotopic-mass': peptide.mono_isotopic})

        self.peptides = pd.DataFrame(peptide_list)

    def write_to_sqlite(self, handle: ExperimentDataHandle):
        if self.verbose:
            print(f'writing {self.peptides.shape[0]} peptides to {handle.db_path}')

        handle.create_table('peptides', self.peptides)
