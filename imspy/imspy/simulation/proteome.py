import numpy as np
import pandas as pd
import sqlite3

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
                 ):

        # generate enzyme (trypsin)
        enzyme_builder = EnzymeBuilder(
            missed_cleavages=missed_cleavages,
            min_len=min_len,
            max_len=max_len,
            cleave_at=cleave_at,
            restrict=restrict,
            c_terminal=c_terminal
        )

        # generate static cysteine modification
        static_mods = {k: v for k, v in [SAGE_KNOWN_MODS.cysteine_static()]}

        # generate variable methionine modification
        variable_mods = {k: v for k, v in [SAGE_KNOWN_MODS.methionine_variable()]}

        # generate SAGE compatible mod representations
        static = validate_mods(static_mods)
        variable = validate_var_mods(variable_mods)

        # read fasta
        with open(fasta_path, 'r') as f:
            fasta = f.read()

        sage_config = SageSearchConfiguration(
            fasta=fasta,
            enzyme_builder=enzyme_builder,
            static_mods=static,
            variable_mods=variable,
            generate_decoys=generate_decoys,
            bucket_size=int(np.power(2, 6))
        )

        indexed_db = sage_config.generate_indexed_database()

        peptide_list = []

        for i in tqdm(range(indexed_db.num_peptides), desc='Digesting peptides', disable=not verbose, ncols=80):
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

    def write_peptides_to_sqlite(self, path: str):
        """ Write the peptides to a sqlite database

        Parameters
        ----------
        path : str
            Path to the sqlite database
        """
        # write code to connect ot an existing database and write the peptide table,  or overwrite the existing one
        # if it exists
        conn = sqlite3.connect(path)
        self.peptides.to_sql('peptides', conn, if_exists='replace', index=False)
        conn.close()
