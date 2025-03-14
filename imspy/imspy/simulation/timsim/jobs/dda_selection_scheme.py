from pathlib import Path
from typing import Tuple

import pandas as pd

from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder

def simulate_dda_pasef_selection_scheme(
        acquisition_builder: TimsTofAcquisitionBuilder,
        verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate DDA selection scheme.

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.

    Returns:
        Tuple of two pandas DataFrames, one holding the DDA PASEF selection scheme and one holding selected precursor information.
    """

    # get builder for synthetic precursor frames
    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(str(Path(acquisition_builder.path) / 'synthetic_data.db'))
    raise NotImplementedError("Not implemented yet.")
