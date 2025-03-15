from pathlib import Path
from typing import Tuple

import numpy as np
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

    # set frame types to unknown
    frame_types = np.array(
        acquisition_builder.frame_table.frame_id.apply(lambda fid: -1).values
    )

    # sets the frame types and saves the updated frame table to the blueprint
    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(str(Path(acquisition_builder.path) / 'synthetic_data.db'))

    # TODO: After the precursor table and pasef_meta table are created, the frame_types need to be set to 0 for MS1 frames and 8 for MS2 frames
    """
    # set frame types to unknown
    frame_types = np.array(
        acquisition_builder.frame_table.frame_id.apply(lambda fid: -1).values
    )
    
    # sets the frame types and saves the updated frame table to the blueprint
    acquisition_builder.calculate_frame_types(frame_types=frame_types)
    """

    raise NotImplementedError("Generation of DDA PASEF tables precursor and pasef_meta not yet implemented.")
