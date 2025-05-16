# If you get a ModuleNotFound error install koinapy with `pip install koinapy`.
from koinapy import Koina
import numpy as np
import pandas as pd
from .input_filters import filter_input_by_model


class ModelFromKoina:
    """
    Class to access peptide property prediction models hosted on Koina. Available properties include:
    - Fragment Intensity
    - Retention Time
    - Collision Cross Section
    - Cross-Linking Fragment Intensity
    - Flability
    """

    def __init__(self, model_name: str, host: str = "koina.wilhelmlab.org:443"):
        """
        Args:
            model_name: Name of the Koina model. See all available models at https://koina.wilhelmlab.org/docs
            host: Host of the Koina server.
        """
        self.model = Koina(model_name, host)
        self.model_name = model_name

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the output of the model.
        Args:
            inputs: Input data for the model. requires a DataFrame with the following columns:
                - 'peptide_sequences': Peptide sequence with unimod format, always required.
                - 'precursor_charges': Precursor charge state. Required for some models.
                - 'collision_energies': Collision energy. Required for some models.
                - 'fragmentation_types': Fragmentation type. Required for some models.
                To see what inputs are required for a specific model, run `model.model_inputs`
                For requirements on input sequence length or modifications, see the Koina documentation.
        Returns:
            pd.DataFrame: Output data from the model.
        """
        # check if required columns are present
        required_columns = self.model.model_inputs.keys()
        for col in required_columns:
            if col not in inputs.columns:
                raise ValueError(
                    f"Input DataFrame is missing required column: {col}. Please check the model requirements."
                )
        inputs = filter_input_by_model(
            self.model_name, inputs
        )  # Filter inputs based on model requirements
        predictions = self.model.predict(inputs)
        return predictions
