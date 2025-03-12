from imspy.simulation.acquisition import TimsTofAcquisitionBuilder, TimsTofAcquisitionBuilderDDA, TimsTofAcquisitionBuilderDIA
from imspy.simulation.utility import read_acquisition_config
from imspy.timstof import TimsDataset, TimsDatasetDIA


def build_acquisition(
        path: str,
        reference_path: str,
        exp_name: str,
        acquisition_type: str = 'dia',
        verbose: bool = False,
        gradient_length: float = None,
        use_reference_ds_layout: bool = True,
        reference_in_memory: bool = True,
        round_collision_energy: bool = True,
        collision_energy_decimals: int = 0,
        use_bruker_sdk: bool = True,
) -> TimsTofAcquisitionBuilder:
    """Build acquisition object from reference path.

    Args:
        path: Path where the acquisition will be saved.
        reference_path: Path to the reference dataset.
        exp_name: Experiment name.
        acquisition_type: Acquisition type, must be 'dia', 'midia', 'slice' or 'synchro'.
        verbose: Verbosity.
        gradient_length: Gradient length.
        use_reference_ds_layout: Use reference dataset layout for synthetic dataset.
        reference_in_memory: Load reference dataset into memory.
        round_collision_energy: Round collision energy.
        collision_energy_decimals: Number of decimals for collision energy (controls coarseness).
        use_bruker_sdk: Use Bruker SDK for reading reference dataset.

    Returns:
        TimsTofAcquisitionBuilderDIA: Acquisition object.
    """

    acquisition_type = acquisition_type.lower()
    known_acquisitions = ['dda', 'dia', 'midia', 'slice', 'synchro']
    assert acquisition_type.lower() in known_acquisitions, f"Unknown acquisition type: {acquisition_type}, " \
                                                              f"must be one of {known_acquisitions}."

    if acquisition_type.lower() == 'dda':

        if verbose:
            print("Using DDA acquisition type...")

        # Load reference dataset
        reference_ds = TimsDataset(reference_path)

        if gradient_length is None:
            gradient_length = 3600.0

        rt_cycle_length = 0.109

        return TimsTofAcquisitionBuilderDDA(
            path=path,
            reference_ds=reference_ds,
            exp_name=exp_name,
            verbose=verbose,
            gradient_length=gradient_length,
            rt_cycle_length=rt_cycle_length
        )

    else:
        config = read_acquisition_config(acquisition_name=acquisition_type)['acquisition']

        if gradient_length is not None:
            config['gradient_length'] = gradient_length

        if verbose:
            print(f"Using acquisition type: {acquisition_type}")
            print(config)

        ref_ds = TimsDatasetDIA(reference_path, in_memory=reference_in_memory, use_bruker_sdk=use_bruker_sdk)

        return TimsTofAcquisitionBuilderDIA.from_config(
            path=path,
            reference_ds=ref_ds,
            exp_name=exp_name,
            config=config,
            verbose=verbose,
            use_reference_layout=use_reference_ds_layout,
            round_collision_energy=round_collision_energy,
            collision_energy_decimals=collision_energy_decimals
        )
