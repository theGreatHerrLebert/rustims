from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
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
) -> TimsTofAcquisitionBuilderDIA:

    acquisition_type = acquisition_type.lower()

    assert acquisition_type in ['dia', 'midia', 'slice', 'synchro'], \
        f"Acquisition type must be 'dia', 'midia', 'slice' or 'synchro', was {acquisition_type}"

    config = read_acquisition_config(acquisition_name=acquisition_type)['acquisition']

    if gradient_length is not None:
        config['gradient_length'] = gradient_length

    if verbose:
        print(f"Using acquisition type: {acquisition_type}")
        print(config)

    ref_ds = TimsDatasetDIA(reference_path, in_memory=reference_in_memory)

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
