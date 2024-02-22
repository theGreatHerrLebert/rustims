from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.simulation.utility import read_acquisition_config


def build_acquisition(
        path: str,
        exp_name: str,
        acquisition_type: str,
        verbose: bool = False,
        gradient_length: float = None,
        job_name: str = "build_acquisition"

) -> TimsTofAcquisitionBuilderDIA:

    acquisition_type = acquisition_type.lower()

    assert acquisition_type in ['dia', 'midia', 'slice', 'synchro'], \
        f"Acquisition type must be 'dia', 'midia', 'slice' or 'synchro', was {acquisition_type}"

    config = read_acquisition_config(acquisition_name=acquisition_type)['acquisition']

    if verbose:
        print(f"Using acquisition type: {acquisition_type}")
        print(config)

    if gradient_length is not None:
        config['gradient_length'] = gradient_length

    return TimsTofAcquisitionBuilderDIA.from_config(
        path=path,
        exp_name=exp_name,
        config=config,
    )
