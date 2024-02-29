import imspy_connector
ims = imspy_connector.py_chemistry


def calculate_mz(mass: float, charge: int) -> float:
    """Calculate m/z value.

    Args:
        mass (float): Mass.
        charge (int): Charge.

    Returns:
        float: m/z value.
    """
    return ims.calculate_mz(mass, charge)

