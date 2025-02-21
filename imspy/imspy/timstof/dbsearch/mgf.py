from collections import Counter

from pathlib import Path

import numpy as np
from sagepy.core import (EnzymeBuilder, Precursor, ProcessedSpectrum,
                         RawSpectrum, Representation, SageSearchConfiguration,
                         Scorer, SpectrumProcessor, Tolerance)

def iter_spectra(
    codelines,
    _start_tag: str = "BEGIN IONS",
    _stop_tag: str = "END IONS",
) -> list[dict]:
    """
    Iterate over the spectra in an MGF file
    Args:
        codelines: List of lines from an MGF file
        _start_tag: the tag that indicates the start of a spectrum
        _stop_tag: the tag that indicates the end of a spectrum

    Returns:
        A list of dictionaries, each representing a spectrum
    """
    recording = False
    for line in codelines:
        line = line.strip()
        # Detect sections based on the keywords
        if line.startswith(_start_tag):
            recording = True
            buffer: list[str] = []
        elif line.startswith(_stop_tag):
            assert recording
            recording = False
            yield buffer
        elif recording:
            buffer.append(line)
        else:
            pass
    assert not recording


def parse_spectrum(line_spectrum) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Parse the spectrum from the lines in the MGF file
    Args:
        line_spectrum: List of lines from an MGF file

    Returns:
        precursor_info: Dictionary containing precursor information
        fragment_mzs: Numpy array of fragment m/z values
        fragment_intensities: Numpy array of fragment intensities
    """
    precursor_info = {}
    fragment_mzs = []
    fragment_intensities = []
    precursor_intensity = 0
    for l in line_spectrum:
        if not l[0].isdigit():
            name, val = l.split("=", 1)
            if name == "CHARGE":
                val = int(val.replace("+", ""))
            elif name == "PEPMASS":
                name = "MZ"
                val, precursor_intensity = val.split(" ")
                precursor_intensity = int(precursor_intensity)
                val = float(val)
            elif name == "ION_MOBILITY":
                val = float(val.split(" ")[-1])
            elif name == "RTINSECONDS":
                val = float(val)
            precursor_info[name] = val
        else:
            frag = l.split("\t")
            frag_mz = frag[0]
            frag_intensity = frag[1]
            fragment_mzs.append(float(frag_mz))
            fragment_intensities.append(int(frag_intensity))
    assert precursor_intensity != 0, "We did not manage to parse out precursor intensity!!!"
    precursor_info["intensity"] = precursor_intensity

    try:
        COLLISION_ENERGY = float(precursor_info["title"].split(',')[2].replace(" ", "").replace("eV", ""))
    except Exception as e:
        raise Exception(f"{e}\nERROR IN PARSING COLLISION ENERGY")

    return precursor_info, np.array(fragment_mzs), np.array(fragment_intensities)

def mgf_to_sagepy_query(mgf_path, top_n: int = 150) -> list[ProcessedSpectrum]:
    """
    Read an MGF file and return a list of ProcessedSpectrum
    Args:
        mgf_path: Path to the MGF file
        top_n: Number of top peaks to keep in the spectrum

    Returns:
        List of ProcessedSpectrum
    """
    spec_processor = SpectrumProcessor(take_top_n=top_n)
    queries = []

    with open(mgf_path, "r") as mgf:
        for specNo, line_spectrum in enumerate(iter_spectra(mgf)):
            precursor_info, fragment_mzs, fragment_intensities = parse_spectrum(line_spectrum)
            precursor = Precursor(
                mz=precursor_info["MZ"],
                charge=precursor_info["CHARGE"],
                intensity=precursor_info["intensity"],
                inverse_ion_mobility=precursor_info.get("ION_MOBILITY", None),
                collision_energy=precursor_info.get("COLLISION_ENERGY", None),
            )
            prec_rt = precursor_info["RTINSECONDS"] / 60.0
            raw_spectrum = RawSpectrum(
                file_id=1,
                spec_id=str(specNo),
                total_ion_current=fragment_intensities.sum(),
                precursors=[precursor],
                mz=fragment_mzs.astype(np.float32),
                intensity=fragment_intensities.astype(np.float32),
                scan_start_time=prec_rt,
                ion_injection_time=prec_rt,
            )
            queries.append(spec_processor.process(raw_spectrum))

    return queries