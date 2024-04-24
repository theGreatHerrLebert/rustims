import rustms_connector
rmsc = rustms_connector.py_unimod

UNIMOD_MASSES = rmsc.get_unimod_masses()
UNIMOD_ATOMIC_COMPOSITIONS = rmsc.get_unimod_atomic_compositions()