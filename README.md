# rustims
A lightning fast hackable API for timsTOF data written in RUST

Hello, there!

# Repository Structure

This repository contains a Python (imspy) and a Julia (IMSJL) library that share a rust backend (mscore and rustdf).
The backend is accesible for Python and Julia code by the respective connectors pyims_connector and imsjl_connector, using [PyO3](https://docs.rs/pyo3/latest/pyo3/) or directly via the [FFI](https://doc.rust-lang.org/nomicon/ffi.html).

# Build pyims from source

Assuming a [rust](https://www.rust-lang.org/learn/get-started) and Python (>=3.10) version is installed on your system, the
build process currently looks like this:

1.  The Python connector `pyims_connector` needs to be built by [Maturin](https://github.com/PyO3/maturin).
    Maturin can be installed via pip:
    ```shell
    pip install maturin[patchelf]
    ```
    The Python library is installed via [Poetry](https://github.com/python-poetry/poetry).
    Poetry can be installed via pip, as well:
    ```shell
    pip install poetry
    ```
2.  Once Maturin is installed navigate to the `pyims_connector` folder and run:
    ```shell
    maturin build --release
    ```
    This generates a `.whl` file that can be installed by pip.
3.  Install the generated `.whl` file:
    ```shell
    pip install --force-reinstall ./target/wheels/[FILE_NAME].whl
    ```
    The `--force-reinstall` flag ensures that pip is overwriting old installations of the bindings. This
    is relevant when you make changes in the rust backend code (i.e. the bindings themselves, `mscore` or `rustdf`). 
4.  Navigate to the `pyims` folder and install it with Poetry.
    ```shell
    poetry install
    ```

