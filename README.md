# rustims

<p align="center">
  <img src="rustims_logo.png" alt="logo" width="250"/>
</p>

`rustims` is a framework developed for processing raw data from Ion-Mobility Spectrometry (IMS) in proteomics mass spectrometry. It draws inspiration from OpenMS but is distinguished by its use of [Rust](https://www.rust-lang.org/) as the backend language, aiming for efficient algorithm implementations and robust data structures. Like OpenMS, `rustims` exposes most of its logic to Python, here via [PyO3](https://docs.rs/pyo3/latest/pyo3/). This setup is intended to enable easy pick-up, quick prototyping, and integration into existing Python-centric scientific workflows. 

`rustims` is about exploring and improving the way we process ion-mobility spectrometry data, providing re-usable building blocks that can be extensively configured. It's a work in progress, reflecting the open-source ethos of collaboration, engagement, and sharing of knowledge. Whether you're here to contribute or learn, we welcome your interest!

# Quickstart
To quickly get started, we recommend installing the Python package `imspy`, the high-level Python API designed for most users to interact with `rustims` functionality, via pip into a separate Virtual
Environment using Python3.11 (currently the only supported Python version due to TensorFlow). If you don't know how to create a Virtual Environment, you can follow the instructions [here](https://docs.python.org/3/library/venv.html).
This way, you can avoid potential dependency conflicts with other Python packages.
The following command installs the latest version of `imspy` from PyPi:
```shell
pip install imspy
```
This will install tensorflow as a dependency without GPU support.
The easiest way to get GPU support is to additionally install the tensorflow[and-cuda] package:
```shell
pip install tensorflow[and-cuda]==2.15.*
```
Which comes with the necessary CUDA and cuDNN libraries.
Have a look at the [imspy-README](https://github.com/theGreatHerrLebert/rustims/tree/main/imspy) to learn about basic functionalities of the package.

# Repository Structure
<figure align="center">
  <img src="rustims_layout.png" alt="RustIMS Project Structure" width="700"/>
  <figcaption>
    The <em>rustims</em> project architecture is designed around two core Rust crates: 
    <code>mscore</code> and <code>rustdf</code>. These crates are the foundation of the project, 
    housing the in-memory data structures, algorithms, and input/output functionalities 
    specifically for TDF files. These Rust components are seamlessly integrated with Python 
    through <code>pyO3</code>, which allows the main functionalities of <code>mscore</code> 
    and <code>rustdf</code> to be accessible in Python by compiling them into a single, 
    installable Python wheel named <code>imspy_connector</code>. On top of this, 
    <code>imspy</code> is a native Python package that not only interfaces with the Rust 
    crates for enhanced performance but also introduces additional logic, such as TensorFlow 
    models for ion-mobility prediction, thereby combining the strengths of Rust and Python in 
    one cohesive framework.
  </figcaption>
</figure>

## Analyzing a DDA dataset from Bruker timsTOF with imspy_dda
You can directly run the `imspy_dda` command to analyze a DDA dataset:
```shell
imspy_dda path/to/bruker.tdf path/to/proteome.fasta
```
The tool has a lot of options, which you can explore by running:
```shell
imspy_dda --help
```

## Dive into processing of timsTOF DDA data with jupyter notebooks
We are now providing [jupyter notebook examples](https://github.com/theGreatHerrLebert/rustims/blob/main/imspy/examples/) that allow you to interactively learn about the functionality of our tooling. You can also checkout the [sagepy notebook examples](https://github.com/theGreatHerrLebert/sagepy/tree/main/sagepy/examples) which are hosted in a separate repository since sagepy is not limited to DB searching of timsTOF data.

## Read the docs
The codbease of all native rust crates and python packages is now available:

* [rustdf](https://thegreatherrlebert.github.io/rustims/main/rustdf/)
* [mscore](https://thegreatherrlebert.github.io/rustims/main/mscore/)
* [rustms](https://thegreatherrlebert.github.io/rustims/main/rustms/)
* [imspy](https://thegreatherrlebert.github.io/rustims/main/imspy/)

## Generating Synthetic PASEF-like Datasets with TimSim

**TimSim** is a versatile simulation tool designed for generating synthetic PASEF-like datasets for proteomics experiments on Bruker TimsTOF instruments. It offers two complementary modes of operation:

### Command-Line Mode

TimSim can be run directly from the terminal. You have two convenient options:

1. **Direct Parameter Specification:**  
   Provide the required positional arguments along with any desired options:
   ```bash
   timsim path/to/output.tdf path/to/reference.tdf path/to/proteome.fasta [--option value ...]
   ```
2. **Configuration File Mode:**  
   To simplify repetitive runs, supply all simulation parameters via a TOML configuration file using the `--config` option:
   ```bash
   timsim --config path/to/config.toml
   ```
   This approach lets you store and reuse complete simulation setups without having to specify each parameter on the command line.

### Graphical User Interface (GUI) Mode

For an interactive, user-friendly experience, launch the TimSim GUI. The GUI allows you to:
- Adjust simulation parameters via intuitive controls.
- Visualize real-time logs and plots.
- Experiment with settings before executing a full simulation.

You can start the GUI with:
```bash
timsim_gui
```
### Example Data and Configuration

To help you get started, example datasets and sample configuration files are available on our [Zenodo repository](https://zenodo.org/record/XXXXXX) (link coming soon). These examples demonstrate common workflows and parameter settings for both the command-line and GUI modes.

## Rust backend: mscore and rustdf
There are two Rrust projects: `mscore` and `rustdf`. The former is a library that contains implementations of in-memory data structures and algorithms for raw-data processing. The latter contains a Rust-native reader and writer of TDF, the serialization format written by [Bruker timsTOF](https://www.bruker.com/en/products-and-solutions/mass-spectrometry/timstof.html) devices. It also contains the implementation of the I/O logic needed for synthetic timsTOF PASEF-like in-silico dataset generation.

## Python bindings: imspy_connector
The `imspy_connector` module bridges Rust code with Python, allowing Rust components to be used in Python with minimal dependencies. This setup keeps the system lightweight for Python users but introduces complexity, especially in development and debugging. Changes in Rust need to be reflected in Python, often requiring updates in multiple places. Despite the added complexity, this architecture is chosen for its benefits. It allows for parts of the code in Rust or Python that don't interact with the other language to be developed independently and asynchronously. However, this flexibility is limited to components that do not require cross-language access.

## Python package: imspy
`imspy` is a Python package designed for end-users. It utilizes `imspy_connector` for accessing Rust functionalities exposed via `pyO3`, incorporating additional libraries like `tensorflow`, `scikit-learn`, and `sagepy`. This setup enables users to perform detailed tasks such as calculating peptide fragment ions, analyzing isotope patterns, studying quadrupole transmission, and applying deep learning to ion mobility and retention time predictions.

## Julia bindings
Julia support is currently experimental. Julia interfaces via `imsjl_connector`, [FFI](https://doc.rust-lang.org/nomicon/ffi.html).

# Installation

## Install via pip
We are now providing stable versions of the python-bound components via Python wheels on PyPi. We recommend that you use a [Python virtual environment](https://docs.python.org/3/library/venv.html) with `python3.11`, since imspy has some heavy weight dependencies like `tensorflow`, `numpy`, and `numba`, where version mismatches can lead to potential issues.
```shell
pip install imspy
```

## Build from source
## Rust backend
Assuming a [rust](https://www.rust-lang.org/learn/get-started) is installed on your system and you cloned this repository, the build process currently looks like this (example for mscore):
```shell
cd rustims/mscore && cargo build --release
```

## Python bindings
Assuming a [rust](https://www.rust-lang.org/learn/get-started) and Python (==3.11) version is installed on your system, the
build process currently looks like this:

1.  The Python connector `imspy_connector` needs to be built by [Maturin](https://github.com/PyO3/maturin).
    Maturin can be installed via pip:
    ```shell
    pip install maturin[patchelf]
    ```
2.  Once Maturin is installed navigate to the `imspy_connector` folder and run:
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
    
## Julia bindings
Julia support is currently experimental.

## Python package
The Python library is installed via [Poetry](https://github.com/python-poetry/poetry).
1.  Poetry can be installed via pip:
    ```shell
    pip install poetry
    ```
2.  Navigate to the `imspy` folder and install it with Poetry.
    ```shell
    poetry install
    ```
## Docker image
For ease of use and reproducable running of our software, we are now providing a docker image for the AMD64 architecture.
It can be also used on macOS ARM64 via virtualization. To get the image, download the here linked [release.zip](https://github.com/MatteoLacki/rustims_docker/blob/main/release.zip), unzip it and follow the installation instructions provided inside the readme.md file.
