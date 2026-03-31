# HiTPoly

A platform for setting up high throughput polymer electrolyte MD simulations.

## Installation

`HiTPoly` is distributed as a Python package named `hitpoly` and requires Python `>=3.6`.

We recommend installing it in a dedicated `conda` environment named `hitpoly`:

```bash
# Clone the repository
git clone https://github.com/learningmatter-mit/HiTPoly.git
cd HiTPoly

# Create and activate a dedicated conda environment
conda create -n hitpoly python=3.10 -y
conda activate hitpoly

# Install HiTPoly from the local source tree
pip install .
```

If you plan to actively develop the package, install it in editable mode instead:

```bash
pip install -e .
```

This installation pulls in the Python dependencies declared in `pyproject.toml`.

## Python Dependencies

The main Python dependencies installed with `pip install .` are:

- `numpy>=1.20.0`
- `pandas`
- `scipy`
- `torch`
- `rdkit`
- `matplotlib`
- `typing-extensions`
- `typed-argument-parser`
- `scikit-learn`

## External Requirements

To run HiTPoly, you also need to install the following external tools separately:

1. Download and install LigParGen locally on your machine following the tutorial [here](https://github.com/learningmatter-mit/ligpargen/)
2. Install Packmol on your workstation, [LINK](https://m3g.github.io/packmol/)
3. Install OpenBabel on your workstation `conda install openbabel -c openbabel`

HiTPoly can run simulations and interface with either Gromacs or OpenMM. These MD engines are not installed automatically by `pip install .`.

## Installation of MD Engines

### Gromacs Installation
Gromacs can be installed via package managers or built from source. For optimal performance, we recommend building from source as described on the [Gromacs website](https://manual.gromacs.org/current/install-guide/index.html)

### OpenMM Installation

Currently the simulation engine is programmed to be using CUDA. To install OpenMM with cuda run either:
`conda install -c conda-forge openmm cuda-version=12`
or
`pip install openmm[cuda12]`

To use HiTPoly with cpu compiled CUDA, platform name has to be adjusted in hitpoly/simulations/openmm_scripts.py


## Usage

Full tutorial coming soon.

## Screening with Bayesian Optimization

If you want to embedd your SMILES as descriped in the paper (PCA on long representation from MolFormer), run the script generate_embedding.py for your trianing and testing data and save it in a folder named batch0 which is a subfolder of where you want to save your batches. With the ht_script.sh script you can run the high throughput BO loop. This pipeline works based on a folder on your local machine that saves the files for each batch. You also need some method/file that keeps track of how many simulations have suceeded for that batch. We suggest a csv file that has the results from all previous simulations which also includes the polymers for which the simulations did not suceed with Nan values for the properties. The example code is written for a csv file that has the columns smiles and property. Further pointers are given in the ht_script.sh script.

Please cite:

## License

This project is licensed under the MIT License.
