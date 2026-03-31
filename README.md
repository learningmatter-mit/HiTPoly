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
3. Install OpenBabel on your workstation `conda install openbabel -c openbabel -y`

HiTPoly can run simulations and interface with either Gromacs or OpenMM. These MD engines are not installed automatically by `pip install .`.

## Installation of MD Engines

### Gromacs Installation
Gromacs can be installed via package managers or built from source. For optimal performance, we recommend building from source as described on the [Gromacs website](https://manual.gromacs.org/current/install-guide/index.html)

### OpenMM Installation

Currently the simulation engine is programmed to be using CUDA. To install OpenMM with cuda run either:
`conda install -c conda-forge openmm cuda-version=12 -y`
or
`pip install openmm[cuda12]`

To use HiTPoly with cpu compiled CUDA, platform name has to be adjusted in hitpoly/simulations/openmm_scripts.py


## Usage
HiTPoly is mainly run through three entry-point scripts, depending on how much of the system specification you want to provide manually and whether you are running a single setup or a screening workflow.

### 1. `builder_ligpargen_openmm.py`

Use `builder_ligpargen_openmm.py` when you want fine-grained control over a simulation setup. This script expects the polymer or molecule identity together with explicit settings such as `solvent_count`, `repeats`, salt concentration, charge scaling, temperature, system type, and simulation length.

`--smiles_path` can be either a SMILES string directly or a file containing one SMILES per line. This is the lower-level entry point used when you already know the exact simulation inputs you want to run.

```bash
python builder_ligpargen_openmm.py \
  --save_path results/example_run \
  --results_path results/example_run \
  --smiles_path "[Cu]CCO[Au]" \
  --solvent_count 30 \
  --repeats 50 \
  --salt_type Li.TFSI \
  --concentration 100 \
  --charge_scale 0.75 \
  --charge_type LPG \
  --temperature 430 \
  --system polymer \
  --simu_length 100
```

An example workflow for reproducing the results from [this paper](https://pubs.acs.org/doi/full/10.1021/acs.macromol.5c00930) is provided in `building.py`. That script reads a CSV of polymer jobs and calls `builder_ligpargen_openmm.py` to generate Slurm scripts, which can also be run directly as bash scripts.

### 2. `builder_ligpargen_openmm_from_molality_multi_system.py`

Use `builder_ligpargen_openmm_from_molality_multi_system.py` when you want HiTPoly to build a polymer, liquid, or gel-polymer electrolyte system from a `smiles_path` input file and a target salt molality. This script derives the composition from the file contents rather than requiring you to manually specify solvent counts and repeat counts for each component.

The `smiles_path` file should contain:

```text
SMILES1.SMILES2.SMILES3
[0.5, 0.3, 0.2]
mol
```

Example for EC/DMC/EMC liquid system:

```text
C1COC(=O)O1.COC(=O)OC.CCOC(=O)OC
[0.33, 0.33, 0.33]
mol
```


The first line contains the component SMILES strings, the second line contains their ratios, and the optional third line sets the ratio type (`mol` or `weight`).

```bash
python builder_ligpargen_openmm_from_molality_multi_system.py \
  --save_path results/molality_run \
  --results_path results/molality_run \
  --smiles_path path/to/system_smiles.txt \
  --salt_type Li.TFSI \
  --molality_salt 1.0 \
  --system liquid \
  --temperature 393 \
  --simu_length 100
```

This is the easiest route when you want to describe a full mixed system in terms of components and ratios rather than hand-tuning every packing input.

## Screening with Bayesian Optimization

If you want to embedd your SMILES as descriped in the paper (PCA on long representation from MolFormer), run the script generate_embedding.py for your trianing and testing data and save it in a folder named batch0 which is a subfolder of where you want to save your batches. With the ht_script.sh script you can run the high throughput BO loop. This pipeline works based on a folder on your local machine that saves the files for each batch. You also need some method/file that keeps track of how many simulations have suceeded for that batch. We suggest a csv file that has the results from all previous simulations which also includes the polymers for which the simulations did not suceed with Nan values for the properties. The example code is written for a csv file that has the columns smiles and property. Further pointers are given in the ht_script.sh script.

## References:

Ruza, J., Leon, P., Jun, K., Johnson, J., Shao-Horn, Y. and Gomez-Bombarelli, R., 2025. [Benchmarking classical molecular dynamics simulations for computational screening of lithium polymer electrolytes.](https://pubs.acs.org/doi/full/10.1021/acs.macromol.5c00930) Macromolecules, 58(13), pp.6732-6742.

Kuhn, A.S., Ruža, J., Jun, K., Leon, P. and Gómez-Bombarelli, R., 2026. [Discovery of Polymer Electrolytes with Bayesian Optimization and High-Throughput Molecular Dynamics simulations.](https://arxiv.org/abs/2602.17595) arXiv preprint arXiv:2602.17595.

## License

This project is licensed under the MIT License.
