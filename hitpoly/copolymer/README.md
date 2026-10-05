# hitpoly.copolymer

Box building for copolymers (random, block, alternating), star polymers, homopolymer blends and
single-ion-conductor (SIC) blends. It does not modify the homopolymer pipeline in
`hitpoly/writers/box_builder.py`; it reuses `hitpoly.data`, `hitpoly.utils` and the OpenMM
protocol in `hitpoly/simulations/openmm_scripts.py` unchanged.

## Running

One system per CSV row; each SLURM array task builds the box and then runs equilibration and
production in the same job:

```bash
cd ~/HiTPoly_copolymer
python run_copolymer.py --csv systems.csv --row 0 --dry_run     # chain length, composition, ion count
sbatch --array=0-27 run_copolymer_array.sh systems.csv ~/results
```

- A row whose box already exists (and passes `check_box`) reuses it; a row whose production run
  finished (`final_state_box.pdb`) is skipped, so resubmitting the whole range is safe.
- Logs: `<results>/logs/<jobid>_<task>.out`. Status per system: `<results>/status/<name>.csv`
  (`box_built`, `sim_finished`, `error`).

CSV layouts (see `pipeline.systems_from_csv`):

1. Case-study lists: `pair_id, cation, structure, fraction_A_B, monomer_A, monomer_B` (names like
   `P14Block_4_8020_Li`; `8020` is the fraction of `monomer_A`).
2. General: `name, kind, monomers` (`A;B`), optional `architecture`, `fractions` (`0.8;0.2`),
   `num_blocks`, `arms`, and any key of `pipeline.DEFAULT_SETTINGS` as a column.

From Python:

```python
from hitpoly.copolymer.pipeline import build_system, run_simulation
build = build_system(dict(kind="copolymer", name="P5Block_4_8020_Li",
                          monomers=["O=C(CCC[Cu])CCCO[Au]", "[Au]CCO[Cu]"],
                          architecture="block", fractions=[0.8, 0.2]),
                     settings={"cation": "Li"}, paths={"results_path": "/path/to/results"})
run_simulation(build)
```

## Monomer orientation

Repeat units are joined with `[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]`. Monomer SMILES have arbitrary
head/tail direction, so a copolymer junction can bond two heteroatoms (Poly 6 + PEO as written gives
an O-O peroxide at every PEO->Poly 6 junction). `junctions.orient_monomers_for_junctions` reverses
monomers where that avoids it and raises if nothing does; `create_long_smiles` and
`create_oligomers` both call it, so oligomer and chain atom types stay consistent, and every
built chain is checked by `junctions.check_chain_junctions`.

## Tests

```bash
python tests/test_copolymer_chains.py    # junctions, composition, counts vs log, oligomers, sizing
```
