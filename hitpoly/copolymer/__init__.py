"""
Copolymer, star, blend and single-ion-conductor box building for HiTPoly.

Kept separate from hitpoly.writers.box_builder (the homopolymer pipeline), which it does
not modify. Modules:
    junctions      monomer orientation so copolymer junctions never bond two heteroatoms
    chains         long chains, LigParGen oligomers, 3D conformers
    concentration  chain sizing and salt counts
    ligpargen      LigParGen runs and the combined parameter dict
    dft_charges    DFT (CHELPG) charges from htvs for SIC components
    box            PACKMOL box and OpenMM force-field files
    labels         RDF atom labels for alternating/block chains
    pipeline       system dicts -> build -> MD; CSV batches (see run_copolymer.py)
"""
