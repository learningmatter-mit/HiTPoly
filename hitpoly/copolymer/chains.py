"""
Polymer chain construction: long chains, LigParGen oligomers, and 3D conformers.

create_long_smiles and create_oligomers grow chains with the same reaction and the same
monomer orientation (see junctions.py), so oligomer atom types map onto the long chains.
Any change to how chains are joined must be made in both.
"""
import itertools
import os
import random

import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Geometry import Point3D
from sklearn.metrics.pairwise import euclidean_distances

from hitpoly.copolymer.junctions import check_chain_junctions, orient_monomers_for_junctions

REACTION = "[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]"
COPOLYMER_MODES = ("random", "alternating", "block")


def _split_counts(total, fractions):
    """Integer unit counts per monomer summing exactly to `total` (largest-remainder rounding)."""
    raw = [total * f for f in fractions]
    counts = [int(x + 1e-9) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


def _grow(units, rxn):
    """Join a list of repeat-unit mols head to tail."""
    new_mol = units[0]
    for unit in units[1:]:
        results = rxn.RunReactants((new_mol, unit))
        if not results or len(results[0]) != 1:
            raise RuntimeError("Chain-growth reaction failed; check that every monomer has one [Cu] and one [Au].")
        new_mol = results[0][0]
    return new_mol


def build_star_smiles(monomer, repeats_per_arm, arms, add_end_Cs=True, reaction=REACTION):
    """
    Star (comb-core) molecule: a core of `arms` carbons, C(arm)C(arm)..., each carrying
    one arm of `repeats_per_arm` repeat units. Arms are grown with the same reaction as
    linear chains, so the result is valid for any monomer SMILES ordering.
    """
    rxn = AllChem.ReactionFromSmarts(reaction)
    unit = Chem.MolFromSmiles(monomer)
    arm_smiles = Chem.MolToSmiles(_grow([unit] * repeats_per_arm, rxn))
    cap = "C" if add_end_Cs else "[H]"
    pieces = [arm_smiles.replace("[Cu]", f"[*:{k}]").replace("[Au]", cap) for k in range(1, arms + 1)]
    core = "".join(f"C([*:{k}])" for k in range(1, arms + 1))
    star = Chem.molzip(Chem.MolFromSmiles(".".join([core] + pieces)))
    return Chem.MolToSmiles(star)


def create_long_smiles(
    save_path=".",
    smiles=[],
    fractions=[],
    total_repeats=10,
    add_end_Cs=True,
    reaction=REACTION,
    product_index=0,
    polymerization_mode="homopolymer",
    num_blocks="None",
    arms="None",
    write_log=True,
    log_filename=None,
    cation="None",
):
    """
    Build one chain and return (final_smiles, monomer_counts).

    total_repeats is the number of repeat units in the whole chain, except for stars where
    it is the number per arm. monomer_counts is keyed by the (possibly reoriented) monomer
    SMILES and always matches the units actually in the chain; the log written here is read
    by the RDF atom-labelling step.
    """
    if not smiles or not fractions:
        raise ValueError("'smiles' and 'fractions' cannot be empty.")
    if len(fractions) != len(smiles):
        raise ValueError("'fractions' must have the same length as 'smiles'.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fractions):.6f}.")
    if total_repeats < 1:
        raise ValueError("total_repeats must be at least 1.")

    if polymerization_mode in COPOLYMER_MODES:
        smiles = orient_monomers_for_junctions(smiles)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("One or more SMILES could not be parsed by RDKit.")

    rxn = AllChem.ReactionFromSmarts(reaction)
    block_sizes_and_monomers = None

    if polymerization_mode == "star":
        if len(smiles) != 1:
            raise ValueError("Star polymerization requires exactly one repeat unit SMILES.")
        if arms == "None" or not isinstance(arms, int):
            raise ValueError("'arms' must be set to an integer for star polymerization.")
        final_smiles = build_star_smiles(smiles[0], total_repeats, arms, add_end_Cs, reaction)
        monomer_counts = {smiles[0]: arms * total_repeats}

    else:
        if polymerization_mode == "homopolymer":
            repeated_mols = [mols[0]] * total_repeats

        elif polymerization_mode == "alternating":
            repeated_mols = [mols[i % len(mols)] for i in range(total_repeats)]

        elif polymerization_mode == "random":
            repeated_mols = []
            for count, mol in zip(_split_counts(total_repeats, fractions), mols):
                repeated_mols.extend([mol] * count)
            random.shuffle(repeated_mols)

        elif polymerization_mode == "block":
            if num_blocks == "None" or not isinstance(num_blocks, int):
                raise ValueError("'num_blocks' must be an integer for block copolymerization.")
            if num_blocks % len(mols):
                raise ValueError(f"'num_blocks' ({num_blocks}) must be a multiple of the number of monomers ({len(mols)}).")
            counts = _split_counts(total_repeats, fractions)
            blocks_per_type = num_blocks // len(mols)
            block_sizes_and_monomers = []
            for b in range(num_blocks):
                i, j = b % len(mols), b // len(mols)
                base, extra = divmod(counts[i], blocks_per_type)
                block_sizes_and_monomers.append((base + (1 if j < extra else 0), mols[i]))
            repeated_mols = []
            for size, mol in block_sizes_and_monomers:
                repeated_mols.extend([mol] * size)
        else:
            raise ValueError(f"Unknown polymerization_mode: '{polymerization_mode}'")

        # Mol objects compare by identity, so count() counts each monomer's own units.
        monomer_counts = {Chem.MolToSmiles(m): repeated_mols.count(m) for m in mols}
        final_smiles = Chem.MolToSmiles(_grow(repeated_mols, rxn), canonical=True)
        if add_end_Cs:
            final_smiles = final_smiles.replace("[Cu]", "C").replace("[Au]", "C")

    check_chain_junctions(final_smiles, monomer_counts)
    total_atoms = Chem.AddHs(Chem.MolFromSmiles(final_smiles)).GetNumAtoms()

    if write_log:
        log_fn = log_filename or f"polymer_log_{polymerization_mode}.txt"
        with open(os.path.join(save_path, log_fn), "w") as lf:
            lf.write(f"Architecture: {polymerization_mode}\n")
            lf.write(f"Monomer counts: {monomer_counts}\n")
            lf.write(f"Final SMILES: {final_smiles}\n")
            lf.write(f"Total number of atoms: {total_atoms}\n")
            if block_sizes_and_monomers is not None:
                lf.write(f"Block sizes and monomers: {[(s, Chem.MolToSmiles(m)) for s, m in block_sizes_and_monomers]}\n")

    return final_smiles, monomer_counts


def create_oligomers(smiles, polymerization_mode, arms=3, oligomer_repeats_list=None):
    """
    Short oligomers for LigParGen parameterization, as (products_dict, products_list).

    For blend homopolymer mode, `oligomer_repeats_list` sets the length of each
    component's oligomer independently. Defaults to trimers.
    """
    if len(smiles) == 0 or len(smiles) > 10:
        raise ValueError("smiles list must contain 1-10 monomers.")

    if polymerization_mode in COPOLYMER_MODES:
        smiles = orient_monomers_for_junctions(smiles)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("One or more SMILES strings could not be parsed.")

    rxn = AllChem.ReactionFromSmarts(REACTION)

    if polymerization_mode in ["random", "alternating"]:
        labels = ["A", "B", "C"][: len(mols)]
        mapping = {labels[i]: mols[i] for i in range(len(mols))}
        patterns = {"".join(c): [mapping[x] for x in c] for c in itertools.product(labels, repeat=3)}
        # Add tetramers so that junction atoms (which change neighborhood after orient_monomers_for_junctions)
        # are seen in an interior context (surrounded by real monomers on both sides, not end-caps).
        # Trimers like ABA have A at the end capped with C, giving junction atoms a different depth-2
        # neighborhood than in a long chain. BABA/ABAB put each monomer interior to the other.
        if len(labels) == 2:
            for combo in ["ABAB", "BABA", "ABBA", "BAAB"]:
                patterns[combo] = [mapping[ch] for ch in combo]

    elif polymerization_mode == "block":
        # Kept as-is (including the duplicate patterns for 2 monomers) so LigParGen charge
        # averaging matches the boxes already built with this list.
        valid_combos = [
            "AAA", "BBB", "CCC", "AAB", "ABB", "BAA", "BBA",
            "AAC", "ACC", "CAA", "CCA", "BBC", "BCC", "CBB", "CCB",
        ]
        # Add tetramers for the A-run/B-run junction: in a block copolymer the dihedral
        # spanning the junction needs an A atom whose [Cu] side sees a real B, not a C end-cap.
        if len(mols) == 2:
            valid_combos += ["AABB", "BBAA", "ABBA", "BAAB"]
        mapping = {
            "A": mols[0],
            "B": mols[1] if len(mols) > 1 else mols[0],
            "C": mols[2] if len(mols) > 2 else mols[0],
        }
        patterns = {c: [mapping[ch] for ch in c] for c in valid_combos}

    elif polymerization_mode == "star":
        if len(mols) != 1:
            raise ValueError("Star polymerization requires exactly one repeat unit SMILES.")
        patterns = {"AAA": [mols[0]] * 3, "Core": build_star_smiles(smiles[0], 2, arms)}

    elif polymerization_mode == "homopolymer":
        if len(mols) == 1:
            n = oligomer_repeats_list[0] if oligomer_repeats_list else 3
            patterns = {"A" * n: [mols[0]] * n}
        else:
            labels = [chr(65 + i) for i in range(len(mols))]
            if oligomer_repeats_list is not None:
                if len(oligomer_repeats_list) != len(mols):
                    raise ValueError("oligomer_repeats_list must have same length as smiles.")
                patterns = {
                    labels[i] * oligomer_repeats_list[i]: [mols[i]] * oligomer_repeats_list[i]
                    for i in range(len(mols))
                }
            else:
                patterns = {label * 3: [mol] * 3 for label, mol in zip(labels, mols)}
    else:
        raise ValueError(f"Unknown polymerization_mode: '{polymerization_mode}'")

    products_dict, products_list = {}, []
    for name, pattern in patterns.items():
        if isinstance(pattern, list):
            pattern_smiles = Chem.MolToSmiles(_grow(pattern, rxn), canonical=True)
        else:
            pattern_smiles = pattern  # already a SMILES string
        pattern_smiles = pattern_smiles.replace("[Cu]", "C").replace("[Au]", "C")
        products_dict[name] = pattern_smiles
        products_list.append(pattern_smiles)

    return products_dict, products_list


def get_atom_count(smiles, fractions=None, salt_smiles=[], poly_count=1, salt_count=1):
    """Total atom count (with H) of a SMILES, or a fraction-weighted list of monomer SMILES."""
    num_atoms = 0
    if fractions:
        if len(smiles) != len(fractions):
            raise ValueError("'smiles' and 'fractions' must have the same length.")
        for smile, frac in zip(smiles, fractions):
            num_atoms += Chem.AddHs(Chem.MolFromSmiles(smile)).GetNumAtoms() * frac * poly_count
    else:
        num_atoms += Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms() * poly_count
    for salt in salt_smiles:
        num_atoms += Chem.AddHs(Chem.MolFromSmiles(salt)).GetNumAtoms() * salt_count
    return num_atoms


def get_mol_mass(smiles):
    """Molar mass (g/mol, with H). Same as hitpoly.writers.box_builder.get_mol_mass, kept
    here so chain building does not import torch/OpenMM."""
    return sum(atom.GetMass() for atom in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms())


def create_conformer_pdb(path_to_save, smiles, name=None, enforce_generation=False):
    """
    Write a 3D conformer PDB for a SMILES. Tries RDKit ETKDGv3 first, then Pybel.
    Returns (filename, mol, minimize_flag) or (None, None, None) on failure.
    """
    if not name:
        name = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    file_name = f"{name}.pdb"
    file_path = os.path.join(path_to_save, file_name)
    minimize = True

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    try:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise ValueError("RDKit ETKDGv3 embedding returned -1.")
        AllChem.UFFOptimizeMolecule(mol)
        print("3D conformer generated via RDKit ETKDGv3.")
    except Exception as e:
        print(f"RDKit ETKDGv3 failed: {e}. Falling back to Pybel.")
        mol = None

    if mol is None:
        opt_steps = 1
        while True:
            pybel_mol = pybel.readstring("smi", smiles)
            pybel_mol.make3D(steps=opt_steps)
            pybel_mol.write("pdb", file_path, overwrite=True)
            try:
                temp_mol = Chem.MolFromPDBFile(file_path, removeHs=False)
                if temp_mol is None:
                    raise ValueError("PDB could not be read back by RDKit.")
                X = temp_mol.GetConformer(0).GetPositions()
                dists = euclidean_distances(X, X)
                np.fill_diagonal(dists, np.inf)
                if dists.min() > 0.5:
                    minimize = False
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    rdDepictor.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    for i, atom in enumerate(pybel_mol.atoms):
                        conf.SetAtomPosition(i, Point3D(*atom.coords))
                    print(f"3D conformer generated via Pybel ({opt_steps} steps).")
                    break
                raise ValueError(f"atoms overlap (min distance {dists.min():.2f} A)")
            except Exception as e:
                print(f"Pybel attempt with {opt_steps} steps failed: {e}")
                opt_steps *= 10
                if opt_steps > 10000:
                    print("All conformer generation attempts failed. Skipping molecule.")
                    return None, None, None

    writer = Chem.PDBWriter(file_path)
    writer.write(mol)
    writer.close()
    return file_name, mol, minimize
