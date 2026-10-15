"""
Box packing and OpenMM force-field files for copolymer / blend / SIC systems.

create_box_and_ff_files_openmm runs: PACKMOL box -> hitpoly typing of chains + salt ->
parameter assignment from the oligomer param_dict -> PDB/mol_dict matching -> atom
renaming -> force_field.xml, force_field_resids.xml and packed_box.pdb.
"""
import itertools
import math
import string as STRING
import subprocess
from collections import Counter

import numpy as np
import torch
from openmm.app import PDBFile
from rdkit import Chem
from scipy.spatial import distance_matrix

from hitpoly.data.builder import TopologyBuilder, TopologyDataset
from hitpoly.data.ff_file_reader import ForceFieldFileReader
from hitpoly.utils.args import hitpolyArgs
from hitpoly.writers.box_builder import param_scaling_openmm

TFSI_SMILES = "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"


def salt_setup(cation, hitpoly_path):
    """Salt SMILES, geometry and LAMMPS parameter files for <cation>TFSI."""
    return dict(
        salt_smiles=[f"[{cation}+]", TFSI_SMILES],
        salt_paths=[
            f"{hitpoly_path}/data/pdb_files/geometry_file_{cation}.pdb",
            f"{hitpoly_path}/data/pdb_files/geometry_file_TFSI.pdb",
        ],
        salt_data_paths=[
            f"{hitpoly_path}/data/forcefield_files/lammps_{cation}_q100.data",
            f"{hitpoly_path}/data/forcefield_files/lammps_TFSI_q100.data",
        ],
    )


def create_packmol_input_file(
    poly_paths,
    salt_paths,
    output_path,
    output_name,
    salt_smiles=None,
    polymer_count=25,
    salt_concentrations=[100, 100],
    tolerance=5.0,
    box_multiplier=3,
    random_seed=-1,
    salt=True,
):
    """
    Write a PACKMOL input file. polymer_count is an int (same for every polymer file) or a
    list with one count per file. Salts with a zero count (e.g. TFSI in SIC systems) are left out.
    """
    if isinstance(polymer_count, (int, float)):
        polymer_count = [int(polymer_count)] * len(poly_paths)
    if len(polymer_count) != len(poly_paths):
        raise ValueError("Length of polymer_count must match number of polymer files")

    total_vol = 0.0
    for path, count in zip(poly_paths, polymer_count):
        # Only positions are needed here. Bond perception can assign an O a third
        # bond when two chain atoms sit too close, which makes the parse return None.
        mol = Chem.MolFromPDBFile(
            path, removeHs=False, sanitize=False, proximityBonding=False
        )
        pos = mol.GetConformer().GetPositions()
        radius = distance_matrix(pos, pos).max() / 2
        total_vol += 4 / 3 * np.pi * radius**3 * count
    total_vol *= box_multiplier
    box_radi = (3 / (4 * np.pi) * total_vol) ** (1 / 3)
    print(f"{len(poly_paths)} polymer file(s), {sum(polymer_count)} chains, box side {box_radi:.2f} A")

    cation_paths, anion_paths, cation_conc, anion_conc = [], [], [], []
    if salt:
        if len(salt_paths) > 2 and salt_smiles:
            for ind, (path, smi) in enumerate(zip(salt_paths, salt_smiles)):
                net_charge = smi.count("+") - smi.count("-")
                if net_charge == 1 and salt_concentrations[ind] > 0:
                    cation_paths.append(path)
                    cation_conc.append(salt_concentrations[ind])
                elif net_charge == -1 and salt_concentrations[ind] > 0:
                    anion_paths.append(path)
                    anion_conc.append(salt_concentrations[ind])
        elif len(salt_paths) == 2:
            if salt_concentrations[0] > 0:
                cation_paths, cation_conc = [salt_paths[0]], [salt_concentrations[0]]
            if salt_concentrations[1] > 0:
                anion_paths, anion_conc = [salt_paths[1]], [salt_concentrations[1]]
        else:
            print("Warning: Salt input incomplete or ambiguous - no salts will be packed.")
    print(f"Cations: {cation_conc}, anions: {anion_conc}")

    box = f"  inside box 0 0 0 {box_radi:.2f} {box_radi:.2f} {box_radi:.2f}\n"
    with open(output_path, "w+") as f:
        f.write(f"tolerance {tolerance}\nfiletype pdb\noutput {output_name}\n\n")
        for paths, counts in ((poly_paths, polymer_count), (anion_paths, anion_conc), (cation_paths, cation_conc)):
            for path, n in zip(paths, counts):
                f.write(f"structure {path}\n  number {n}\n{box}end structure\n\n")


def run_packmol(save_path, packmol_path):
    result = subprocess.run(
        f"{packmol_path} < {save_path}/packmol.inp",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PACKMOL failed:\nSTDOUT: {result.stdout[-3000:]}\nSTDERR: {result.stderr}")
    print("PACKMOL completed successfully.")


def load_hitpoly_params(poly_smiles, salt_smiles, salt_data_paths=[], charge_scale=None,
                        poly_count=1, neighborhood_depth=2):
    """
    Topology for every unique polymer SMILES followed by the salts. Salt parameters come
    from LAMMPS data files; polymer parameters are assigned later by assign_lpg_params.
    """
    train_args = hitpolyArgs()
    train_args.discrete_neighborhood_depth = neighborhood_depth
    train_dataset = TopologyDataset(data=[
        TopologyBuilder(smiles=[s], train_args=train_args, load_geoms=False, lj_from_file=False)
        for s in poly_smiles + salt_smiles
    ])

    for mol, path in zip(train_dataset[poly_count:], salt_data_paths):
        ff_read = ForceFieldFileReader(mol, path)
        ff_read.load_pair_params()
        ff_read.load_bond_params()
        ff_read.load_angle_params()
        ff_read.load_dihedral_params()

    if charge_scale:
        # Polymer charges are overwritten in assign_lpg_params, so this only affects the salt.
        for mol in train_dataset[1:]:
            mol.pair_params[:, 0] = mol.pair_params[:, 0] * charge_scale
    return train_dataset


def assign_lpg_params(
    short_smiles_list,
    atoms_long_list,
    atom_names_long_list,
    atoms_short,
    atom_names_short,
    train_dataset,
    param_dict,
    sic_component_indices=None,
):
    """
    Assign LigParGen parameters from param_dict to every polymer entry of train_dataset by
    atom-type ID (falling back to charge-matched bonds/angles and type-matched dihedrals),
    then neutralize the non-SIC polymers. Raises if any bond, angle or dihedral is unmatched.
    """
    sic_component_indices = sic_component_indices or []

    def _try_int(k):
        try:
            return int(k)
        except (TypeError, ValueError):
            return k

    def _rq(x, nd=3):
        return round(float(x), nd)

    def _coerce(seq, template_item):
        T = type(template_item)
        out = []
        for s in seq:
            try:
                out.append(T(s))
            except Exception:
                out.append(s)
        return out

    param_dict["atoms"] = {_try_int(k): v for k, v in param_dict.get("atoms", {}).items()}

    charges_dict = {}
    for key, val in param_dict["atoms"].items():
        qs = val.get("charges", [])
        q = (sum(qs) / len(qs)) if qs else 0.0
        if isinstance(q, float) and (math.isnan(q) or math.isinf(q)):
            print(f"[charges][warn] non-finite mean charge for type {key}; set to 0.")
            q = 0.0
        charges_dict[key] = q

    def charge_of(name):
        return charges_dict.get(name, charges_dict.get(_try_int(name), 0.0))

    for i, atom_names_long in enumerate(atom_names_long_list):
        mol = train_dataset[i]
        print(f"Assigning parameters to polymer {i} ({len(atom_names_long)} atoms)")

        # Pairs (q, epsilon, sigma)
        pp = mol.pair_params
        pp.zero_()
        unmatched = []
        for ind, name in enumerate(atom_names_long):
            entry = param_dict["atoms"].get(name) or param_dict["atoms"].get(_try_int(name))
            if entry is None:
                unmatched.append(name)
                continue
            pp[ind] = torch.tensor([float(charge_of(name)), float(entry["epsilon"]), float(entry["sigma"])],
                                   dtype=pp.dtype, device=pp.device)
        if unmatched:
            print(f"  [pairs][warn] {len(unmatched)} atoms not in param_dict['atoms'], e.g. {unmatched[:12]}")
        qcol = pp[:, 0]
        if not bool(torch.isfinite(qcol).all()):
            qcol[~torch.isfinite(qcol)] = 0.0
        total_q = qcol.sum()
        if not torch.isfinite(total_q):
            raise ValueError(f"Initial charge for molecule {i} is NaN or infinite.")
        if i not in sic_component_indices and abs(float(total_q)) > 1e-4:
            print(f"  Neutralizing polymer {i} (net charge {float(total_q):.6f})")
            pp[:, 0] -= total_q / pp.shape[0]

        # Bonds
        charge_bonds = {}
        for val in param_dict["bonds"].values():
            kf = tuple(_rq(charges_dict.get(_try_int(x), 0.0)) for x in _coerce(val["other_idx"], atom_names_long[0]))
            charge_bonds[kf] = charge_bonds[kf[::-1]] = [val["k"], val["length"]]
        by_name = by_charge = 0
        for ind, (a, b) in enumerate(mol.bonds):
            names = [atom_names_long[a], atom_names_long[b]]
            hit = None
            for val in param_dict["bonds"].values():
                if set(_coerce(val["other_idx"], names[0])) == set(names):
                    hit = [val["k"], val["length"]]
                    by_name += 1
                    break
            if hit is None:
                c1 = tuple(_rq(charges_dict.get(_try_int(x), 0.0)) for x in names)
                hit = charge_bonds.get(c1) or charge_bonds.get(c1[::-1])
                if hit is None:
                    raise KeyError(f"No bond parameters by name or charge for atom types {names} (charges {c1}).")
                by_charge += 1
            mol.bond_params[ind] = torch.tensor(hit, dtype=mol.bond_params.dtype, device=mol.bond_params.device)
        print(f"  bonds: {len(mol.bonds)} (by type {by_name}, by charge {by_charge})")

        # Angles
        charge_angles = {}
        for val in param_dict["angles"].values():
            kf = tuple(_rq(charges_dict.get(_try_int(x), 0.0)) for x in _coerce(val["other_idx"], atom_names_long[0]))
            charge_angles[kf] = charge_angles[kf[::-1]] = [val["k"], val["angle"]]
        bonded_pairs = set()
        for a, b in mol.bonds:
            bonded_pairs.add((int(a), int(b)))
            bonded_pairs.add((int(b), int(a)))
        by_name = by_charge = 0
        keep_angles = []
        for ind, (a, b, c) in enumerate(mol.angles):
            names = [atom_names_long[a], atom_names_long[b], atom_names_long[c]]
            hit = None
            for val in param_dict["angles"].values():
                oi = _coerce(val["other_idx"], names[0])
                if oi == names or oi == names[::-1]:
                    hit = [val["k"], val["angle"]]
                    by_name += 1
                    break
            if hit is None:
                c1 = tuple(_rq(charges_dict.get(_try_int(x), 0.0)) for x in names)
                hit = charge_angles.get(c1) or charge_angles.get(c1[::-1])
                if hit is None:
                    if (int(a), int(c)) in bonded_pairs:
                        # The outer atoms are bonded, so this angle closes a 3-membered
                        # ring. Its three bond lengths already fix all three interior
                        # angles, which is why LigParGen emits no parameter for it.
                        # Substituting a generic ~110 deg angle would tear the ring open.
                        continue
                    raise KeyError(f"No angle parameters by name or charge for atom types {names} (charges {c1}).")
                by_charge += 1
            keep_angles.append(ind)
            mol.angle_params[ind] = torch.tensor(hit, dtype=mol.angle_params.dtype, device=mol.angle_params.device)
        n_ring = len(mol.angles) - len(keep_angles)
        if n_ring:
            print(f"  dropped {n_ring} redundant 3-membered-ring angles")
            mol.angles = mol.angles[keep_angles]
            mol.angle_params = mol.angle_params[keep_angles]
        print(f"  angles: {len(mol.angles)} (by type {by_name}, by charge {by_charge})")

        # Dihedrals: exact atom-type IDs, then the central bond. param_dict keys atoms by
        # hitpoly type ID, so the names already are the types; a central-bond match covers
        # ring torsions whose end atoms are typed differently in the long chain than in
        # any oligomer.
        name_index, central_index = {}, {}
        for val in param_dict["dihedrals"].values():
            oi = tuple(_coerce(val["other_idx"], atom_names_long[0]))
            name_index.setdefault(oi, []).append(val)
            name_index.setdefault(oi[::-1], []).append(val)
            central_index.setdefault((oi[1], oi[2]), []).append(val)
            central_index.setdefault((oi[2], oi[1]), []).append(val)
        counts = Counter()
        for ind, quad in enumerate(mol.dihedrals):
            names4 = tuple(atom_names_long[x] for x in quad)
            cand = name_index.get(names4) or name_index.get(names4[::-1])
            how = "name"
            if not cand:
                cand, how = central_index.get((names4[1], names4[2])), "central"
            if not cand:
                # No match found through name, OPLS type, or central-bond lookup.
                # This happens at copolymer junctions where the partial-charge
                # context of the long chain differs slightly from all oligomers,
                # causing a different OPLS type assignment. A zero torsional
                # barrier is the standard OPLS-AA treatment for unparameterised
                # generic aliphatic/ether dihedrals.
                print(f"  WARNING: No dihedral for atom types {names4}, using k=0.")
                counts["zero"] += 1
                continue
            counts[how] += 1
            mol.dihedral_params[ind] = torch.tensor(cand[0]["k"], dtype=mol.dihedral_params.dtype,
                                                    device=mol.dihedral_params.device)
        print(f"  dihedrals: {len(mol.dihedrals)} {dict(counts)}")

        # Impropers: keep only the ones with parameters
        keep = []
        for ind, quad in enumerate(mol.impropers):
            names4 = [atom_names_long[x] for x in quad]
            for val in param_dict["impropers"].values():
                oi = _coerce(val["other_idx"], names4[0])
                if oi == names4 or oi == names4[::-1] or (oi[0] == names4[0] and np.isin(oi[1:], names4[1:]).all()):
                    mol.improper_params[ind] = torch.tensor(val["k"], dtype=mol.improper_params.dtype,
                                                            device=mol.improper_params.device)
                    keep.append(ind)
                    break
        mol.impropers = mol.impropers[keep]
        mol.improper_params = mol.improper_params[keep]

    print("Species charges after assignment:")
    for i in range(len(train_dataset)):
        ds = train_dataset[i]
        print(f"  idx={i:>2} atoms={len(ds.atomic_nums):>4} charge={float(ds.pair_params[:, 0].sum()):+.6f} {ds.smiles[0][:60]}")
    return train_dataset


def load_pdb_create_mol_dict(save_path, train_dataset, polymer=False, packed_name=None):
    """
    Match the packed box's PDB chains to train_dataset entries: polymers by order (the last
    two entries are always the cation and TFSI), salts by element composition.
    """
    if packed_name is None:
        packed_name = "polymer_conformation" if polymer else "packed_box"
    pdb_file = PDBFile(f"{save_path}/{packed_name}.pdb")
    mol_dict = {}

    # Cation and TFSI stay in train_dataset even for SIC systems, where TFSI is not in the
    # box; it simply finds no matching chain.
    n_salt = 2
    polymer_train_data = train_dataset[:-n_salt]
    salt_train_data = train_dataset[-n_salt:]
    pdb_chains = list(pdb_file.topology.chains())

    for idx, chain in enumerate(pdb_chains[: len(polymer_train_data)]):
        mol = polymer_train_data[idx]
        mol_dict[mol.smiles[0]] = {"mol_ffnet": mol, "mol_pdb": chain}
        pdb_Z = [a.element.atomic_number for a in chain._residues[0].atoms()]
        if pdb_Z != list(mol.atomic_nums):
            raise ValueError(f"PDB chain {idx} atom order does not match its SMILES {mol.smiles[0][:60]}")

    def _sig(atomic_nums):
        return tuple(sorted(Counter(atomic_nums).items()))

    for salt_chain in pdb_chains[-n_salt:]:
        sig = _sig([a.element.atomic_number for a in salt_chain._residues[0].atoms()])
        for salt_mol in salt_train_data:
            if _sig(list(salt_mol.atomic_nums)) == sig:
                mol_dict[salt_mol.smiles[0]] = {"mol_ffnet": salt_mol, "mol_pdb": salt_chain}
                break

    if not mol_dict:
        raise ValueError("No molecules were matched between RDKit and PDB.")
    print(f"Matched {len(mol_dict)} species to {len(pdb_chains)} PDB chains.")
    return mol_dict, pdb_file


def atom_name_reindexing(mol_dict):
    """Give every atom in every species a unique name (element + suffix) for the FF XML."""
    digits, letters = STRING.digits, STRING.ascii_uppercase
    index_list = [str(i) for i in range(1, 10)]
    index_list += ["".join(i) for i in itertools.product(digits, repeat=2)]
    index_list += ["".join(i) for i in itertools.product(letters, digits)]
    index_list += ["".join(i) for i in itertools.product(letters, repeat=2)]
    index_list += ["".join(i) for i in itertools.product(letters, repeat=3)]
    index_list += ["".join(i) for i in itertools.product(letters, digits, letters)]
    index_list += ["".join(i) for i in itertools.product(digits, repeat=3)]

    def expand_index_list():
        for length in range(4, 10):
            for combo in itertools.product(letters + digits, repeat=length):
                yield "".join(combo)

    atom_types_list = []  # [name, class, element, mass]
    atom_dict = {}
    for mol in mol_dict.values():
        for ind, atom in enumerate(mol["mol_pdb"]._residues[0]._atoms):
            symbol = atom.element.symbol
            atom_name = atom.name[len(symbol):]
            if symbol not in atom_dict:
                atom_dict[symbol] = atom_name
            else:
                prev_atom_ind = index_list.index(atom_dict[symbol])
                try:
                    if index_list.index(atom.name[len(symbol):]) <= prev_atom_ind:
                        if prev_atom_ind + 1 < len(index_list):
                            atom_name = index_list[prev_atom_ind + 1]
                        else:
                            atom_name = next(expand_index_list())
                            index_list.append(atom_name)
                        for res in mol["mol_pdb"]._residues:
                            res._atoms[ind].name = symbol + atom_name
                    atom_dict[symbol] = atom_name
                except ValueError:
                    atom_name = index_list[0]
                    atom_dict[symbol] = atom_name
            atom_types_list.append([atom.name, atom.name, symbol, atom.element.mass._value])
    print(f"Renamed atoms: {len(atom_types_list)} atom types")
    return atom_dict, atom_types_list


def _residue_names(mol_dict, index=None):
    names, ani, cat, ply = [], 1, 1, 1
    for key in mol_dict:
        charge = key.count("+") - key.count("-")
        if charge == -1:
            names.append(f"AN{ani}")
            ani += 1
        elif charge == 1:
            names.append(f"CA{cat}")
            cat += 1
        elif index is None:
            names.append(f"P{ply}")
            ply += 1
        else:
            names.append(f"P{index + 1}")
    return names


def creating_ff_and_resid_files(mol_dict, atom_types_list, index=None):
    """Build the OpenMM force-field XML and residue XML as lists of lines."""
    name_iterables = _residue_names(mol_dict, index)

    ff = ["<ForceField>", "<AtomTypes>"]
    for a_name, a_class, a_elem, a_mass in atom_types_list:
        ff.append(f'<Type name="{a_name}" class="{a_class}" element="{a_elem}" mass="{a_mass}" />')
    ff += ["</AtomTypes>", "<Residues>"]

    for n, mol in zip(name_iterables, mol_dict.values()):
        atoms = mol["mol_pdb"]._residues[0]._atoms
        ff.append(f'<Residue name="{n}">')
        for atom_pdb in atoms:
            ff.append(f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />')
        for bond in mol["mol_ffnet"].bonds.numpy():
            if bond[0] < len(atoms) and bond[1] < len(atoms):
                ff.append(f'<Bond from="{bond[0]}" to="{bond[1]}" />')
        ff.append("</Residue>")
    ff.append("</Residues>")

    ff.append("<HarmonicBondForce>")
    for mol in mol_dict.values():
        atoms = mol["mol_pdb"]._residues[0]._atoms
        for bond, p in zip(mol["mol_ffnet"].bonds.detach().numpy(), mol["mol_ffnet"].bond_params_openmm.detach().numpy()):
            if bond[0] < len(atoms) and bond[1] < len(atoms):
                ff.append(f'<Bond k="{p[0]}" length="{p[1]}" class1="{atoms[bond[0]].name}" class2="{atoms[bond[1]].name}" />')
    ff.append("</HarmonicBondForce>")

    ff.append("<HarmonicAngleForce>")
    for mol in mol_dict.values():
        atoms = mol["mol_pdb"]._residues[0]._atoms
        for ang, p in zip(mol["mol_ffnet"].angles.detach().numpy(), mol["mol_ffnet"].angle_params_openmm.detach().numpy()):
            if all(x < len(atoms) for x in ang[:3]):
                ff.append(f'<Angle k="{p[0]}" angle="{p[1]}" '
                          f'class1="{atoms[ang[0]].name}" class2="{atoms[ang[1]].name}" class3="{atoms[ang[2]].name}" />')
    ff.append("</HarmonicAngleForce>")

    ff.append("<PeriodicTorsionForce>")
    for mol in mol_dict.values():
        atoms = mol["mol_pdb"]._residues[0]._atoms
        for dih, p in zip(mol["mol_ffnet"].dihedrals.detach().numpy(), mol["mol_ffnet"].dihedral_params_openmm.detach().numpy()):
            if all(x < len(atoms) for x in dih[:4]):
                ff.append(
                    f'<Proper k1="{p[0]}" k2="{p[1]}" k3="{p[2]}" k4="{p[3]}" '
                    'periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" '
                    'phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793" '
                    f'class1="{atoms[dih[0]].name}" class2="{atoms[dih[1]].name}" '
                    f'class3="{atoms[dih[2]].name}" class4="{atoms[dih[3]].name}" />'
                )
    ff.append("</PeriodicTorsionForce>")

    ff.append('<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">')
    for mol in mol_dict.values():
        for atom_pdb, p in zip(mol["mol_pdb"]._residues[0]._atoms, mol["mol_ffnet"].pair_params_openmm.detach().numpy()):
            ff.append(f'<Atom charge="{p[0]}" epsilon="{p[1]}" sigma="{p[2]}" type="{atom_pdb.name}" />')
    ff += ["</NonbondedForce>", "</ForceField>"]

    resid = ["<Residues>"]
    for n, mol in zip(name_iterables, mol_dict.values()):
        atoms = mol["mol_pdb"]._residues[0]._atoms
        resid.append(f'<Residue name="{n}">')
        for atom_pdb in atoms:
            resid.append(f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />')
        for bond in mol["mol_ffnet"].bonds.numpy():
            resid.append(f'<Bond from="{atoms[bond[0]].name}" to="{atoms[bond[1]].name}" />')
        resid.append("</Residue>")
    resid.append("</Residues>")

    return ff, resid, name_iterables


def write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables,
                       polymer=False, packed_name=None, index=None):
    """Write force_field(.xml) files and rewrite the PDB with residue names and no CONECT records."""
    if packed_name is None:
        packed_name = "polymer_conformation" if polymer else "packed_box"
    suffix = "" if index is None else f"_{index}"
    with open(f"{save_path}/force_field{suffix}.xml", "w") as f:
        f.write("\n".join(ff_file) + "\n")
    with open(f"{save_path}/force_field_resids{suffix}.xml", "w") as f:
        f.write("\n".join(ff_file_resid) + "\n")
    if index is not None:
        name_iterables = [f"P{index + 1}"]

    for chain, name in zip(pdb_file.topology._chains, name_iterables):
        for residue in chain._residues:
            residue.name = name

    pdb_path = f"{save_path}/{packed_name}.pdb"
    with open(pdb_path, "w") as f:
        pdb_file.writeFile(topology=pdb_file.topology, positions=pdb_file.positions, file=f)
    with open(pdb_path) as f:
        lines = [line for line in f if "CONECT" not in line]
    with open(pdb_path, "w") as f:
        f.writelines(lines)


def create_box_and_ff_files_openmm(
    save_path,
    long_smiles,
    short_smiles,
    filename,
    polymer_count,
    concentration,
    packmol_path,
    atoms_short,
    atoms_long,
    atom_names_short,
    atom_names_long,
    param_dict,
    charge_scale,
    salt_smiles,
    salt_paths,
    salt_data_paths,
    box_multiplier,
    salt=True,
    single_ion_conductor=False,
    sic_component_indices=None,
    neighborhood_depth=2,
):
    """
    PACKMOL box -> typing -> parameter assignment -> OpenMM files.
    polymer_count: one entry per chain file (`filename`), i.e. per entry of long_smiles.
    """
    poly_paths = [f"{save_path}/{name}" for name in filename]
    create_packmol_input_file(
        poly_paths, salt_paths,
        f"{save_path}/packmol.inp", f"{save_path}/packed_box.pdb",
        salt_smiles=salt_smiles,
        polymer_count=polymer_count,
        salt_concentrations=concentration,
        box_multiplier=box_multiplier,
        tolerance=5.0,
        salt=salt,
    )
    run_packmol(save_path, packmol_path)

    if salt and len(concentration) != len(salt_smiles):
        raise ValueError("concentration length must match number of salt_smiles")

    train_dataset = load_hitpoly_params(
        long_smiles, salt_smiles,
        salt_data_paths=salt_data_paths,
        charge_scale=charge_scale,
        poly_count=len(set(long_smiles)),
        neighborhood_depth=neighborhood_depth,
    )
    train_dataset = assign_lpg_params(
        short_smiles_list=short_smiles,
        atoms_long_list=atoms_long,
        atom_names_long_list=atom_names_long,
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        train_dataset=train_dataset,
        param_dict=param_dict,
        sic_component_indices=sic_component_indices,
    )

    mol_dict, pdb_file = load_pdb_create_mol_dict(save_path, train_dataset)
    _, atom_types_list = atom_name_reindexing(mol_dict)

    # LAMMPS -> OpenMM unit scaling applies to the salt only (cation, plus TFSI unless SIC);
    # polymer parameters from LigParGen are already in OpenMM units.
    n_to_scale = 1 if single_ion_conductor else 2
    for idx, key in enumerate(mol_dict):
        mol_dict[key]["mol_ffnet"] = param_scaling_openmm(
            mol_dict[key]["mol_ffnet"], scaling=idx >= len(mol_dict) - n_to_scale
        )

    ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(mol_dict, atom_types_list)
    write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables)
    print("Force-field files written.")
