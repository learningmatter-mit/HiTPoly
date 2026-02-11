import os
import sys

sys.setrecursionlimit(5000)
import time
import math
import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
import string as STRING
import itertools
from openbabel import pybel
import subprocess
from typing import List, Tuple

from rdkit import Chem
from rdkit.Geometry import Point3D

from hitpoly.utils.args import hitpolyArgs
from hitpoly.data.builder import TopologyBuilder, TopologyDataset
from hitpoly.data.ff_file_reader import ForceFieldFileReader
from hitpoly.utils.constants import NUM_TO_ELEMENT, ELEMENT_TO_NUM
from hitpoly.simulations.openmm_scripts import equilibrate_polymer

from openmm.app import PDBFile, ForceField


def index_of(input, source):
    source, sorted_index, inverse = np.unique(
        source.tolist(), return_index=True, return_inverse=True, axis=0
    )
    index = torch.cat([torch.tensor(source), input]).unique(
        sorted=True, return_inverse=True, dim=0
    )[1][-len(input) :]
    try:
        index = torch.tensor(sorted_index)[index]
    except:
        print("error in one-hot encoding")
        import IPython

        IPython.embed()
    return index


def create_long_smiles(
    smile,
    repeats=None,
    req_length=None,
    mw=None,
    add_end_Cs=True,  # default is to add Cs
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
):
    # check if smile is a polymer
    if "Cu" in smile:
        #         calculate required repeats so smiles > 30 atoms long
        if repeats:
            repeats -= 1
        elif req_length:
            num_heavy = Chem.Lipinski.HeavyAtomCount(Chem.MolFromSmiles(smile)) - 2
            repeats = math.ceil(req_length / num_heavy) - 1
        elif mw:
            temp_smiles = smile.replace("[Cu]", "").replace("[Au]", "")
            repeats = math.ceil(
                mw
                / (Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(temp_smiles)) / 1000)
            )
        if repeats > 0:
            try:
                # code to increase length of monomer
                mol = Chem.MolFromSmiles(smile)
                new_mol = mol

                # join repeats number of monomers into polymer
                for i in range(repeats):
                    # join two polymers together at Cu and Au sites
                    rxn = Chem.AllChem.ReactionFromSmarts(reaction)
                    results = rxn.RunReactants((mol, new_mol))
                    assert len(results) == 1 and len(results[0]) == 1, smile
                    new_mol = results[product_index][0]

                new_smile = Chem.MolToSmiles(new_mol)

            except:
                # make smile none if reaction fails
                return "None"

        # if monomer already long enough use 1 monomer unit
        else:
            new_smile = smile

        # caps ends of polymers with carbons
        if add_end_Cs:
            new_smile = (
                new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "")
            )
        else:
            new_smile = (
                new_smile.replace("[Cu]", "").replace("[Au]", "").replace("[Ca]", "")
            )
    else:
        new_smile = smile
        repeats = 0

    # make sure new smile in cannonical
    long_smile = Chem.MolToSmiles(Chem.MolFromSmiles(new_smile))
    return long_smile, repeats + 1


def get_mol_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Sum up the atomic masses of all atoms in the molecule
    mass = 0
    for atom in mol.GetAtoms():
        mass += atom.GetMass()

    return mass


def get_atom_count(
    poly_smiles,
    salt_smiles=[],
    poly_count=1,
    salt_count=1,
):
    num_atoms = 0

    mol = Chem.MolFromSmiles(poly_smiles)
    mol = Chem.AddHs(mol)
    num_atoms += mol.GetNumAtoms() * poly_count
    for i in salt_smiles:
        mol = Chem.MolFromSmiles(i)
        mol = Chem.AddHs(mol)
        num_atoms += mol.GetNumAtoms() * salt_count
    return num_atoms


def create_ligpargen(
    smiles,
    repeats,
    add_end_Cs,
    ligpargen_path,
    hitpoly_path,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    platform="local",
    resid_name=None,
):
    smiles_initial, repeats = create_long_smiles(
        smile=smiles,
        repeats=repeats,
        add_end_Cs=add_end_Cs,
        reaction=reaction,
        product_index=product_index,
    )

    mol_initial = Chem.MolFromSmiles(smiles_initial)
    mol_initial = Chem.AddHs(mol_initial, addCoords=True)

    initial_confs = Chem.AllChem.EmbedMultipleConfs(
        mol_initial,
        numConfs=1,
        maxAttempts=1000,
        boxSizeMult=5,
        useRandomCoords=False,
    )
    if len(initial_confs) == 0:
        initial_confs = Chem.AllChem.EmbedMultipleConfs(
            mol_initial,
            numConfs=1,
            maxAttempts=20000,
            boxSizeMult=5,
            useRandomCoords=True,
        )

    X = mol_initial.GetConformer(0).GetPositions()
    a = euclidean_distances(X, X)
    np.fill_diagonal(a, np.inf)
    if a.min() < 0.7:
        initial_confs = Chem.AllChem.EmbedMultipleConfs(
            mol_initial,
            numConfs=1,
            maxAttempts=20000,
            boxSizeMult=5,
            useRandomCoords=True,
        )

    print(
        Chem.MolToMolBlock(mol_initial), file=open(f"{ligpargen_path}/poly.mol", "w+")
    )
    if not resid_name:
        resid_name = "PLY"

    if platform == "local":
        os.chdir(ligpargen_path)
        command = f"$LigParGen -m poly.mol -o 0 -c 0 -r {resid_name} -d . -l"
        subprocess.run(command, shell=True)
        os.chdir(hitpoly_path)
    elif platform == "supercloud":
        supercloud_ligpargen(ligpargen_path, resid_name)
    elif platform == "perlmutter":
        perlmutter_ligpargen(ligpargen_path, mol_filename, output_prefix)

    return mol_initial, smiles_initial

def perlmutter_ligpargen(ligpargen_path, mol_filename, output_prefix):
    # 1. Check environment variable
    ligpargen = os.environ.get("LigParGen")
    print(f"LigParGen path: {ligpargen}")  # Debugging line

    if ligpargen is None:
        print("ERROR: Environment variable LigParGen is not set!")
        return

    print(f"Molecule filename: {mol_filename}")  # Debugging line
    print(f"Output prefix: {output_prefix}")  # Debugging line

    # 2. Build the full path to the run script
    run_script_path = os.path.join(ligpargen_path, "run.sh")

    # 3. Write the shell script (remove SBATCH lines)
    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("# Load modules\n")
        f.write("source $HOME/.bashrc\n")
        f.write("conda activate htvs\n")
        
        #Replace with your path
        f.write("export PYTHONPATH=/global/homes/t/trios/htvs/ligpargen\n")
        
        f.write("cwd=$(pwd)\n")
        f.write(f"cd {ligpargen_path}\n")
        f.write(f"{ligpargen} -m {mol_filename} -o 0 -c 0 -r {output_prefix} -d . -l\n")
        f.write("cd $cwd\n")

    # Make sure the script is executable
    os.chmod(run_script_path, 0o755)

    # 4. Confirm the file exists
    if not os.path.exists(run_script_path):
        print("ERROR: run.sh script was not created!")
        return

    # 5. Print script for debugging
    print("\n--- Contents of run.sh ---")
    with open(run_script_path) as f:
        print(f.read())
    print("--- End of run.sh ---\n")

    # 6. Run the script directly via bash
    print("Running run.sh directly via bash...")

    result = subprocess.run(
        f"bash {run_script_path}",
        shell=True,
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout.strip())
    print("STDERR:", result.stderr.strip())

    # 7. Check for errors in execution
    if result.returncode != 0:
        print("ERROR: run.sh execution failed!")
        return

    # 8. Wait for the expected output
    expected_output_file = os.path.join(ligpargen_path, f"{output_prefix}.xml")
    t0 = time.time()
    while True:
        if os.path.exists(expected_output_file):
            time.sleep(2)
            print(f"Output file {expected_output_file} found.")
            break
        elif time.time() - t0 > 300:
            print(f"Timeout: {expected_output_file} not found within the time limit.")
            break
        else:
            time.sleep(10)

def supercloud_ligpargen(ligpargen_path, resid_name, ligpargen_version="2.0"):
    if ligpargen_version == "2.0":
        with open(f"{ligpargen_path}/run.sh", "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write("#SBATCH --job-name=ligpargen" + "\n")
            f.write("#SBATCH --partition=xeon-p8" + "\n")
            f.write("#SBATCH --nodes=1" + "\n")
            f.write("#SBATCH --ntasks-per-node=1" + "\n")
            f.write("#SBATCH --cpus-per-task=1" + "\n")
            f.write("#SBATCH --time=2:00:00" + "\n")
            f.write("\n")
            f.write("cwd=$(pwd)" + "\n")
            f.write(f"cd {ligpargen_path}" + "\n")
            f.write(f"export XDG_DATA_HOME=/state/partition1/user/$(id -un)" + "\n")
            f.write(f"podman load -i $HOME/containers/ligpargen.tar" + "\n")
            f.write(f"podman run --rm -w /app/RUN -v .:/app/RUN:Z -v .:/app/RUN:Z ligpargen:latest python ../LigParGen/Converter.py -m poly.mol -o 0 -c 0 -r {resid_name} -d . -l" + "\n")
            f.write("cd $cwd" + "\n")
        command = f"sbatch {ligpargen_path}/run.sh"
        subprocess.run(command, shell=True)
        t0 = time.time()
        while True:
            if os.path.exists("PLY.xml"):
                time.sleep(2)
                break
            elif time.time() - t0 > 120:
                break
            else:
                time.sleep(10)
    elif ligpargen_version == "2.1":
        raise NotImplementedError("LigParGen 2.1 is not implemented yet.")



def create_ligpargen_smiles(
    smiles,
    repeats,
    add_end_Cs,
    reaction,
    product_index,
):
    smiles_initial, repeats = create_long_smiles(
        smile=smiles,
        repeats=repeats,
        add_end_Cs=add_end_Cs,
        reaction=reaction,
        product_index=product_index,
    )

    mol_initial = Chem.MolFromSmiles(smiles_initial)
    mol_initial = Chem.AddHs(mol_initial, addCoords=True)

    return mol_initial, smiles_initial


def create_ligpargen_short_polymer(
    smiles,
    add_end_Cs,
    reaction,
    product_index,
    atom_names_long,
):
    atom_names_long_sorted = sorted(set(atom_names_long))

    for ligpargen_repeats in range(30, 1, -1):
        mol_initial, smiles_initial = create_ligpargen_smiles(
            smiles=smiles,
            repeats=ligpargen_repeats,
            add_end_Cs=add_end_Cs,
            reaction=reaction,
            product_index=product_index,
        )

        r, atom_names, atoms, bonds_typed = generate_atom_types(mol_initial, 2)

        if len(atom_names) < 200 and sorted(set(atom_names)) == atom_names_long_sorted:
            break
        else:
            continue

    return (
        ligpargen_repeats,
        smiles_initial,
        mol_initial,
        r,
        atom_names,
        atoms,
        bonds_typed,
    )


def create_conformer_pdb(
    path_to_save,
    smiles,
    name=None,
    enforce_generation=False,
):
    if not name:
        name = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol, addCoords=True)

    # Figure this out to make 'straighter' polymer chains
    minimize = True
    conf = False
    opt_steps = 1
    while not conf:
        pybel_mol = pybel.readstring("smi", smiles)
        pybel_mol.make3D(steps=opt_steps)
        pybel_mol.write("pdb", f"{path_to_save}/{name}", overwrite=True)
        try:
            print(f"Attempting to generate pybel conf with {opt_steps} steps")
            temp_mol = Chem.MolFromPDBFile(
                f"{path_to_save}/{name}", removeHs=False
            )
            # checking minimal distance between atoms
            X = temp_mol.GetConformer(0).GetPositions()
            a = euclidean_distances(X, X)
            np.fill_diagonal(a, np.inf)
            if a.min() > 0.5:
                conf = True
                minimize = False

                Chem.rdDepictor.Compute2DCoords(mol)
                conformation = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    # Checking if the atoms are the same acroos the whole molecule
                    #  if not, then everything fails
                    assert (
                        pybel_mol.atoms[i].atomicnum == mol.GetAtoms()[i].GetAtomicNum()
                    )
                    x, y, z = (
                        pybel_mol.atoms[i].coords[0],
                        pybel_mol.atoms[i].coords[1],
                        pybel_mol.atoms[i].coords[2],
                    )

                    conformation.SetAtomPosition(i, Point3D(x, y, z))

                writer = Chem.PDBWriter(f"{path_to_save}/{name}")
                writer.write(mol, confId=mol.GetConformer().GetId())
                writer.close()
                print(
                    f"Successfully generated pybel 3D conformer with {opt_steps} steps"
                )
            else:
                raise ValueError(f"Failed with {opt_steps}, increasing to {opt_steps*10}")
        except:
            print(f"Failed with {opt_steps}, increasing to {opt_steps*10}")
            opt_steps *= 10

        if opt_steps == 10 and not conf:
            print("Attempting to generate conf with rdkit 2D geom generation")
            initial_confs = Chem.rdDepictor.Compute2DCoords(mol)
            writer = Chem.PDBWriter(f"{path_to_save}/{name}")
            writer.write(mol, confId=mol.GetConformer().GetId())
            writer.close()
            temp_mol = Chem.MolFromPDBFile(
                f"{path_to_save}/{name}", removeHs=False
            )
            # checking minimal distance between atoms
            try:
                X = temp_mol.GetConformer(0).GetPositions()
                a = euclidean_distances(X, X)
                np.fill_diagonal(a, np.inf)
                if a.min() > 0.5:
                    conf = True
                    print("Successfully generated rdkit 2D conformer")
            except:
                continue

        if opt_steps > 100:
            # Doing a BFS over the 2D molecule to manually move all the atoms around the z axis
            initial_confs = Chem.rdDepictor.Compute2DCoords(mol)
            neighbors = [[x.GetIdx() for x in y.GetNeighbors()] for y in mol.GetAtoms()]
            neigh_dict = {}
            for ind, i in enumerate(neighbors):
                neigh_dict[ind] = i
            visited = []  # List for visited nodes.
            queue = []  # Initialize a queue
            move_dist = 0
            conformation = mol.GetConformer()

            def bfs(visited, graph, node, move_dist):  # function for BFS
                visited.append(node)
                queue.append(node)

                while queue:  # Creating loop to visit each node
                    m = queue.pop(0)
                    for neighbour in graph[m]:
                        if neighbour not in visited:
                            visited.append(neighbour)
                            queue.append(neighbour)
                            x, y, z = (
                                mol.GetConformer().GetPositions()[neighbour][0],
                                mol.GetConformer().GetPositions()[neighbour][1],
                                mol.GetConformer().GetPositions()[neighbour][2],
                            )
                            conformation.SetAtomPosition(
                                neighbour, Point3D(x, y, z + move_dist)
                            )
                            move_dist += 0.1

            bfs(visited, neigh_dict, 0, move_dist)
            writer = Chem.PDBWriter(f"{path_to_save}/{name}")
            writer.write(mol, confId=mol.GetConformer().GetId())
            writer.close()
            print(
                "Generated 2D conf and moved atoms around the z-axis, sometimes works, sometimes not"
            )
            conf = True

    return minimize


def create_packmol_input_file(
    poly_paths: list,
    salt_paths: list,
    output_path: str,
    output_name: str,
    salt_smiles: list = None,
    polymer_count: list = [25],
    salt_concentrations: list = [100, 100],
    tolerance: float = 10.0,
    box_multiplier: int = 2,
    random_seed: int = -1,
    salt: bool = True,
):
    """
    poly_paths: list - a list of pdb filenames for the
        polymer molecules to be packed
    salt_paths: list - a list of pdb filenames for the
        salt molecule to be packed
    output_name: str - path, name of the output file
    polymer_count: float - amount of solvent chains in the box
    salt_concentrations: list - a list of how many of each salt in the box
    tolerance: float - 5.0, packmol packing distance
    random_seed: int - -1, random seed
    salt: bool - True, whether or not to include salt molecules in the pdb
    """
    mol_list = []
    single_vol = 0
    all_paths = poly_paths + salt_paths if salt_paths else poly_paths
    
    for i in all_paths:
        mol = Chem.MolFromPDBFile(i, removeHs=False)
        mol_list.append(mol)
        conf = mol.GetConformer()
        radius = distance_matrix(conf.GetPositions(), conf.GetPositions()).max() / 2
        # The volume of a polymer chain + anion, cation
        single_vol += 4 / 3 * np.pi * radius**3

    if sum(polymer_count) == 0:
        total_vol = single_vol * box_multiplier * salt_concentrations[0]
    else:
        total_vol = single_vol * sum(polymer_count) * box_multiplier
    box_radi = np.power(3 / (4 * np.pi) * total_vol, 1 / 3)

    # If more than one anion, cation, create sublists
    if salt:
        if len(salt_paths) > 2:
            anion_paths = []
            cation_paths = []
            anion_conc = []
            cation_conc = []
            for ind, (i, s) in enumerate(zip(salt_paths, salt_smiles)):
                minuses = s.count("-")
                pluses = s.count("+")
                if pluses - minuses == -1:
                    anion_paths.append(i)
                    if salt_concentrations:
                        anion_conc.append(salt_concentrations[ind])
            for ind, (i, s) in enumerate(zip(salt_paths, salt_smiles)):
                minuses = s.count("-")
                pluses = s.count("+")
                if pluses - minuses == 1:
                    cation_paths.append(i)
                    if salt_concentrations:
                        cation_conc.append(salt_concentrations[ind])
        elif (
            len(salt_paths) == 0
            or salt_concentrations[0] == 0
            or salt_concentrations[1] == 0
        ):
            cation_paths = []
            anion_paths = []
        else:
            # If it's only two, the order actually doens't matter
            cation_paths = [salt_paths[0]]
            anion_paths = [salt_paths[1]]
            cation_conc = [salt_concentrations[0]]
            anion_conc = [salt_concentrations[1]]
    else:
        anion_paths = []
        cation_paths = []

    with open(output_path, "w+") as f:
        f.write(f"tolerance {tolerance}\n")
        f.write(f"filetype pdb\n")
        f.write(f"output {output_name}\n")
        f.write(f"seed {random_seed}\n\n")
        for ind, i in enumerate(poly_paths):
            f.write(f"structure {i}\n")
            f.write(f"  number {polymer_count[ind]}\n")
            f.write(f"  inside box 0 0 0 {box_radi} {box_radi} {box_radi}\n")
            f.write(f"end structure\n\n")
        for ind, i in enumerate(anion_paths):
            f.write(f"structure {i}\n")
            f.write(f"  number {anion_conc[ind]}\n")
            f.write(f"  inside box 0 0 0 {box_radi} {box_radi} {box_radi}\n")
            f.write(f"end structure\n\n")
        for ind, i in enumerate(cation_paths):
            f.write(f"structure {i}\n")
            f.write(f"  number {cation_conc[ind]}\n")
            f.write(f"  inside box 0 0 0 {box_radi} {box_radi} {box_radi}\n")
            f.write(f"end structure\n\n")
    print("Created packmol file")


def run_packmol(save_path, packmol_path):
    print(f"Running packmol at {save_path}")
    command = f"{packmol_path}/packmol < {save_path}/packmol.inp"
    process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if process.returncode != 0:
        print(f"Error executing packmol")
        print(f"STDOUT: {process.stdout}")
        print(f"STDERR: {process.stderr}")
    else:
        print("Run packmol successfully.")
        print(f"STDOUT: {process.stdout}")


def load_hitpoly_params(
    poly_smiles: list,
    salt_smiles: list,
    salt_data_paths: list = [],
    charge_scale: float = None,
):
    smiles = poly_smiles + salt_smiles
    train_args = hitpolyArgs()

    train_molecule_data = [
        TopologyBuilder(
            smiles=[i], train_args=train_args, load_geoms=False, lj_from_file=False
        )
        for i in smiles
    ]

    train_dataset = TopologyDataset(data=train_molecule_data)

    for i, name in zip(train_dataset[len(poly_smiles):], salt_data_paths):
        ff_read = ForceFieldFileReader(
            i,
            name,
        )
        ff_read.load_pair_params()
        ff_read.load_bond_params()
        ff_read.load_angle_params()
        ff_read.load_dihedral_params()

    if charge_scale:
        for i in train_dataset[len(poly_smiles):]:
            i.pair_params[:, 0] = i.pair_params[:, 0] * charge_scale

    for i in train_dataset:
        print(f"Charges for {i.smiles[0]} - ", i.pair_params[:, 0].sum())

    return train_dataset


def load_pdb_create_mol_dict(
    save_path,
    train_dataset,
    polymer=False,
):
    if polymer:
        packed_name = polymer
    else:
        packed_name = "packed_box.pdb"

    pdb_file = PDBFile(f"{save_path}/{packed_name}")
    mol_dict = {}
    for mol in pdb_file.topology.chains():
        temp_atoms = []
        for atom in mol._residues[0].atoms():
            temp_atoms.append(atom.element.atomic_number)
        for i in train_dataset:
            if np.array_equal(np.array(i.atomic_nums), np.array(temp_atoms)):
                mol_dict[i.smiles[0]] = {}
                mol_dict[i.smiles[0]]["mol_hitpoly"] = i
                mol_dict[i.smiles[0]]["mol_pdb"] = mol
            else:
                ValueError(
                    "The rdkit molecule ordering is different than the one from LigParGen"
                )

    return mol_dict, pdb_file


def smiles_encoder(smiles):
    smiles_chars = list(set(smiles))

    smi2index = dict((c, i) for i, c in enumerate(smiles_chars))
    index2smi = dict((i, c) for i, c in enumerate(smiles_chars))

    X = np.zeros((len(smiles), len(smiles_chars)))
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
    return X


def generate_atom_types(mol, depth):
    neighbors = [[x.GetIdx() for x in y.GetNeighbors()] for y in mol.GetAtoms()]
    atoms = [x.GetSymbol() for x in mol.GetAtoms()]

    bonds = [list(itertools.product(x, [ind])) for ind, x in enumerate(neighbors)]
    bonds = list(itertools.chain(*bonds))
    bonds = torch.LongTensor([list(b) for b in bonds])

    bonds_typed = bonds[bonds[:, 1] > bonds[:, 0]].tolist()

    a = bonds.view(-1, 2)
    d = torch.tensor([len(x) for x in neighbors])
    r = torch.tensor(smiles_encoder(atoms))

    unique_values = {}

    for rad in range(depth + 1):
        if rad != 0:
            # the message is in the direction (atom 1 -> atom 0) for each edge, so the message is the current atom label of atom1
            # the messages from each incoming atom are then split by receiving atom,
            # so the messages that are going into a particular atom are all grouped together
            messages = list(torch.split(r[a[:, 0]], d.tolist()))
            # the messages incoming to each atom are then added together to enforce permutation invariance
            messages = [messages[n].sum(0) for n in range(len(d))]
            messages = torch.stack(messages)

            # the message is then appended to the current state to remember order of messages
            r = torch.cat([r, messages], dim=1)
        if rad not in unique_values.keys():
            unique_values[rad], one_hot_mapping = r.unique(dim=0, return_inverse=True)
        index = index_of(r, unique_values[rad])
        r = torch.eye(len(unique_values[rad])).to(torch.long)[index]
    print("One-hot encoding has the shape", r.unique(dim=1).shape)

    atom_names = []
    for i in torch.nonzero(r):
        atom_names.append(atoms[i[0]] + str(i[1].cpu().numpy()))

    return r, atom_names, atoms, bonds_typed


def generate_parameter_dict(save_path, atom_names, atoms, bonds_typed, file_name = None, resid_name = None):
    if not file_name:
        file_name = "PLY"
    if not resid_name:
        resid_name = "PLY"
    forceFieldFiles = [f"{save_path}/{file_name}.xml"]

    forcefield = ForceField(*forceFieldFiles)

    # Checking if the atoms have the same order and if the bonds are the same!
    atoms_ff = [i.element._symbol for i in forcefield._templates[resid_name].atoms]
    assert atoms_ff == atoms

    bond_list = [list(i) for i in forcefield._templates[resid_name].bonds]
    same_bonds = 0
    for i in bond_list:
        for j in bonds_typed:
            if set(i) == set(j):
                same_bonds += 1
    assert same_bonds == len(bond_list)
    print("LigParGen and RDkit atoms and bonds are the same")

    param_dict = {}
    param_dict["atoms"] = {}
    param_dict["bonds"] = {}
    param_dict["angles"] = {}
    param_dict["dihedrals"] = {}
    param_dict["impropers"] = {}

    for ind, atom in enumerate(forcefield._templates[resid_name].atoms):
        param_dict["atoms"][atom.name] = {}
        param_dict["atoms"][atom.name]["type"] = atom.type

        param_dict["atoms"][atom.name]["other_type"] = atom_names[ind]
        param_dict["atoms"][atom.name]["element"] = atom.element._symbol
        param_dict["atoms"][atom.name]["charge"] = forcefield._forces[
            3
        ].params.paramsForType[atom.type]["charge"]
        param_dict["atoms"][atom.name]["sigma"] = forcefield._forces[
            3
        ].params.paramsForType[atom.type]["sigma"]
        param_dict["atoms"][atom.name]["epsilon"] = forcefield._forces[
            3
        ].params.paramsForType[atom.type]["epsilon"]
    type2atomdict = {}
    atom2typedict = {}
    atom2index = forcefield._templates[resid_name].atomIndices
    for atom in forcefield._templates[resid_name].atoms:
        type2atomdict[atom.type] = atom.name
        atom2typedict[atom.name] = atom.type

    atomff2atom1hotdict = {}
    atom1hotdict2atomff = {}
    for key, val in param_dict["atoms"].items():
        atomff2atom1hotdict[key] = val["other_type"]
        atom1hotdict2atomff[val["other_type"]] = key

    for ind, (t1, t2) in enumerate(
        zip(forcefield._forces[0].types1, forcefield._forces[0].types2)
    ):
        a1 = type2atomdict[list(t1)[0]]
        a2 = type2atomdict[list(t2)[0]]
        b_name = f"bond_{ind}"
        param_dict["bonds"][b_name] = {}
        param_dict["bonds"][b_name]["ff_idx"] = [a1, a2]
        param_dict["bonds"][b_name]["other_idx"] = [
            atomff2atom1hotdict[a1],
            atomff2atom1hotdict[a2],
        ]
        param_dict["bonds"][b_name]["length"] = forcefield._forces[0].length[ind]
        param_dict["bonds"][b_name]["k"] = forcefield._forces[0].k[ind]

    for ind, (t1, t2, t3) in enumerate(
        zip(
            forcefield._forces[1].types1,
            forcefield._forces[1].types2,
            forcefield._forces[1].types3,
        )
    ):
        a1 = type2atomdict[list(t1)[0]]
        a2 = type2atomdict[list(t2)[0]]
        a3 = type2atomdict[list(t3)[0]]
        b_name = f"angle_{ind}"
        param_dict["angles"][b_name] = {}
        param_dict["angles"][b_name]["ff_idx"] = [a1, a2, a3]
        param_dict["angles"][b_name]["other_idx"] = [
            atomff2atom1hotdict[a1],
            atomff2atom1hotdict[a2],
            atomff2atom1hotdict[a3],
        ]
        param_dict["angles"][b_name]["angle"] = forcefield._forces[1].angle[ind]
        param_dict["angles"][b_name]["k"] = forcefield._forces[1].k[ind]

    for ind, dih in enumerate(forcefield._forces[2].proper):
        a1 = type2atomdict[list(dih.types1)[0]]
        a2 = type2atomdict[list(dih.types2)[0]]
        a3 = type2atomdict[list(dih.types3)[0]]
        a4 = type2atomdict[list(dih.types4)[0]]
        b_name = f"dihedral_{ind}"
        param_dict["dihedrals"][b_name] = {}
        param_dict["dihedrals"][b_name]["ff_idx"] = [a1, a2, a3, a4]
        param_dict["dihedrals"][b_name]["other_idx"] = [
            atomff2atom1hotdict[a1],
            atomff2atom1hotdict[a2],
            atomff2atom1hotdict[a3],
            atomff2atom1hotdict[a4],
        ]
        param_dict["dihedrals"][b_name]["periodicity"] = (
            forcefield._forces[2].proper[ind].periodicity
        )
        param_dict["dihedrals"][b_name]["phase"] = (
            forcefield._forces[2].proper[ind].phase
        )
        param_dict["dihedrals"][b_name]["k"] = forcefield._forces[2].proper[ind].k

    for ind, dih in enumerate(forcefield._forces[2].improper):
        a1 = type2atomdict[list(dih.types1)[0]]
        a2 = type2atomdict[list(dih.types2)[0]]
        a3 = type2atomdict[list(dih.types3)[0]]
        a4 = type2atomdict[list(dih.types4)[0]]
        b_name = f"improper_{ind}"
        param_dict["impropers"][b_name] = {}
        param_dict["impropers"][b_name]["ff_idx"] = [a1, a2, a3, a4]
        param_dict["impropers"][b_name]["other_idx"] = [
            atomff2atom1hotdict[a1],
            atomff2atom1hotdict[a2],
            atomff2atom1hotdict[a3],
            atomff2atom1hotdict[a4],
        ]
        param_dict["impropers"][b_name]["periodicity"] = (
            forcefield._forces[2].improper[ind].periodicity
        )
        param_dict["impropers"][b_name]["phase"] = (
            forcefield._forces[2].improper[ind].phase
        )
        param_dict["impropers"][b_name]["k"] = forcefield._forces[2].improper[ind].k
    return param_dict

def assign_lpg_params(
    atoms_long,
    atom_names_long,
    atoms_short,
    atom_names_short,
    train_dataset,
    param_dict,
    lit_charges_save_path,
    polynames,
    charges,
):
    for ind_dataset in range(len(param_dict)):
        test_atoms = [NUM_TO_ELEMENT[i] for i in train_dataset[ind_dataset].atomic_nums]
        assert test_atoms == atoms_long[ind_dataset]
        print(
            "Atoms same between Topology builder (hitpoly) and atom types (ligpargen) atoms"
        )

        for ind, i in enumerate(train_dataset[ind_dataset].pair_params):
            for key, val in param_dict[ind_dataset]["atoms"].items():
                if val["other_type"] == atom_names_long[ind_dataset][ind]:
                    pairs = [val["charge"], val["epsilon"], val["sigma"]]
                    train_dataset[ind_dataset].pair_params[ind] = torch.tensor(pairs)
                    break

        if charges == "LPG":
            charge_dict_temp = {}
            for key, val in param_dict[ind_dataset]["atoms"].items():
                if val["other_type"] not in charge_dict_temp.keys():
                    charge_dict_temp[val["other_type"]] = []
                    charge_dict_temp[val["other_type"]].append(val["charge"])
                else:
                    charge_dict_temp[val["other_type"]].append(val["charge"])

            charges_dict = {}
            for key, val in charge_dict_temp.items():
                charges_dict[key] = np.array(val).mean()
        elif charges == "LIT":
            charges_list = []
            with open(f"{lit_charges_save_path}/LIT_charges/{polynames[ind_dataset]}.csv", "r") as f:
                lines = f.readlines()
                try:
                    for i, a in zip(lines, atoms_short[ind_dataset]):
                        if i.split(",")[1] == a:
                            charges_list.append(float(i.split(",")[0]))
                except:
                    # If the charges file starts with a description
                    print(lines[1:])
                    print(atoms_short[ind_dataset])
                    for i, a in zip(lines[1:], atoms_short[ind_dataset]):
                        if i.split(",")[1] == a:
                            charges_list.append(float(i.split(",")[0]))
            print(len(charges_list), len(atoms_short[ind_dataset]))
            assert len(charges_list) == len(atoms_short[ind_dataset])

            df = pd.DataFrame({"charge": charges_list, "names": atom_names_short[ind_dataset]})

            charges_dict = {}
            for i in df.groupby("names", as_index=False).mean().to_dict("records"):
                charges_dict[i["names"]] = i["charge"]
        else:
            raise ValueError(f"Wrong charge name {charges}")

        for ind, i in enumerate(train_dataset[ind_dataset].pair_params):
            train_dataset[ind_dataset].pair_params[ind, 0] = charges_dict[atom_names_long[ind_dataset][ind]]

        mean = train_dataset[ind_dataset].pair_params[:, 0].mean()
        smallest_charge = train_dataset[ind_dataset].pair_params[:, 0].abs().min()
        sum_char = train_dataset[ind_dataset].pair_params[:, 0].sum()
        print(
            f"Mean of the charges {mean:.4}, sum of the charges {sum_char}, and smallest charge {smallest_charge}"
        )
        if train_dataset[ind_dataset].pair_params[:, 0].sum().abs() * 50 > 1e-3:
            charges_all = train_dataset[ind_dataset].pair_params[:, 0]
            pol_pos = charges_all[charges_all > 0].sum()
            pol_neg = charges_all[charges_all < 0].sum()
            pol_sc = torch.mean(torch.tensor((pol_pos, torch.abs(pol_neg))))
            charges_all[charges_all > 0] = (charges_all[charges_all > 0] / pol_pos) * pol_sc
            charges_all[charges_all < 0] = (
                charges_all[charges_all < 0] / torch.abs(pol_neg)
            ) * pol_sc
            train_dataset[ind_dataset].pair_params[:, 0] = charges_all
            mean = train_dataset[ind_dataset].pair_params[:, 0].mean()
            smallest_charge = train_dataset[ind_dataset].pair_params[:, 0].abs().min()
            sum_char = train_dataset[ind_dataset].pair_params[:, 0].sum()
            print("Charges have been rescaled")
            print(
                f"Mean of the charges {mean:.4}, sum of the charges {sum_char}, and smallest charge {smallest_charge}"
            )
        if train_dataset[ind_dataset].pair_params[:, 0].sum().abs() * 10 > 1e-3:
            raise ValueError("CHARGES HAVE NOT BEEN PROPERLY RESCALAED")

        # Sometimes not all bonds are assinged, this creates a bond dict fromt the unique atom
        #  types, based on the charges
        charge_bonds = {}
        for key, val in param_dict[ind_dataset]["bonds"].items():
            charge_bonds[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = [val["k"], val["length"]]

        bonds_added = 0
        for ind, i in enumerate(train_dataset[ind_dataset].bonds):
            bonds_temp = [atom_names_long[ind_dataset][i[0]], atom_names_long[ind_dataset][i[1]]]
            bond_added = False
            for key, val in param_dict[ind_dataset]["bonds"].items():
                if set(val["other_idx"]) == set(bonds_temp):
                    bond_p = [val["k"], val["length"]]
                    train_dataset[ind_dataset].bond_params[ind] = torch.tensor(bond_p)
                    bond_added = True
                    bonds_added += 1
                    break
            if not bond_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in bonds_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in bonds_temp[::-1]]
                )
                if charge_temp1 in charge_bonds.keys():
                    bond_p = charge_bonds[charge_temp1]
                elif charge_temp2 in charge_bonds.keys():
                    bond_p = charge_bonds[charge_temp2]
                train_dataset[ind_dataset].bond_params[ind] = torch.tensor(bond_p)
                bonds_added += 1

        # Check if there are the same amount of angles parametrized
        #  as there are angles created
        assert len(train_dataset[ind_dataset].bonds) == bonds_added

        # Sometimes not all angles are assinged, this creates a angle dict fromt the unique atom
        #  types, based on the charges
        charge_angles = {}
        for key, val in param_dict[ind_dataset]["angles"].items():
            charge_angles[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = [val["k"], val["angle"]]

        angles_added = 0
        for ind, i in enumerate(train_dataset[ind_dataset].angles):
            angles_temp = [
                atom_names_long[ind_dataset][i[0]],
                atom_names_long[ind_dataset][i[1]],
                atom_names_long[ind_dataset][i[2]],
            ]
            angle_added = False
            for key, val in param_dict[ind_dataset]["angles"].items():
                if val["other_idx"] == angles_temp or val["other_idx"] == angles_temp[::-1]:
                    angles_p = [val["k"], val["angle"]]
                    train_dataset[ind_dataset].angle_params[ind] = torch.tensor(angles_p)
                    angle_added = True
                    angles_added += 1
                    break
            if not angle_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in angles_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in angles_temp[::-1]]
                )
                if charge_temp1 in charge_angles.keys():
                    angles_p = charge_angles[charge_temp1]
                elif charge_temp2 in charge_angles.keys():
                    angles_p = charge_angles[charge_temp2]
                train_dataset[ind_dataset].angle_params[ind] = torch.tensor(angles_p)
                angles_added += 1

        # Check if there are the same amount of angles parametrized
        #  as there are angles created
        assert len(train_dataset[ind_dataset].angles) == angles_added

        # Sometimes not all dihedrals are assinged, this creates a dihedral dict fromt the unique atom
        #  types, based on the charges
        charge_dihs = {}
        for key, val in param_dict[ind_dataset]["dihedrals"].items():
            charge_dihs[
                ",".join([str(round(charges_dict[c], 3)) for c in val["other_idx"]])
            ] = val["k"]

        dihedrals_added = 0
        for ind, i in enumerate(train_dataset[ind_dataset].dihedrals):
            dihedrals_temp = [
                atom_names_long[ind_dataset][i[0]],
                atom_names_long[ind_dataset][i[1]],
                atom_names_long[ind_dataset][i[2]],
                atom_names_long[ind_dataset][i[3]],
            ]
            dih_added = False
            for key, val in param_dict[ind_dataset]["dihedrals"].items():
                if (
                    val["other_idx"] == dihedrals_temp
                    or val["other_idx"] == dihedrals_temp[::-1]
                ):
                    dihedral_p = val["k"]
                    train_dataset[ind_dataset].dihedral_params[ind] = torch.tensor(dihedral_p)
                    dih_added = True
                    dihedrals_added += 1
                    break
            if not dih_added:
                charge_temp1 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in dihedrals_temp]
                )
                charge_temp2 = ",".join(
                    [str(round(charges_dict[c], 3)) for c in dihedrals_temp[::-1]]
                )
                if charge_temp1 in charge_dihs.keys():
                    dihedral_p = charge_dihs[charge_temp1]
                elif charge_temp2 in charge_dihs.keys():
                    dihedral_p = charge_dihs[charge_temp2]
                train_dataset[ind_dataset].dihedral_params[ind] = torch.tensor(dihedral_p)
                dihedrals_added += 1

        # Check if there are the same amount of dihedrals parametrized
        #  as there are dihedrals created
        assert len(train_dataset[ind_dataset].dihedrals) == dihedrals_added

        # There are usually less imporpers created in ligpargen than
        #  there are actual impropers, so everything that is not assigned
        #  is going to be deleted
        improp_indeces = []
        for ind, i in enumerate(train_dataset[ind_dataset].impropers):
            impropers_temp = [
                atom_names_long[ind_dataset][i[0]],
                atom_names_long[ind_dataset][i[1]],
                atom_names_long[ind_dataset][i[2]],
                atom_names_long[ind_dataset][i[3]],
            ]
            for key, val in param_dict[ind_dataset]["impropers"].items():
                if (
                    val["other_idx"] == impropers_temp
                    or val["other_idx"] == impropers_temp[::-1]
                ):
                    improper_p = val["k"]
                    train_dataset[ind_dataset].improper_params[ind] = torch.tensor(improper_p)
                    improp_indeces.append(ind)
                    break
                elif val["other_idx"][0] == impropers_temp[0]:
                    if np.isin(val["other_idx"][1:], impropers_temp[1:]).all():
                        improper_p = val["k"]
                        train_dataset[ind_dataset].improper_params[ind] = torch.tensor(improper_p)
                        improp_indeces.append(ind)
                        break
        # Dropping all the impropers that are not parametrized
        train_dataset[ind_dataset].impropers = train_dataset[ind_dataset].impropers[improp_indeces]
        train_dataset[ind_dataset].improper_params = train_dataset[ind_dataset].improper_params[improp_indeces]

    return train_dataset


def minimize_polymer(
    save_path,
    long_smiles,
    atoms_long,
    atom_names_long,
    atoms_short,
    atom_names_short,
    param_dict,
    lit_charges_save_path,
    charges,
    name,
    cuda_device="0",
):
    train_dataset = load_hitpoly_params(
        [long_smiles],
        [],
    )
    train_dataset = assign_lpg_params(
        atoms_long=atoms_long,
        atom_names_long=atom_names_long,
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        train_dataset=train_dataset,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
    )

    mol_dict, pdb_file = load_pdb_create_mol_dict(
        save_path, train_dataset, polymer=name
    )

    atom_dict, atom_types_list = atom_name_reindexing(mol_dict)
    # Apply scaling: False for first n solvents, True for remaining molecules (anions/cations)
    mol_keys = list(mol_dict.keys())
    n_solvents = len(long_smiles)  # number of solvent molecules
    
    for i, key in enumerate(mol_keys):
        if i < n_solvents:
            # First n keys are solvents - no scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=False
            )
        else:
            # Remaining keys are anions/cations - apply scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=True
            )
    ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
        mol_dict, atom_types_list
    )

    write_openmm_files(
        save_path, pdb_file, ff_file, ff_file_resid, name_iterables, polymer=name
    )
    print("Temp polymer force field built!")
    equilibrate_polymer(
        save_path=save_path,
        name=name,
        cuda_device=cuda_device,
    )
    print(f"Polymer structure has been minimized and saved at {save_path}")


def atom_name_reindexing(mol_dict):
    dig_let = STRING.digits + STRING.ascii_uppercase
    digits = STRING.digits
    letters = STRING.ascii_uppercase

    temp_len = (len(dig_let) - 1) * len(digits) + (len(digits))
    full_len = (len(letters) - 1) * len(letters) + len(letters) + temp_len

    index_list = []
    for i in itertools.product(digits, repeat=2):
        if i[0] == "0":
            if i[1] == "0":
                continue
            index_list.append(str(i[1]))
        else:
            index_list.append("".join(i))
    for i in itertools.product(letters, digits, repeat=1):
        index_list.append("".join(i))
    for i in itertools.product(letters, letters, repeat=1):
        index_list.append("".join(i))

    atom_types_list = []  # list of lists where each sublist has 4 elements
    # name, class, element, mass
    atom_dict = {}

    # I've got to rename the atoms that repeat, and keep the names
    #  rename them in the loaded pdb file and then save a new pdb file
    #  Gotta save the change in one residue and then transfer to the rest

    for key, mol in mol_dict.items():
        for ind, atom in enumerate(mol["mol_pdb"]._residues[0]._atoms):
            atom_name = atom.name[len(atom.element.symbol) :]
            if atom.element.symbol not in atom_dict:
                atom_dict[atom.element.symbol] = atom_name
            else:
                prev_atom_ind = index_list.index(atom_dict[atom.element.symbol])

                if (
                    index_list.index(atom.name[len(atom.element.symbol) :])
                    <= prev_atom_ind
                ):
                    atom_name = index_list[prev_atom_ind + 1]
                    full_name = atom.element.symbol + atom_name
                    for i in mol["mol_pdb"]._residues:
                        i._atoms[ind].name = full_name
                    atom_dict[atom.element.symbol] = atom_name
                else:
                    atom_dict[atom.element.symbol] = atom_name
            atom_types_list.append(
                [atom.name, atom.name, atom.element.symbol, atom.element.mass._value]
            )

    return atom_dict, atom_types_list


def param_scaling_openmm(topology, scaling=False):
    # scaling mol dict params from lammps to openmm
    topology.bond_params_openmm = topology.bond_params.clone()
    topology.angle_params_openmm = topology.angle_params.clone()
    topology.dihedral_params_openmm = topology.dihedral_params.clone()
    topology.improper_params_openmm = topology.improper_params.clone()
    topology.pair_params_openmm = topology.pair_params.clone()

    if scaling:
        if topology.bond_params.numel():
            topology.bond_params_openmm[:, 0] = (
                topology.bond_params_openmm[:, 0] * 2 * 4.184 * 100
            )
            topology.bond_params_openmm[:, 1] = topology.bond_params_openmm[:, 1] / 10
        if topology.angle_params.numel():
            topology.angle_params_openmm[:, 0] = (
                topology.angle_params_openmm[:, 0] * 2 * 4.184
            )
        if topology.dihedral_params.numel():
            topology.dihedral_params_openmm = (
                topology.dihedral_params_openmm / 2 * 4.184
            )
        if topology.improper_params.numel():
            topology.improper_params_openmm = topology.improper_params_openmm * 4.184
        topology.pair_params_openmm[:, 1] = (
            topology.pair_params_openmm[:, 1] * 4.184
        )  # Epsilon
        topology.pair_params_openmm[:, 2] = (
            topology.pair_params_openmm[:, 2] / 10
        )  # Sigma

    return topology


def param_scaling_lammps(topology, scaling=False):
    # scaling mol dict params from openmm to lammps
    topology.bond_params_openmm = topology.bond_params.clone()
    topology.angle_params_openmm = topology.angle_params.clone()
    topology.dihedral_params_openmm = topology.dihedral_params.clone()
    topology.pair_params_openmm = topology.pair_params.clone()

    if scaling:
        if topology.bond_params.numel():
            topology.bond_params[:, 0] = topology.bond_params_openmm[:, 0] / (
                2 * 4.184 * 100
            )
            topology.bond_params[:, 1] = topology.bond_params_openmm[:, 1] * 10
        if topology.angle_params.numel():
            topology.angle_params[:, 0] = topology.angle_params_openmm[:, 0] / (
                2 * 4.184
            )
        if topology.dihedral_params.numel():
            topology.dihedral_params = topology.dihedral_params_openmm * 2 / 4.184
        topology.pair_params[:, 1] = (
            topology.pair_params_openmm[:, 1] / 4.184
        )  # Epsilon
        topology.pair_params[:, 2] = topology.pair_params_openmm[:, 2] * 10  # Sigma

    return topology


def creating_ff_and_resid_files(mol_dict, atom_types_list):
    ff_file = []
    ff_file.append("<ForceField>")
    ff_file.append("<AtomTypes>")
    for a_name, a_class, a_elem, a_mass in atom_types_list:
        string = f'<Type name="{a_name}" class="{a_class}" element="{a_elem}" mass="{a_mass}" />'
        ff_file.append(string)
    ff_file.append("</AtomTypes>")
    ff_file.append("<Residues>")

    name_iterables = []
    ani_ind = 1
    cat_ind = 1
    ply_ind = 1
    for key in mol_dict.keys():
        minuses = key.count("-")
        pluses = key.count("+")
        if pluses - minuses <= -1:
            name_iterables.append(f"AN{ani_ind}")
            ani_ind += 1
        elif pluses - minuses >= 1:
            name_iterables.append(f"CA{cat_ind}")
            cat_ind += 1
        else:
            name_iterables.append(f"PL{ply_ind}")
            ply_ind += 1

    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        ff_file.append(f'<Residue name="{n}">')  # mol["mol_pdb"].id
        for ind, (atom_pdb, charge) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params[:, 0].detach().numpy(),
            )
        ):
            string = f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />'
            ff_file.append(string)
        for bond in mol["mol_hitpoly"].bonds.numpy():
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f'<Bond from="{bond[0]}" to="{bond[1]}" />'
            ff_file.append(string)
        ff_file.append("</Residue>")
    ff_file.append("</Residues>")
    ff_file.append("<HarmonicBondForce>")
    for key, mol in mol_dict.items():
        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f'<Bond k="{param[0]}" length="{param[1]}" class1="{atom1}" class2="{atom2}" />'
            ff_file.append(string)
    ff_file.append("</HarmonicBondForce>")
    ff_file.append("<HarmonicAngleForce>")
    for key, mol in mol_dict.items():
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string1 = f'<Angle k="{param[0]}" angle="{param[1]}" '
            string2 = f'class1="{atom1}" class2="{atom2}" class3="{atom3}" />'
            ff_file.append(string1 + string2)
    ff_file.append("</HarmonicAngleForce>")
    ff_file.append("<PeriodicTorsionForce>")
    for key, mol in mol_dict.items():
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            string1 = f'<Proper k1="{param[0]}" k2="{param[1]}" k3="{param[2]}" k4="{param[3]}" '
            string2 = (
                f'periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" '
            )
            string3 = 'phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793" '
            string4 = f'class1="{atom1}" class2="{atom2}" class3="{atom3}" class4="{atom4}" />'
            ff_file.append(string1 + string2 + string3 + string4)
    for key, mol in mol_dict.items():
        for improper, param in zip(
            mol["mol_hitpoly"].impropers.detach().numpy(),
            mol["mol_hitpoly"].improper_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[improper[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[improper[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[improper[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[improper[3]].name
            string1 = f'<Improper k1="{param[0]}" k2="{param[1]}" k3="{param[2]}" k4="{param[3]}" '
            string2 = (
                f'periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" '
            )
            string3 = 'phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793" '
            string4 = f'class1="{atom1}" class2="{atom2}" class3="{atom3}" class4="{atom4}" />'
            ff_file.append(string1 + string2 + string3 + string4)
    ff_file.append("</PeriodicTorsionForce>")
    ff_file.append('<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">')
    # ff_file.append('<UseAttributeFromResidue name="charge" />')
    for key, mol in mol_dict.items():
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            string = f'<Atom charge="{param[0]}" epsilon="{param[1]}" sigma="{param[2]}" type="{atom_pdb.name}" />'
            ff_file.append(string)
    ff_file.append("</NonbondedForce>")
    ff_file.append("</ForceField>")

    ff_file_resid = []
    ff_file_resid.append("<Residues>")
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        ff_file_resid.append(f'<Residue name="{n}">')  # mol["mol_pdb"].id
        for ind, (atom_pdb, charge) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params[:, 0].detach().numpy(),
            )
        ):
            string = f'<Atom name="{atom_pdb.name}" type="{atom_pdb.name}" />'
            ff_file_resid.append(string)
        for bond in mol["mol_hitpoly"].bonds.numpy():
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f'<Bond from="{atom1}" to="{atom2}" />'
            ff_file_resid.append(string)
        ff_file_resid.append("</Residue>")
    ff_file_resid.append("</Residues>")

    return ff_file, ff_file_resid, name_iterables


def creating_ff_and_resid_files_gromacs(
    mol_dict,
    atom_types_list,
    save_path,
    pdb_file,
    polymer_count,
    salt_count,
):
    name_iterables = []
    ani_ind = 1
    cat_ind = 1
    ply_ind = 1
    for key in mol_dict.keys():
        minuses = key.count("-")
        pluses = key.count("+")
        if pluses - minuses == -1:
            name_iterables.append(f"AN{ani_ind}")
            ani_ind += 1
        elif pluses - minuses == 1:
            name_iterables.append(f"CA{cat_ind}")
            cat_ind += 1
        else:
            name_iterables.append(f"PL{ply_ind}")
            ply_ind += 1

    atom_ind = 0
    atom_ind_name_dict = {}
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            atom_ind_name_dict[a_name] = atom_ind + 1
            atom_ind += 1

    ff_files_separate = []
    atom_ind = 0
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        ff_file = []
        ff_file.append(";")
        ff_file.append("; GENERATED BY HitPoly")
        ff_file.append("; Jurgis Ruza @ MIT ")
        ff_file.append(";")
        ff_file.append("[ moleculetype ]")
        ff_file.append("; Name               nrexcl")
        ff_file.append(f"{n:>3}{3:>20}")
        ff_file.append("[ atoms ]")

        # I've no idea what this cgnr is and what does it mean
        ff_file.append(
            ";   nr       type  resnr residue  atom   cgnr     charge       mass"
        )
        cgnr = 1
        temp_atom_dict = {}
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            if atom_ind + 1 == 32 * cgnr:
                cgnr += 1
            string = f"{ind+1:>6}{a_name:>11}{resid_ind+1:>6}{n:>7}"
            string1 = f"{a_name:>6}{str(cgnr):>7}{str(round(param[0],6)):>11}{str(round(a_mass,3)):>11}"
            ff_file.append(string + string1)
            temp_atom_dict[a_name] = ind + 1
            atom_ind += 1

        ff_file.append("[ bonds ]")
        ff_file.append(";  ai    aj  funct            c0            c1")

        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>6}{1:>6}"
            string1 = f"{str(round(param[1],4)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)

        ff_file.append("[ angles ]")
        ff_file.append(";  ai    aj    ak funct            c0            c1")
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>6}{temp_atom_dict[atom3]:>6}"
            string1 = f"{1:>6}{str(round(param[1]*57.29578,3)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)

        ff_file.append("[ dihedrals ]")
        ff_file.append("; IMPROPER DIHEDRAL ANGLES ")
        ff_file.append(
            ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
        )
        ff_file.append("")
        ff_file.append("[ dihedrals ]")
        ff_file.append("; PROPER DIHEDRAL ANGLES")
        ff_file.append(
            ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
        )
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            # The *2 multiplication comes from the openmm-gromacs dihedral difference
            C0 = round(
                param[1] + 1 / 2 * (param[0] + param[2]) * 2, 3
            )  # C0 = F2 + 1/2 (F1+F3)
            C1 = round(1 / 2 * (-param[0] + 3 * param[2]) * 2, 3)  # C1 = 1/2 (-F1+3F3)
            C2 = round(-param[1] + 4 * param[3] * 2, 3)  # C2 = -F2 + 4F4
            C3 = round(-2 * param[2] * 2, 3)  # C3 = -2F3
            C4 = round(-4 * param[3] * 2, 3)  # C4 = -4F4
            C5 = round(0, 3)  # C5 = 0
            string = f"{temp_atom_dict[atom1]:>5}{temp_atom_dict[atom2]:>5}{temp_atom_dict[atom3]:>5}"
            string1 = f"{temp_atom_dict[atom4]:>5}{3:>8}{C0:>12}{C1:>8}{C2:>8}{C3:>8}{C4:>8}{C5:>8}"
            ff_file.append(string + string1)
        ff_file.append("")
        ff_files_separate.append(ff_file)

    ff_file = []
    ff_file.append(";")
    ff_file.append("; GENERATED BY HitPoly")
    ff_file.append("; Jurgis Ruza @ MIT ")
    ff_file.append(";")
    ff_file.append("[ atomtypes ]")
    ff_file.append("; name  at.num  mass  charge particletype   sigma    epsilon")
    atom_ind = 0
    for resid_ind, (n, (key, mol)) in enumerate(zip(name_iterables, mol_dict.items())):
        for ind, (atom_pdb, param) in enumerate(
            zip(
                mol["mol_pdb"]._residues[0]._atoms,
                mol["mol_hitpoly"].pair_params_openmm.detach().numpy(),
            )
        ):
            a_name, a_class, a_elem, a_mass = atom_types_list[atom_ind]
            string = f"{a_name:>10}{ELEMENT_TO_NUM[a_elem]:>6}{round(a_mass,4):>11}{str(round(param[0],6)):>10}"
            string1 = (
                f'{"A":>5}{str(round(param[2],6)):>15}{str(round(param[1],6)):>14}'
            )
            ff_file.append(string + string1)
            atom_ind += 1

    ff_file.append("[ bondtypes ]")
    ff_file.append(";  i    j  funct            b0            kb")
    for n, (key, mol) in zip(name_iterables, mol_dict.items()):
        for bond, param in zip(
            mol["mol_hitpoly"].bonds.detach().numpy(),
            mol["mol_hitpoly"].bond_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[bond[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[bond[1]].name
            string = f"{atom1:>5}{atom2:>6}{1:>6}"
            string1 = f"{str(round(param[1],4)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)
    ff_file.append("[ angletypes ]")
    ff_file.append(";  i    j    k funct            theta            k0")
    for key, mol in mol_dict.items():
        for angle, param in zip(
            mol["mol_hitpoly"].angles.detach().numpy(),
            mol["mol_hitpoly"].angle_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[angle[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[angle[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[angle[2]].name
            string = f"{atom1:>5}{atom2:>6}{atom3:>6}"
            string1 = f"{1:>6}{str(round(param[1]*57.29578,3)):>12}{str(round(param[0],3)):>11}"
            ff_file.append(string + string1)

    ff_file.append("[ dihedraltypes ]")
    ff_file.append("; IMPROPER DIHEDRAL ANGLES ")
    ff_file.append(
        ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
    )
    ff_file.append("")
    ff_file.append("[ dihedraltypes ]")
    ff_file.append("; PROPER DIHEDRAL ANGLES")
    ff_file.append(
        ";  ai    aj    ak    al funct            c0            c1            c2            c3            c4            c5"
    )
    for key, mol in mol_dict.items():
        for dihedral, param in zip(
            mol["mol_hitpoly"].dihedrals.detach().numpy(),
            mol["mol_hitpoly"].dihedral_params_openmm.detach().numpy(),
        ):
            atom1 = mol["mol_pdb"]._residues[0]._atoms[dihedral[0]].name
            atom2 = mol["mol_pdb"]._residues[0]._atoms[dihedral[1]].name
            atom3 = mol["mol_pdb"]._residues[0]._atoms[dihedral[2]].name
            atom4 = mol["mol_pdb"]._residues[0]._atoms[dihedral[3]].name
            # The *2 multiplication comes from the openmm-gromacs dihedral difference
            C0 = round(
                param[1] + 1 / 2 * (param[0] + param[2]) * 2, 3
            )  # C0 = F2 + 1/2 (F1+F3)
            C1 = round(1 / 2 * (-param[0] + 3 * param[2]) * 2, 3)  # C1 = 1/2 (-F1+3F3)
            C2 = round(-param[1] + 4 * param[3] * 2, 3)  # C2 = -F2 + 4F4
            C3 = round(-2 * param[2] * 2, 3)  # C3 = -2F3
            C4 = round(-4 * param[3] * 2, 3)  # C4 = -4F4
            C5 = round(0, 3)  # C5 = 0
            string = f"{atom1:>5}{atom2:>5}{atom3:>5}{atom4:>5}"
            string1 = f"{3:>8}{C0:>12}{C1:>8}{C2:>8}{C3:>8}{C4:>8}{C5:>8}"
            ff_file.append(string + string1)
    ff_file.append("")

    box_file = []
    box_file.append("Simulation box")
    box_file.append(f"{len(pdb_file._positions[0])}")
    with open(f"{save_path}/packed_box.pdb", "r") as f:
        lines = f.readlines()
        for i in lines:
            if i.startswith("HETATM"):
                atom_ind = int(i[6:12])
                resid_ind = i[22:26]
                resid = i[12:].split()[1]
                atom = i[12:].split()[0]
                x = round(float(i[28:].split()[0]) / 10, 3)
                y = round(float(i[28:].split()[1]) / 10, 3)
                z = round(float(i[28:].split()[2]) / 10, 3)
                box_file.append(
                    f"{resid_ind:>5}{resid}{atom:>7}{atom_ind:>5}{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                )
    # If we ever are doing a not cube box, ur code probably broke here ;)
    max_box_dimension = pdb_file.positions.max().max()._value
    pbc = round(max_box_dimension, 5)
    box_file.append(f"{pbc:>10}{pbc:>9}{pbc:>10}")
    box_file.append("")

    topology_file = []
    topology_file.append("[ defaults ]")
    topology_file.append(
        "; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ"
    )
    topology_file.append("1               3               yes             0.5     0.5")
    topology_file.append("")
    topology_file.append('#include "./force_field.itp"')
    for i in name_iterables:
        topology_file.append(f'#include "./{i}.top"')
    topology_file.append("")
    topology_file.append("[ system ]")
    topology_file.append("; Name")
    topology_file.append("OPV")
    topology_file.append("")
    topology_file.append("[ molecules ]")
    topology_file.append("; Compound        #mols")
    for name, count in zip(
        name_iterables,
        [polymer_count, *salt_count],
    ):
        topology_file.append(f"{name:<18}{count}")

    return ff_files_separate, box_file, name_iterables, ff_file, topology_file


def write_openmm_files(
    save_path,
    pdb_file,
    ff_file,
    ff_file_resid,
    name_iterables,
    polymer=False,
):
    if polymer:
        packed_name = polymer
    else:
        packed_name = "packed_box.pdb"

    print(f"Writting Force field files at {save_path}")
    with open(f"{save_path}/force_field.xml", "w") as f:
        for i in ff_file:
            f.write(i + "\n")

    with open(f"{save_path}/force_field_resids.xml", "w") as f:
        for i in ff_file_resid:
            f.write(i + "\n")

    for chain, name in zip(pdb_file.topology._chains, name_iterables):
        for ind, i in enumerate(chain._residues):
            chain._residues[ind].name = name  # chain.id

    print("Saving the pdb file with the packed polymer.")
    with open(f"{save_path}/{packed_name}", "w") as f:
        pdb_file.writeFile(
            topology=pdb_file.topology,
            positions=pdb_file.positions,
            file=f,
        )
    print("Rewriting the packed box file with removing the connectivity")
    with open(f"{save_path}/{packed_name}", "r") as f:
        lines = f.readlines()
        new_lines = []
        for i in lines:
            if "CONECT" not in i:
                new_lines.append(i)
    with open(f"{save_path}/{packed_name}", "w") as f:
        for i in new_lines:
            f.write(i)


def write_gromacs_files(
    save_path,
    ff_files,
    ff_file,
    box_file,
    name_iterables,
    topology_file,
):
    print(f"Writting Force field files at {save_path}/gromacs")
    if not os.path.isdir(save_path + "/gromacs"):
        os.makedirs(save_path + "/gromacs")

    with open(f"{save_path}/gromacs/force_field.itp", "w") as f:
        for i in ff_file:
            f.write(i + "\n")

    with open(f"{save_path}/gromacs/packed_box.top", "w") as f:
        for i in topology_file:
            f.write(i + "\n")

    for file, name in zip(ff_files, name_iterables):
        with open(f"{save_path}/gromacs/{name}.top", "w") as f:
            for i in file:
                f.write(i + "\n")

    with open(f"{save_path}/gromacs/packed_box.gro", "w") as f:
        for i in box_file:
            f.write(i + "\n")


def get_concentration_from_molality(
    molality,
    polymer_count,
    smiles,
    add_end_Cs,
    polymer_chain_length=1000,
):
    atom_count_monomer = get_atom_count(smiles)
    repeat_units = round(
        polymer_chain_length / atom_count_monomer
    )  # chain should have around 1000 atoms
    long_smiles, _ = create_long_smiles(
        smiles, repeats=repeat_units, add_end_Cs=add_end_Cs
    )

    def change_chain_length(long_smiles, repeat_units):
        if get_atom_count(long_smiles) > polymer_chain_length+int(polymer_chain_length/10):
            repeat_units -= 1
            long_smiles, repeats = create_long_smiles(
                smiles, repeats=repeat_units, add_end_Cs=True
            )
        elif get_atom_count(long_smiles) < polymer_chain_length-int(polymer_chain_length/10):
            repeat_units += 1
            long_smiles, repeats = create_long_smiles(
                smiles, repeats=repeat_units, add_end_Cs=True
            )
        return long_smiles, repeat_units

    # polymer of 1000+-100 atoms
    while get_atom_count(long_smiles) > polymer_chain_length+int(polymer_chain_length/10) or get_atom_count(long_smiles) < polymer_chain_length-int(polymer_chain_length/10):
        long_smiles, repeat_units = change_chain_length(long_smiles, repeat_units)

    concentration = round(molality * get_mol_mass(long_smiles) * polymer_count / 1000)

    #concentration needs to be a list here for create_packmol_input_file !!! 
    return [concentration, concentration], repeat_units


def create_box_and_ff_files(
    save_path,
    long_smiles,
    short_smiles,
    filename,
    polymer_count,
    concentration,
    packmol_path,
    random_seed,
    atoms_short,
    atoms_long,
    atom_names_short,
    atom_names_long,
    param_dict,
    lit_charges_save_path,
    charges,
    charge_scale,
    salt_smiles,
    salt_paths,
    salt_data_paths,
    box_multiplier,
    salt,
):
    poly_paths = [f"{save_path}/{filename}"]

    print(
        f"Creating and running packmol files with {polymer_count} chains and {concentration} {salt_smiles}"
    )
    print(f"at {save_path} path")
    create_packmol_input_file(
        poly_paths,
        salt_paths,
        f"{save_path}/packmol.inp",
        f"{save_path}/packed_box.pdb",
        salt_smiles=salt_smiles,
        polymer_count=polymer_count,
        salt_concentrations=concentration,
        box_multiplier=box_multiplier,  # for small bent molecule x4, for straight-ish molecules around 0.5
        tolerance=2.0,
        salt=salt,
        random_seed=random_seed
    )

    run_packmol(save_path, packmol_path)

    print(f"Making all the force field files for simulations")
    train_dataset = load_hitpoly_params(
        [long_smiles],
        salt_smiles,
        salt_data_paths=salt_data_paths,
        charge_scale=charge_scale,
    )

    train_dataset = assign_lpg_params(
        atoms_long=atoms_long,
        atom_names_long=atom_names_long,
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        train_dataset=train_dataset,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
    )

    mol_dict, pdb_file = load_pdb_create_mol_dict(save_path, train_dataset)

    atom_dict, atom_types_list = atom_name_reindexing(mol_dict)
    # Apply scaling: False for first n solvents, True for remaining molecules (anions/cations)
    mol_keys = list(mol_dict.keys())
    n_solvents = len(long_smiles)  # number of solvent molecules
    
    for i, key in enumerate(mol_keys):
        if i < n_solvents:
            # First n keys are solvents - no scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=False
            )
        else:
            # Remaining keys are anions/cations - apply scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=True
            )
    ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
        mol_dict, atom_types_list
    )
    write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables)

    ff_files, box_file, name_iterables, ff_file, topology_file = (
        creating_ff_and_resid_files_gromacs(
            mol_dict,
            atom_types_list,
            save_path,
            pdb_file,
            polymer_count,
            concentration,
        )
    )
    write_gromacs_files(
        save_path=save_path,
        ff_files=ff_files,
        ff_file=ff_file,
        box_file=box_file,
        name_iterables=name_iterables,
        topology_file=topology_file,
    )
    print(f"Box builder for - done!")


def create_box_and_ff_files_openmm(
    save_path,
    long_smiles,
    filename,
    solvent_count,
    concentration,
    packmol_path,
    atoms_short,
    atoms_long,
    atom_names_short,
    atom_names_long,
    param_dict,
    lit_charges_save_path,
    polynames,
    charges,
    charge_scale,
    salt_smiles,
    salt_paths,
    salt_data_paths,
    box_multiplier,
    salt,
):
    print("Creating box with OpenMM equilibration")
    poly_paths = [f"{save_path}/{name}" for name in filename]

    print(
        f"Creating and running packmol files with {solvent_count} chains and {concentration} {salt_smiles}"
    )
    print(f"at {save_path} path")
    create_packmol_input_file(
        poly_paths,
        salt_paths,
        f"{save_path}/packmol.inp",
        f"{save_path}/packed_box.pdb",
        salt_smiles=salt_smiles,
        polymer_count=solvent_count,
        salt_concentrations=concentration,
        box_multiplier=box_multiplier,  # for small bent molecule x4, for straight-ish molecules around 0.5
        tolerance=2.0,
        salt=salt,
    )

    run_packmol(save_path, packmol_path)

    print(f"Making all the force field files for simulations")
    train_dataset = load_hitpoly_params(
        long_smiles,
        salt_smiles,
        salt_data_paths=salt_data_paths,
        charge_scale=charge_scale,
    )

    train_dataset = assign_lpg_params(
        atoms_long=atoms_long,
        atom_names_long=atom_names_long,
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        train_dataset=train_dataset,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        polynames=polynames,
        charges=charges,
    )

    mol_dict, pdb_file = load_pdb_create_mol_dict(save_path, train_dataset)

    atom_dict, atom_types_list = atom_name_reindexing(mol_dict)
    # Apply scaling: False for first n solvents, True for remaining molecules (anions/cations)
    mol_keys = list(mol_dict.keys())
    n_solvents = len(long_smiles)  # number of solvent molecules
    
    for i, key in enumerate(mol_keys):
        if i < n_solvents:
            # First n keys are solvents - no scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=False
            )
        else:
            # Remaining keys are anions/cations - apply scaling
            mol_dict[key]["mol_hitpoly"] = param_scaling_openmm(
                mol_dict[key]["mol_hitpoly"], scaling=True
            )
    ff_file, ff_file_resid, name_iterables = creating_ff_and_resid_files(
        mol_dict, atom_types_list
    )
    write_openmm_files(save_path, pdb_file, ff_file, ff_file_resid, name_iterables)

    print(f"Box builder for - done!")
