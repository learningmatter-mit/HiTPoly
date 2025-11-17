from hitpoly.writers.box_builder import create_long_smiles, get_atom_count, get_mol_mass
import numpy as np
from typing import List, Tuple
from rdkit import Chem

def calculate_box_numbers(
    smiles,
    repeats,
    concentration=None,
    polymer_count=None,
    solv_atom_count=None,
    end_Cs=True,
    salt_smiles=None,
):
    long_smiles, _ = create_long_smiles(smiles, repeats=repeats, add_end_Cs=end_Cs)
    print(
        f"Polymer chain has {get_atom_count(long_smiles)} atoms, should be around 800 to not brake."
    )
    if not concentration and not polymer_count:
        return

    if not salt_smiles:
        salt_smiles = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "[Li+]"]

    print(
        "Amount of atoms in box",
        get_atom_count(long_smiles, salt_smiles, polymer_count, concentration),
    )

    try:
        print(
            "Repeat unit molecular mass",
            get_mol_mass(smiles.replace("[Cu]", "").replace("[Au]", "")),
        )
    except:
        print(
            "Repeat unit molecular mass",
            get_mol_mass(smiles.replace("[Cu]", "C").replace("[Au]", "C")),
        )

  # molality
    print(
        "Molality", concentration / (get_mol_mass(long_smiles) * polymer_count) * 1000
    )


    # Li:solv_atoms
    print(
        "Li:SolvAtom ratio",
        (concentration / (repeats * polymer_count * solv_atom_count)),
    )
  
    return long_smiles


def salt_string_to_values(hitpoly_path, salt_string, concentration):
    """
    Convert salt string to salt smiles, paths, data paths, and anion name for RDF analysis.
    Also define the concentration based on the salt identity, ex, if Li is the cation then the anion is the same 
    amount as the cation, if Zn is the cation then the anion is 1/2 the amount of the cation.

    To use divalent anions, the code has to be modified to use the correct concentration.

    Args:
        hitpoly_path (str): Path to hitpoly data directory.
        salt_string (str): String representing the salt, e.g. "Li.TFSI".
    """

    cation = salt_string.split(".")[0]
    anion = salt_string.split(".")[1]

    
    salt_path = f"{hitpoly_path}/data/pdb_files"
    if cation == "Na":
        salt_smiles = ["[Na+]"]
        salt_paths = [
            f"{salt_path}/geometry_file_Na.pdb",
        ]
        salt_data_paths = [
            f"{hitpoly_path}/data/forcefield_files/lammps_Na_q100.data",
        ]
        concentration = [concentration, concentration]
    elif cation == "Li":
        salt_smiles = ["[Li+]"]
        salt_paths = [
            f"{salt_path}/geometry_file_Li.pdb",
        ]
        salt_data_paths = [
            f"{hitpoly_path}/data/forcefield_files/lammps_Li_q100.data",
        ]
        concentration = [concentration, concentration]
    elif cation == "Zn":
        salt_smiles = ["[Zn++]"]
        salt_paths = [
            f"{salt_path}/geometry_file_Zn.pdb",
        ]
        salt_data_paths = [
            f"{hitpoly_path}/data/forcefield_files/lammps_Zn_q100.data",
        ]
        concentration = [concentration, concentration*2]
    else:
        raise ValueError(f"Cation {cation} not supported")
    
    if anion == "TFSI":
        salt_smiles.append("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F")
        salt_paths.append(f"{salt_path}/geometry_file_TFSI.pdb")
        salt_data_paths.append(f"{hitpoly_path}/data/forcefield_files/lammps_TFSI_q100.data")
        ani_name_rdf = "N,O"
    elif anion == "PF6":
        salt_smiles.append("F[P-](F)(F)(F)(F)F")
        salt_paths.append(f"{salt_path}/geometry_file_PF6.pdb")
        salt_data_paths.append(f"{hitpoly_path}/data/forcefield_files/lammps_PF6_q100.data")
        ani_name_rdf = "P,F"
    elif anion == "FSI":
        salt_smiles.append("[N-](S(=O)(=O)F)S(=O)(=O)F")
        salt_paths.append(f"{salt_path}/geometry_file_FSI.pdb")
        salt_data_paths.append(f"{hitpoly_path}/data/forcefield_files/lammps_FSI_q100.data")
        ani_name_rdf = "N,O"
    else:
        raise ValueError(f"Anion {anion} not supported")

    return salt_smiles, salt_paths, salt_data_paths, ani_name_rdf, concentration


def calculate_composition_by_mol_fractions(
    molecular_weights: 'List[float]',
    atoms_per_molecule: 'List[int]',
    mole_fractions: 'List[float]',
    total_atoms: int
) -> 'Tuple[List[int], List[float], List[float]]':
    """
    Calculates the number of molecules for each component based on mole fractions and a target total atom count.
    This lean version is optimized for systems with large size disparities between molecules.

    Args:
        molecular_weights (list[float]): List of molecular weights (g/mol) for each component.
        atoms_per_molecule (list[int]): List of atoms per molecule/monomer for each component.
        mole_fractions (list[float]): List of desired mole fractions for each component (must sum to 1.0).
        total_atoms (int): The target total number of atoms in the simulation box.

    Returns:
        tuple[list[int], list[float], list[float]]: A tuple containing three lists:
            - number_of_molecules (list[int]): The calculated integer number of molecules for each component.
            - mol_percent (list[float]): The mole percent for each component.
            - weight_percent (list[float]): The resulting weight percent for each component.
    """
    # 1. Calculate the average number of atoms per molecule, weighted by mole fraction
    avg_apm = sum([mf * apm for mf, apm in zip(mole_fractions, atoms_per_molecule)])
    if avg_apm == 0:
        n = len(molecular_weights)
        return [0] * n, [mf * 100 for mf in mole_fractions], [0.0] * n

    # 2. Calculate ideal (unrounded) molecule counts
    total_molecules_ideal = total_atoms / avg_apm
    unrounded_counts = [mf * total_molecules_ideal for mf in mole_fractions]

    # 3. Find the least abundant (non-zero) component to use as the reference
    non_zero_counts = [(count, i) for i, count in enumerate(unrounded_counts) if count > 0]
    if not non_zero_counts:
        n = len(molecular_weights)
        return [0] * n, [mf * 100 for mf in mole_fractions], [0.0] * n
        
    key_count, key_component_index = min(non_zero_counts)

    # 4. Set the integer count for this key component, ensuring it's at least 1
    num_key_component = round(key_count) or 1
        
    # 5. Recalculate all other component counts based on the key component to preserve ratios
    key_mf = mole_fractions[key_component_index]
    number_of_molecules = [round(num_key_component * (mf / key_mf)) for mf in mole_fractions]

    # 6. Calculate final weight percentages from the actual number of molecules
    masses = [n * mw for n, mw in zip(number_of_molecules, molecular_weights)]
    total_mass = sum(masses)
    weight_prcnt = [(m / total_mass) * 100 if total_mass > 0 else 0.0 for m in masses]
    
    # 7. Convert mole fractions to mole percentages for the output
    mol_prcnt = [mf * 100 for mf in mole_fractions]
    
    return np.array(number_of_molecules), mol_prcnt, weight_prcnt

def calculate_composition_by_weight_fractions(
    molecular_weights: 'List[float]',
    atoms_per_molecule: 'List[int]',
    weight_fractions: 'List[float]',
    total_atoms: int
) -> 'Tuple[List[int], List[float], List[float]]':
    """
    Calculates the number of molecules for each component based on weight fractions and a target total atom count.
    Optimized for systems with large size disparities between molecules.

    Args:
        molecular_weights (list[float]): List of molecular weights (g/mol) for each component.
        atoms_per_molecule (list[int]): List of atoms per molecule/monomer for each component.
        weight_fractions (list[float]): List of desired weight fractions for each component (should sum to 1.0).
        total_atoms (int): The target total number of atoms in the simulation box.

    Returns:
        tuple[list[int], list[float], list[float]]: A tuple containing three lists:
            - number_of_molecules (list[int]): The calculated integer number of molecules for each component.
            - mol_percent (list[float]): The resulting mole percent for each component (from realized counts).
            - weight_percent (list[float]): The resulting weight percent for each component (from realized counts).
    """

    # Normalize weight fractions if they don't sum to 1 due to numerical noise
    total_w = float(sum(weight_fractions))
    if total_w <= 0:
        n = len(molecular_weights)
        return np.array([0] * n), [0.0] * n, [0.0] * n
    w_norm = [w / total_w for w in weight_fractions]

    # Convert weight fractions to mole-fraction proxies via w_i / M_i
    mole_proportions = [w / mw if mw > 0 else 0.0 for w, mw in zip(w_norm, molecular_weights)]
    total_mole_prop = sum(mole_proportions)
    if total_mole_prop == 0:
        n = len(molecular_weights)
        return np.array([0] * n), [0.0] * n, [w * 100 for w in w_norm]
    mole_fractions = [mp / total_mole_prop for mp in mole_proportions]

    # Compute average atoms per molecule with the derived mole fractions
    avg_apm = sum([mf * apm for mf, apm in zip(mole_fractions, atoms_per_molecule)])
    if avg_apm == 0:
        n = len(molecular_weights)
        return np.array([0] * n), [0.0] * n, [w * 100 for w in w_norm]

    # Ideal unrounded molecule counts to target the total atom budget
    total_molecules_ideal = total_atoms / avg_apm
    unrounded_counts = [mf * total_molecules_ideal for mf in mole_fractions]

    # Choose the least abundant non-zero component as a stable reference
    non_zero_counts = [(count, i) for i, count in enumerate(unrounded_counts) if count > 0]
    if not non_zero_counts:
        n = len(molecular_weights)
        return np.array([0] * n), [0.0] * n, [w * 100 for w in w_norm]
    key_count, key_idx = min(non_zero_counts)

    # Set the integer count for the key component (at least 1)
    num_key_component = round(key_count) or 1

    # Preserve the (derived) mole-fraction ratios for the rest
    key_mf = mole_fractions[key_idx]
    number_of_molecules = [round(num_key_component * (mf / key_mf)) for mf in mole_fractions]

    # Calculate realized mol% from counts
    total_molecules_real = sum(number_of_molecules)
    mol_prcnt = [(n_i / total_molecules_real) * 100 if total_molecules_real > 0 else 0.0 for n_i in number_of_molecules]

    # Calculate realized wt% from counts
    masses = [n_i * mw for n_i, mw in zip(number_of_molecules, molecular_weights)]
    total_mass = sum(masses)
    weight_prcnt = [(m / total_mass) * 100 if total_mass > 0 else 0.0 for m in masses]

    return np.array(number_of_molecules), mol_prcnt, weight_prcnt

def get_concentraiton_from_molality_multi_system(
    smiles:list,
    molality:float,
    add_end_Cs:bool=True,
    system:str="liquid",
    atom_count:int=None,# 15k for liquids, 25k for polymers
    polymer_chain_length:int=None,
    ratios:list=None,
    ratios_type:str="mol",
    salt_smiles:str="O=S(=O)(NS(=O)(=O)C(F)(F)F)C(F)(F)F.[Li]" # LiTFSI,
):
    if not atom_count:
        if system == "liquid":
            for smile in smiles:
                if "[Cu]" in smile or "[Au]" in smile:
                    raise ValueError("Liquid must not contain [Cu] or [Au]")
            atom_count = 16000
        elif system == "polymer":
            for smile in smiles:
                if "[Cu]" not in smile and "[Au]" not in smile:
                    raise ValueError("Polymer must contain [Cu] and [Au]")
            atom_count = 23000
        elif system == "gel":
            if not ratios:
                raise ValueError("Ratios must be provided for gel")
            if np.array(ratios).sum() != 1:
                raise ValueError("Ratios must sum to 1")
            if len(ratios) != len(smiles):
                raise ValueError("Ratios must have the same length as smiles")
            atom_count = 23000
        else:
            raise ValueError("System must be either liquid, polymer or gel")
    
    if system == "liquid" or system == "polymer":
        if ratios or len(smiles)>1:
            if np.array(ratios).sum() != 1:
                raise ValueError("Ratios must sum to 1")
            if len(ratios) != len(smiles):
                raise ValueError("Ratios must have the same length as smiles")

    poly_name = []
    atom_count_solvent = []
    mol_mass = []
    repeat_units = []
    for ind, smile in enumerate(smiles):
        if "[Cu]" in smile and "[Au]" in smile:
            try:
                temp_smile = smile.replace("[Cu]", "").replace("[Au]", "")
                atom_count_monomer = get_atom_count(temp_smile)
            except:
                temp_smile = smile.replace("[Cu]", "C").replace("[Au]", "C")
                atom_count_monomer = get_atom_count(temp_smile)
                print("Replacing end groups with carbons to avoid syntax errors for", temp_smile)
            
            if not polymer_chain_length:
                polymer_chain_length = 1100
            repeat_units.append(round(
                polymer_chain_length // atom_count_monomer
            ))  # chain should have around 1000 atoms
            # print(f"Repeat units: {repeat_units[ind]}, atoms monomer: {atom_count_monomer}")
            smile, _ = create_long_smiles(
                smile, repeats=repeat_units[ind], add_end_Cs=add_end_Cs
            )
            # print(f"Chain length for {smiles[ind]}, atoms: ", get_atom_count(smile))
            atom_count_solvent.append(get_atom_count(smile))
            mol_mass.append(get_mol_mass(smile))
            poly_name.append(Chem.MolFromSmiles(temp_smile).GetAtomWithIdx(0).GetSymbol())
        else:
            repeat_units.append(1)
            atom_count_solvent.append(get_atom_count(smile))
            mol_mass.append(get_mol_mass(smile))
            poly_name.append(Chem.MolFromSmiles(smile).GetAtomWithIdx(0).GetSymbol())
    mol_mass = np.array(mol_mass)
    atom_count_solvent = np.array(atom_count_solvent)

    if len(smiles)>1:
        if ratios_type == "mol":
            number_of_molecules, mol_prcnt, weight_prcnt = calculate_composition_by_mol_fractions(mol_mass, atom_count_solvent, ratios, atom_count)
        elif ratios_type == "weight":
            number_of_molecules, mol_prcnt, weight_prcnt = calculate_composition_by_weight_fractions(mol_mass, atom_count_solvent, ratios, atom_count)
        else:
            raise ValueError("Ratios type must be either mol or weight")
        concentration = (molality * mol_mass.dot(number_of_molecules) / 1000).astype(int)
        total_atoms = get_atom_count(salt_smiles)*concentration+number_of_molecules.dot(atom_count_solvent)
    else:
        number_of_molecules = np.array([atom_count//atom_count_solvent[0]])
        concentration = (molality * mol_mass * number_of_molecules / 1000).astype(int)[0]
        weight_prcnt = [100]
        total_atoms = get_atom_count(salt_smiles)*concentration+number_of_molecules[0]*atom_count_solvent[0]

    print(f"Concentration: {concentration}, number_of_molecules: {number_of_molecules}, repeat_units: {repeat_units}, weight_prcnt: {weight_prcnt}, total_atoms: {total_atoms}")
    return [concentration, concentration], number_of_molecules.tolist(), repeat_units, weight_prcnt, total_atoms, poly_name