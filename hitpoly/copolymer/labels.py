"""Per-atom monomer labels for RDF analysis of alternating and block copolymers."""
import ast

from openmm.app import PDBFile
from rdkit import Chem


def write_atom_names_rdf_from_pdb(pdb_path, output_path="atom_names_rdf.txt"):
    """One 'element, residue name, residue id' line per atom of a PDB."""
    pdb = PDBFile(pdb_path)
    with open(output_path, "w") as f:
        for atom in pdb.topology.atoms():
            f.write(f"{atom.element.symbol}, {atom.residue.name}, {atom.residue.id}\n")
    print(f"Created {output_path}")


def write_atom_labels_from_log(atom_names_file, log_file, output_file="atom_labels_rdf.txt"):
    """
    Label every heavy atom of each P1 chain with its monomer index (0, 1, ...; 2 for end
    caps) using the chain log from create_long_smiles; H and non-polymer atoms get 3.
    """
    architecture = monomer_counts = final_smiles = total_atoms = block_sizes_and_monomers = None
    with open(log_file) as f:
        for line in f:
            if line.startswith("Architecture:"):
                architecture = line.split("Architecture:")[1].strip()
            elif line.startswith("Block sizes and monomers:"):
                block_sizes_and_monomers = ast.literal_eval(line.split("Block sizes and monomers:")[1].strip())
            elif line.startswith("Monomer counts:"):
                monomer_counts = ast.literal_eval(line.split("Monomer counts:")[1].strip())
            elif line.startswith("Final SMILES:"):
                final_smiles = line.split("Final SMILES:")[1].strip()
            elif line.startswith("Total number of atoms:"):
                total_atoms = int(line.split("Total number of atoms:")[1].strip())

    if architecture not in ["alternating", "block"]:
        raise NotImplementedError("Only 'alternating' and 'block' architectures are supported.")
    if not monomer_counts or not final_smiles or not total_atoms:
        raise ValueError("Missing required fields in log file.")

    unique_monomers = list(monomer_counts.keys())
    monomer_id_map = {smi: idx for idx, smi in enumerate(unique_monomers)}
    heavy_counts = {
        smi: Chem.MolFromSmiles(smi.replace("[Cu]", "").replace("[Au]", "")).GetNumAtoms()
        for smi in unique_monomers
    }

    label_pattern = [2]  # initial cap
    if architecture == "alternating":
        for i in range(sum(monomer_counts.values())):
            smi = unique_monomers[i % 2]
            label_pattern.extend([i % 2] * heavy_counts[smi])
    else:
        for block_size, smi in block_sizes_and_monomers:
            label_pattern.extend([monomer_id_map[smi]] * (block_size * heavy_counts[smi]))
    label_pattern.append(2)  # final cap

    with open(atom_names_file) as f:
        atoms = [line.strip().split(",") for line in f]

    chains, current = [], []
    for i, (element, resname, _) in enumerate(atoms):
        if resname.strip() == "P1" and element.strip() != "H":
            current.append(i)
        elif current:
            chains.append(current)
            current = []
    if current:
        chains.append(current)

    labels = [3] * len(atoms)
    for chain in chains:
        if len(chain) != len(label_pattern):
            raise ValueError(f"Chain length {len(chain)} != expected pattern length {len(label_pattern)}")
        for idx, label in zip(chain, label_pattern):
            labels[idx] = label

    with open(output_file, "w") as f:
        for (element, resname, _), label in zip(atoms, labels):
            f.write(f"{element.strip()}, {resname.strip()}, {label}\n")
    print(f"Saved labeled RDF file to: {output_file}")
