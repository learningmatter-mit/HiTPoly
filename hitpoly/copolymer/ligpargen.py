"""
LigParGen parameterization of oligomers and the combined parameter dict.

LigParGen is run only on the short oligomers: long chains are too large for it. The long
chains are typed jointly with the oligomers (same TopologyDataset) so atom-type IDs match,
and their parameters come from the oligomers through those shared IDs.
"""
import os
import shutil
import subprocess
import time

from openmm.app import ForceField
from rdkit import Chem
from rdkit.Chem import AllChem

from hitpoly.data.builder import TopologyBuilder, TopologyDataset
from hitpoly.utils.args import hitpolyArgs
from hitpoly.writers.box_builder import perlmutter_ligpargen

NUMBER_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
}

LIGPARGEN_TIMEOUT_S = 6 * 3600  # queue wait + run; LigParGen itself takes minutes


def _typing_args(neighborhood_depth):
    args = hitpolyArgs()
    args.discrete_flag = True
    args.discrete_neighborhood_depth = neighborhood_depth
    return args


def write_ligpargen_mol(smiles, mol_path):
    """Embed the molecule in 3D and write the .mol file LigParGen reads."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles), addCoords=True)
    confs = AllChem.EmbedMultipleConfs(mol, numConfs=1, maxAttempts=1000, boxSizeMult=5, useRandomCoords=False)
    if len(confs) == 0:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, maxAttempts=20000, boxSizeMult=5, useRandomCoords=True)
    with open(mol_path, "w") as f:
        f.write(Chem.MolToMolBlock(mol))
    return mol


def submit_supercloud_ligpargen(ligpargen_path, mol_filename, output_prefix):
    """
    Submit LigParGen (podman container) as its own CPU job and return the SLURM job id.

    LigParGen/BOSS writes fixed-name scratch files (optzmat, out, log, plt.pdb, ...) in its
    working directory, so each oligomer runs in its own subfolder (with its own podman
    storage) and only <prefix>.xml is moved up into ligpargen_path when it is complete.
    """
    workdir = os.path.join(ligpargen_path, output_prefix)
    os.makedirs(workdir, exist_ok=True)
    mol_basename = os.path.basename(mol_filename)
    shutil.copy(mol_filename, os.path.join(workdir, mol_basename))
    script = os.path.join(workdir, f"run_{output_prefix}.sh")
    with open(script, "w") as f:
        f.write("\n".join([
            "#!/bin/bash",
            f"#SBATCH --job-name=lpg_{output_prefix}",
            "#SBATCH --partition=xeon-p8",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --time=2:00:00",
            f"#SBATCH --output={workdir}/slurm_%j.out",
            "",
            f"cd {workdir}",
            "export XDG_DATA_HOME=/state/partition1/user/$(id -un)/podman_$SLURM_JOB_ID",
            "mkdir -p $XDG_DATA_HOME",
            "podman load -i $HOME/containers/ligpargen.tar",
            # On some nodes the container finishes but never exits; the timeout stops it, and
            # the XML is moved up before cleanup (which can hang the same way) as long as it is complete.
            "timeout 30m podman run --rm -w /app/RUN -v .:/app/RUN:Z ligpargen:latest python ../LigParGen/Converter.py "
            f"-m {mol_basename} -o 0 -c 0 -r {output_prefix} -d . -l",
            f"if grep -q '</ForceField>' {output_prefix}.xml 2>/dev/null; then cp {output_prefix}.xml ../.{output_prefix}.xml.tmp "
            f"&& mv ../.{output_prefix}.xml.tmp ../{output_prefix}.xml; fi",
            "timeout 5m podman unshare rm -rf $XDG_DATA_HOME 2>/dev/null || rm -rf $XDG_DATA_HOME",
            "",
        ]))
    out = subprocess.run(["sbatch", "--parsable", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script}: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def _job_in_queue(job_id):
    out = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() != ""


def wait_for_ligpargen(xml_paths_by_job, timeout=LIGPARGEN_TIMEOUT_S, poll=30):
    """
    Wait until every XML exists. Raises if a LigParGen job leaves the queue without
    writing its XML, or if the timeout is reached.
    """
    pending = dict(xml_paths_by_job)
    t0 = time.time()
    while pending:
        for job_id, xml in list(pending.items()):
            if os.path.exists(xml):
                pending.pop(job_id)
            elif not _job_in_queue(job_id):
                time.sleep(poll)  # file system lag after the job ends
                if not os.path.exists(xml):
                    raise RuntimeError(
                        f"LigParGen job {job_id} finished without writing {xml}; "
                        f"see slurm_{job_id}.out in {xml[:-4]}/."
                    )
                pending.pop(job_id)
        if pending:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"LigParGen still running after {timeout/3600:.1f} h: {sorted(pending.values())}")
            time.sleep(poll)
    time.sleep(5)  # let LigParGen finish writing its other files


def run_ligpargen(jobs, ligpargen_path, platform):
    """
    jobs: list of (smiles, output_prefix). Writes <prefix>.mol and produces <prefix>.xml
    in ligpargen_path for each. On SuperCloud all jobs are submitted first, then awaited.
    """
    submitted = {}
    for smiles, prefix in jobs:
        mol_filename = os.path.join(ligpargen_path, f"{prefix}.mol")
        write_ligpargen_mol(smiles, mol_filename)
        if platform == "local":
            subprocess.run(
                f"LigParGen -m {os.path.basename(mol_filename)} -o 0 -c 0 -r {prefix} -d . -l",
                shell=True, cwd=ligpargen_path,
            )
        elif platform == "supercloud":
            job_id = submit_supercloud_ligpargen(ligpargen_path, mol_filename, prefix)
            print(f"  Submitted LigParGen for {prefix} as job {job_id}")
            submitted[job_id] = os.path.join(ligpargen_path, f"{prefix}.xml")
        elif platform == "perlmutter":
            perlmutter_ligpargen(ligpargen_path, mol_filename, prefix)
        else:
            raise ValueError(f"Unsupported platform for copolymer LigParGen: '{platform}'")
    if submitted:
        wait_for_ligpargen(submitted)


def generate_parameter_dict(save_path, output_prefix, atom_names, atoms, bonds_typed):
    """
    Read a LigParGen XML and return a parameter dict keyed by hitpoly atom-type IDs.
    Checks that atom elements and bonds match between LigParGen and RDKit.
    """
    forcefield = ForceField(f"{save_path}/{output_prefix}.xml")
    template_name = next(n for n in forcefield._templates if n.startswith("poly"))
    template = forcefield._templates[template_name]

    atoms_ff = [a.element._symbol for a in template.atoms]
    assert atoms_ff == atoms, f"Atom element mismatch:\n  FF:    {atoms_ff}\n  RDKit: {atoms}"

    bond_list = [list(b) for b in template.bonds]
    matched = sum(1 for bl in bond_list for bt in bonds_typed if set(bl) == set(bt))
    assert matched == len(bond_list), f"Bond count mismatch: {matched} matched vs {len(bond_list)} in FF"
    print(f"[generate_parameter_dict] LigParGen <-> RDKit atoms and bonds verified for '{output_prefix}'")

    param_dict = {"atoms": {}, "bonds": {}, "angles": {}, "dihedrals": {}, "impropers": {}}
    type2atom, atomff2typeid = {}, {}
    nonbonded = forcefield._forces[3].params.paramsForType

    for ind, atom in enumerate(template.atoms):
        ff_type, type_id = atom.type, atom_names[ind]
        p = nonbonded[ff_type]
        if type_id not in param_dict["atoms"]:
            param_dict["atoms"][type_id] = {"type": ff_type, "element": atom.element._symbol,
                                            "charges": [p["charge"]], "sigma": p["sigma"], "epsilon": p["epsilon"]}
        else:
            param_dict["atoms"][type_id]["charges"].append(p["charge"])
        type2atom[ff_type] = atom.name
        atomff2typeid[atom.name] = type_id

    bonds = forcefield._forces[0]
    for ind, (t1, t2) in enumerate(zip(bonds.types1, bonds.types2)):
        a1, a2 = type2atom[list(t1)[0]], type2atom[list(t2)[0]]
        param_dict["bonds"][f"bond_{ind}"] = {
            "ff_idx": [a1, a2], "other_idx": [atomff2typeid[a1], atomff2typeid[a2]],
            "length": bonds.length[ind], "k": bonds.k[ind],
        }

    angles = forcefield._forces[1]
    for ind, (t1, t2, t3) in enumerate(zip(angles.types1, angles.types2, angles.types3)):
        names = [type2atom[list(t)[0]] for t in (t1, t2, t3)]
        param_dict["angles"][f"angle_{ind}"] = {
            "ff_idx": names, "other_idx": [atomff2typeid[a] for a in names],
            "angle": angles.angle[ind], "k": angles.k[ind],
        }

    for ind, dih in enumerate(forcefield._forces[2].proper):
        names = [type2atom[list(t)[0]] for t in (dih.types1, dih.types2, dih.types3, dih.types4)]
        param_dict["dihedrals"][f"dihedral_{ind}"] = {
            "ff_idx": names, "other_idx": [atomff2typeid[a] for a in names],
            "periodicity": dih.periodicity, "phase": dih.phase, "k": dih.k,
        }

    return param_dict


def create_combined_param_dict(oligomers_list, long_smiles_list, ligpargen_path, platform, neighborhood_depth=2):
    """
    Run LigParGen on every oligomer (skipping ones whose XML already exists) and merge the
    results into one parameter dict. Charges are collected per atom type across oligomers
    so they can be averaged later.

    Returns (param_dict, atoms_short, atom_names_short, train_dataset); train_dataset holds
    the oligomers followed by the long chains, all typed jointly.
    """
    def _to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    train_args = _typing_args(neighborhood_depth)
    train_molecule_data = [
        TopologyBuilder(smiles=[smi], train_args=train_args, load_geoms=False, lj_from_file=False)
        for smi in list(oligomers_list) + list(long_smiles_list)
    ]
    train_dataset = TopologyDataset(data=train_molecule_data, train_args=train_args)

    prefixes = [f"poly_{oi + 1}" for oi in range(len(oligomers_list))]
    missing = [(smi, p) for smi, p in zip(oligomers_list, prefixes)
               if not os.path.exists(os.path.join(ligpargen_path, f"{p}.xml"))]
    if missing:
        print(f"Running LigParGen on {len(missing)} of {len(oligomers_list)} oligomers ({platform})")
        run_ligpargen(missing, ligpargen_path, platform)

    combined = {"atoms": {}, "bonds": {}, "angles": {}, "dihedrals": {}, "impropers": {}}
    atoms_short, atom_names_short = [], []

    for oi, (smi, prefix) in enumerate(zip(oligomers_list, prefixes)):
        if not os.path.exists(os.path.join(ligpargen_path, f"{prefix}.xml")):
            raise FileNotFoundError(f"LigParGen produced no {prefix}.xml for oligomer {oi} ('{smi[:40]}').")

        mol_data = train_molecule_data[oi]
        atom_names = mol_data.atom_type.tolist()
        param_dict = generate_parameter_dict(
            ligpargen_path, prefix,
            atom_names=atom_names,
            atoms=[NUMBER_TO_SYMBOL.get(z, str(z)) for z in mol_data.atomic_nums],
            bonds_typed=mol_data.bonds.tolist(),
        )
        atoms_short.append(mol_data.atomic_nums)
        atom_names_short.append(atom_names)

        # Atoms: extend charges, keep LJ params from the first occurrence.
        for k, v in param_dict["atoms"].items():
            dst = combined["atoms"].setdefault(_to_int(k), {"charges": []})
            dst["charges"].extend(v.get("charges", []))
            dst.setdefault("epsilon", v.get("epsilon"))
            dst.setdefault("sigma", v.get("sigma"))

        # Bonded terms: store both directions for lookup.
        for v in param_dict["bonds"].values():
            i, j = map(_to_int, v["other_idx"])
            for a, b in ((i, j), (j, i)):
                combined["bonds"][f"bond_{a}_{b}"] = {"other_idx": [a, b], "k": v["k"], "length": v["length"]}
        for v in param_dict["angles"].values():
            i, j, k = map(_to_int, v["other_idx"])
            for t in ((i, j, k), (k, j, i)):
                combined["angles"][f"angle_{t[0]}_{t[1]}_{t[2]}"] = {"other_idx": list(t), "k": v["k"], "angle": v["angle"]}
        for v in param_dict["dihedrals"].values():
            q = tuple(map(_to_int, v["other_idx"]))
            for t in (q, q[::-1]):
                combined["dihedrals"][f"dih_{t[0]}_{t[1]}_{t[2]}_{t[3]}"] = {
                    "other_idx": list(t), "periodicity": v["periodicity"], "phase": v["phase"], "k": v["k"],
                }

    return combined, atoms_short, atom_names_short, train_dataset
