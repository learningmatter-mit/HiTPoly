import argparse
import os
import time
import shutil
import uuid
from hitpoly.writers.box_builder import *
from hitpoly.utils.building_utils import salt_string_to_values
from hitpoly.simulations.gromacs_writer import GromacsWriter
from hitpoly.simulations.openmm_scripts import (
    equilibrate_system_1,
    equilibrate_system_2,
    prod_run_nvt,
    write_analysis_script,
)
from distutils.dir_util import copy_tree
import sys

sys.setrecursionlimit(5000)


def run(
    save_path,
    results_path,
    smiles,
    molality=1.0,
    charge_scale=0.75,
    polymer_count=30,
    charges="LPG",
    add_end_Cs=True,
    md_save_time=10000,
    hitpoly_path=None,
    timestep=0.002,
    htvs_path=None,
    salt_type=None,
    lit_charges_save_path=None,
    charges_path=None,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    box_multiplier=2,
    enforce_generation=False,
    simu_type="conductivity",
    simu_temp=430,
    simu_length=100,
    platform="local",
):
    # don't forget to export the path to your packmol in the bashrc
    packmol_path = os.environ["packmol"]
    if not hitpoly_path:
        hitpoly_path = f"{os.path.expanduser('~')}/HiTPoly"

    htvs_details = {}
    # htvs_details["geom_config_name"] = "nvt_conf_generation_ligpargen_lammps"
    if salt_type:
        salt_smiles, salt_paths, salt_data_paths, ani_name_rdf = salt_string_to_values(hitpoly_path, salt_type)
        salt = True
    else:
        salt = False
        salt_paths = []
        salt_data_paths = []
        salt_smiles = []
    concentration, repeats = get_concentration_from_molality(
        molality=molality,
        polymer_count=polymer_count,
        smiles=smiles,
        add_end_Cs=add_end_Cs
    )
    with open(f"{save_path}/repeats.txt", "w") as f:
        f.write(str(repeats))

    ligpargen_path = f"{save_path}/ligpargen"
    print(f"ligpargen path: {ligpargen_path}")
    if not os.path.isdir(ligpargen_path):
        os.makedirs(ligpargen_path)

    long_smiles, _ = create_long_smiles(
        smiles,
        repeats=repeats,
        add_end_Cs=add_end_Cs,
        reaction=reaction,
        product_index=product_index,
    )

    mol_long = Chem.MolFromSmiles(long_smiles)
    mol_long = Chem.AddHs(mol_long)
    r_long, atom_names_long, atoms_long, bonds_typed_long = generate_atom_types(
        mol_long, 2
    )

    (
        ligpargen_repeats,
        smiles_initial,
        mol_initial,
        r,
        atom_names,
        atoms,
        bonds_typed,
    ) = create_ligpargen_short_polymer(
        smiles,
        add_end_Cs=add_end_Cs,
        reaction=reaction,
        product_index=product_index,
        atom_names_long=atom_names_long,
    )

    mol_initial, smiles_initial = create_ligpargen(
        smiles=smiles,
        repeats=ligpargen_repeats,
        add_end_Cs=add_end_Cs,
        ligpargen_path=ligpargen_path,
        hitpoly_path=hitpoly_path,
        reaction=reaction,
        product_index=product_index,
        platform=platform,
    )

    print(f"Created ligpargen files at {ligpargen_path}")

    r, atom_names, atoms, bonds_typed = generate_atom_types(mol_initial, 2)

    param_dict = generate_parameter_dict(ligpargen_path, atom_names, atoms, bonds_typed)

    if ligpargen_repeats == repeats:
        long_smiles = smiles
        filename = "polymer_conformation.pdb"
        shutil.move(f"{ligpargen_path}/PLY.pdb", f"{save_path}/{filename}")
        minimize = False

    if not ligpargen_repeats == repeats:
        filename, mol, minimize = create_conformer_pdb(
            save_path,
            long_smiles,
            name="polymer_conformation",
            enforce_generation=enforce_generation,
        )
    print(f"Saved conformer pdb.")

    minimize_polymer(
        short_smiles=smiles_initial,
        save_path=save_path,
        long_smiles=long_smiles,
        atoms_long=atoms_long,
        atoms_short=atoms,
        atom_names_short=atom_names,
        atom_names_long=atom_names_long,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        htvs_path=htvs_path,
        htvs_details=htvs_details,
    )

    create_box_and_ff_files_openmm(
        short_smiles=smiles_initial,
        save_path=save_path,
        long_smiles=long_smiles,
        filename=filename,
        polymer_count=polymer_count,
        concentration=concentration,
        packmol_path=packmol_path,
        atoms_long=atoms_long,
        atoms_short=atoms,
        atom_names_short=atom_names,
        atom_names_long=atom_names_long,
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        charge_scale=charge_scale,
        htvs_path=htvs_path,
        htvs_details=htvs_details,
        salt_smiles=salt_smiles,
        salt_paths=salt_paths,
        salt_data_paths=salt_data_paths,
        box_multiplier=box_multiplier,
        salt=salt,
    )

    final_save_path = f"{save_path}/openmm_saver"
    if not os.path.isdir(final_save_path):
        os.makedirs(final_save_path)

    equilibrate_system_1(
        save_path=save_path,
        final_save_path=final_save_path,
    )

    equilibrate_system_2(
        save_path=save_path,
        final_save_path=final_save_path,
    )

    prod_run_nvt(
        save_path=save_path,
        final_save_path=final_save_path,
        simu_temp=simu_temp,
        mdOutputTime=md_save_time,
        simu_time=simu_length,
        timestep=timestep,
    )

    write_analysis_script(
        save_path=save_path,
        results_path=results_path,
        platform=platform,
        repeat_units=repeats,
        simu_temperature=simu_temp,
        prod_run_time=simu_length,
        ani_name_rdf=ani_name_rdf,
        cation=salt_type.split(".")[0],
        anion=ani_name_rdf.split(",")[0],
        xyz_output=int(md_save_time*timestep),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxbuilder for OpenMM simulations")
    parser.add_argument(
        "-p", "--save_path", help="Path to the where the new directory to be created"
    )
    parser.add_argument(
        "-pr",
        "--results_path",
        help="Path to the where the new directory to be created",
    )
    parser.add_argument(
        "-s",
        "--smiles_path",
        help="Smiles of the polymer to be created, of the form [Cu]*[Au]",
    )
    parser.add_argument(
        "-pc",
        "--polymer_count",
        help="How many polymer chains or molecules to be packed",
        default="30",
    )
    parser.add_argument(
        "--salt_type",
        help="Type of the salt to be added to the simulation",
        default="Li.TFSI",
    )
    parser.add_argument(
        "-M",
        "--molality_salt",
        help="Molality of salt in mol/kg",
        default="1",
    )
    parser.add_argument(
        "-cs",
        "--charge_scale",
        help="To what value the charges of the salts be scaled",
        default="0.75",
    )
    parser.add_argument(
        "-ct", "--charge_type", help="What type of charges to select", default="LPG"
    )
    parser.add_argument(
        "-ecs",
        "--end_carbons",
        help="When creating polymer if end carbons should be added",
        default="True",
    )
    parser.add_argument(
        "-f",
        "--hitpoly_path",
        help="Path towards the HiTPoly folder",
        default="None",
    )
    parser.add_argument(
        "-react",
        "--reaction",
        help="Reaction that creates the polymer",
        default="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    )
    parser.add_argument(
        "-pi",
        "--product_index",
        help="Product index which to use for the smarts reaction",
        default="0",
    )
    parser.add_argument(
        "-box",
        "--box_multiplier",
        help="PBC box size multiplier for packmol, poylmers 1, other molecules 4-10",
        default="1",
    )
    parser.add_argument(
        "-conf",
        "--enforce_generation",
        help="Whether to force rdkit to create a conformation",
        default="False",
    )
    parser.add_argument(
        "--simu_type",
        help="What type of simulation to perform, options [conductivity, tg]}",
        default="conductivity",
    )
    parser.add_argument(
        "-md_save_time",
        "--md_save_time",
        help="MD save time",
        default="10000",
    )
    parser.add_argument("--temperature", help="Simulation temperature", default="430")
    parser.add_argument("--simu_length", help="Simulation length, ns", default="100")
    parser.add_argument(
        "--platform", help="For which platform to build the files for", default="local"
    )

    args = parser.parse_args()
    if args.end_carbons == "false" or args.end_carbons == "False":
        add_end_Cs = False
    else:
        add_end_Cs = True

    if args.hitpoly_path == "None":
        args.hitpoly_path = None
    if args.enforce_generation == "False":
        args.enforce_generation = False
    else:
        args.enforce_generation = True
    if args.salt_type == "None":
        args.salt_type = None

    with open(args.smiles_path, "r") as f:
        lines = f.readlines()
        smiles = lines[0]
    run(
        save_path=args.save_path,
        results_path=args.results_path,
        smiles=smiles,
        charge_scale=float(args.charge_scale),
        polymer_count=int(args.polymer_count),
        molality=float(args.molality_salt),
        charges=args.charge_type,
        add_end_Cs=add_end_Cs,
        md_save_time=int(args.md_save_time),
        reaction=args.reaction,
        product_index=int(args.product_index),
        box_multiplier=float(args.box_multiplier),
        enforce_generation=args.enforce_generation,
        simu_type=args.simu_type,
        salt_type=args.salt_type,
        simu_temp=int(args.temperature),
        simu_length=int(args.simu_length),
        platform=args.platform,
    )
