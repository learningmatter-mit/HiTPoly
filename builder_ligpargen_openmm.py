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
    equilibrate_system_liquid1,
    equilibrate_system_liquid2,
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
    charge_scale=0.75,
    solvent_count=[30],
    repeats=1,
    ligpargen_repeats=1,
    salt_type="Li.TFSI",
    concentration=100,
    charges="LPG",
    add_end_Cs=True,
    hitpoly_path=None,
    lit_charges_save_path=None,
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    box_multiplier=2,
    enforce_generation=False,
    simu_temp=430,
    simu_length=100,
    md_save_time=12500,
    platform="local",
    is_liquid=False
):
    """
    is_liquid (boolean) : after "equilibrate_system_1", 
    whether to use second equilibration step as "equilibrate_system_liquid"
    or "equilibrate_system_2"
    liquids do not need high pressure equilibration
    """
    # don't forget to export the path to your packmol in the bashrc
    packmol_path = os.environ["packmol"]
    if not hitpoly_path:
        hitpoly_path = f"{os.path.expanduser('~')}/HiTPoly"

    if salt_type:
        salt_smiles, salt_paths, salt_data_paths, ani_name_rdf, concentration = salt_string_to_values(
            hitpoly_path, salt_type, concentration)
        salt = True
    else:
        salt = False
        salt_paths = []
        salt_data_paths = []
        salt_smiles = []
        ani_name_rdf = None
        
    filename_list = []
    long_smiles_list = []
    atom_names_short_list = []
    atom_names_long_list = []
    atoms_short_list = []
    atoms_long_list = []
    param_dict_list = []

    for ind, i in enumerate(smiles):
        if len(smiles) == 1:
            extra_name = ""
            name = "polymer_conformation.pdb"
            filename_list.append(name)
        else:
            extra_name = f"_{ind}"
            name = f"polymer_conformation{extra_name}.pdb"
            filename_list.append(name)

        with open(f"{save_path}/repeats{extra_name}.txt", "w") as f:
            f.write(str(repeats))

        ligpargen_path = f"{save_path}/ligpargen{extra_name}"
        print(f"ligpargen path: {ligpargen_path}")
        if not os.path.isdir(ligpargen_path):
            os.makedirs(ligpargen_path)
        
        long_smiles, _ = create_long_smiles(
            i,
            repeats=repeats,
            add_end_Cs=add_end_Cs,
            reaction=reaction,
            product_index=product_index,
        )
        long_smiles_list.append(long_smiles)

        mol_long = Chem.MolFromSmiles(long_smiles)
        mol_long = Chem.AddHs(mol_long)
        r_long, atom_names_long, atoms_long, bonds_typed_long = generate_atom_types(
            mol_long, 2
        )
        atom_names_long_list.append(atom_names_long)
        atoms_long_list.append(atoms_long)
        
        mol_initial, _ = create_ligpargen(
            smiles=i,
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
        atom_names_short_list.append(atom_names)
        atoms_short_list.append(atoms)

        param_dict = generate_parameter_dict(ligpargen_path, atom_names, atoms, bonds_typed)
        param_dict_list.append(param_dict)
        
        minimize = create_conformer_pdb(
            save_path,
            long_smiles,
            name=name,
            enforce_generation=enforce_generation,
        )
        
        print(f"Saved conformer pdb.")

        if minimize:
            minimize_polymer(
                save_path=save_path,
                long_smiles=long_smiles,
                atoms_long=atoms_long,
                atoms_short=atoms,
                atom_names_short=atom_names,
                atom_names_long=atom_names_long,
                param_dict=param_dict,
                lit_charges_save_path=lit_charges_save_path,
                charges=charges,
                name=name,
            )

    create_box_and_ff_files_openmm(
        save_path=save_path,
        long_smiles=long_smiles_list,
        filename=filename_list,
        concentration=concentration,
        solvent_count=solvent_count,
        packmol_path=packmol_path,
        atoms_long=atoms_long_list,
        atoms_short=atoms_short_list,
        atom_names_short=atom_names_short_list,
        atom_names_long=atom_names_long_list,
        param_dict=param_dict_list,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        charge_scale=charge_scale,
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

    if not is_liquid:

        equilibrate_system_1(
            save_path=save_path,
            final_save_path=final_save_path,
        )

        equilibrate_system_2(
            save_path=save_path,
            final_save_path=final_save_path,
        )
    else:
        equilibrate_system_liquid1(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_temp=simu_temp,
        )
        equilibrate_system_liquid2(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_temp=simu_temp,
        )

    prod_run_nvt(
        save_path=save_path,
        final_save_path=final_save_path,
        simu_temp=simu_temp,
        mdOutputTime=md_save_time,
        simu_time=simu_length,
    )

    write_analysis_script(
        save_path=save_path,
        results_path=results_path,
        platform=platform,
        repeat_units=repeats,
        cation=salt_type.split(".")[0],
        anion=ani_name_rdf.split(",")[0],
        simu_temperature=simu_temp,
        prod_run_time=simu_length,
        ani_name_rdf=ani_name_rdf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxbuilder for OpenMM simulations")
    parser.add_argument(
        "-p", "--save_path", help="Path to the where the new directory to be created"
    )
    parser.add_argument(
        "-pr",
        "--results_path",
        help="Path to the where the new directory to be created for results",
    )
    parser.add_argument(
        "-s",
        "--smiles_path",
        help="Either the path to a file that contains the smiles string of the polymer (or small molecule), \
            or the smiles string itself. Has to be of the form [Cu]*[Au] if polymer, otherwise can be \
            a rdkit canonical smiles string. The file can contain multiple smiles strings, one per line.",
    )
    parser.add_argument(
        "-sc",
        "--solvent_count",
        help="How many solvent molecules (polymer or small molecules) to be packed, can be a list of values, ex, '30,30'",
        default="30",
    )
    parser.add_argument(
        "--repeats",
        help="How many repeat units in the final polymer chain, can be a list of values, ex, '50,45'",
        default="50",
    )
    parser.add_argument(
        "--ligpargen_repeats",
        help="How many repeat units in the initial polymer chain to be parametrized (max of 200 atoms), can be a list of values, ex, '3,3'",
        default="3",
    )
    parser.add_argument(
        "--salt_type",
        help="Type of the salt to be added to the simulation",
        default="Li.TFSI",
    )
    parser.add_argument(
        "--concentration",
        help="Concentration of the salt",
        default="100",
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
        help="PBC box size multiplier for packmol, poylmers <1, other molecules (solvents) 4-10",
        default="0.5",
    )
    parser.add_argument(
        "-conf",
        "--enforce_generation",
        help="Whether to force rdkit to create a conformation",
        default="False",
    )
    parser.add_argument("--temperature", help="Simulation temperature", default="415")
    parser.add_argument("--simu_length", help="Simulation length, ns", default="100")
    parser.add_argument(
        "--md_save_time", help="Simulation length, ns", default="12500"
    )
    parser.add_argument(
        "--platform", help="For which platform to build the files for", default="local"
    )
    parser.add_argument(
        "--is_liquid", help="Whether to use second equilibration step as 'equilibrate_system_liquid'", default="False"
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

    smiles = []
    if os.path.isfile(args.smiles_path):
        with open(args.smiles_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                smiles.append(line.strip())
    else:
        smiles = args.smiles_path

    solvent_count = args.solvent_count.split(",")
    solvent_count = [int(i) for i in solvent_count]

    run(
        save_path=args.save_path,
        results_path=args.results_path,
        smiles=smiles,
        charge_scale=float(args.charge_scale),
        solvent_count=solvent_count,
        concentration=int(args.concentration),
        repeats=int(args.repeats),
        ligpargen_repeats=int(args.ligpargen_repeats),
        salt_type=args.salt_type,
        charges=args.charge_type,
        add_end_Cs=add_end_Cs,
        hitpoly_path=args.hitpoly_path,
        reaction=args.reaction,
        product_index=int(args.product_index),
        box_multiplier=float(args.box_multiplier),
        enforce_generation=args.enforce_generation,
        simu_temp=int(args.temperature),
        simu_length=int(args.simu_length),
        md_save_time=int(args.md_save_time),
        platform=args.platform,
        is_liquid=bool(args.is_liquid)
    )
