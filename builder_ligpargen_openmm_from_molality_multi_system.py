import argparse
import os
import ast
import numpy as np
from hitpoly.writers.box_builder import *
from hitpoly.utils.building_utils import salt_string_to_values, get_concentraiton_from_molality_multi_system
from hitpoly.simulations.openmm_scripts import (
    equilibrate_system_1,
    equilibrate_system_2,
    prod_run_nvt,
    prod_run_tg,
    write_analysis_script,
    equilibrate_system_liquid1,
    equilibrate_system_liquid2,
)
import sys

sys.setrecursionlimit(5000)


def run(
    save_path: str,
    results_path: str,
    smiles:list,
    charge_scale:float,
    salt_type:str,
    molality:float,
    charges:str,
    polynames:list,
    lit_charges_save_path:str,
    system:str,
    simu_temp:float,
    atom_count:int,
    ratios:list,
    ratios_type:str,
    simu_length:int,
    md_save_time:int,
    hitpoly_path:str,
    platform:str='local',
    polymer_chain_length:int=None,
    simu_type="conductivity",
    htvs_env='htvs',
):
    """Run the MD simulation."""
    cuda_device = "0"

    packmol_path = os.environ["packmol"]
    if not hitpoly_path:
        hitpoly_path = f"{os.path.expanduser('~')}/HiTPoly"

    if salt_type:
        salt_smiles, salt_paths, salt_data_paths, ani_name_rdf, concentration = salt_string_to_values(
            hitpoly_path, salt_type, 0)
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

    concentration, solvent_count, repeats, weight_prcnt, total_atoms, poly_name = get_concentraiton_from_molality_multi_system(
        smiles=smiles,
        molality=molality,
        system=system,
        atom_count=atom_count,
        polymer_chain_length=polymer_chain_length,
        ratios=ratios,
        ratios_type=ratios_type,
        salt_smiles='.'.join(salt_smiles),
    )

    with open(f"{save_path}/repeats.txt", "w") as f:
        f.write(str(repeats))

    for ind, i in enumerate(smiles):
        if len(smiles) == 1:
            extra_name = ""
            name = "polymer_conformation.pdb"
            filename_list.append(name)
        else:
            extra_name = f"_{ind}"
            name = f"polymer_conformation{extra_name}.pdb"
            filename_list.append(name)

        ligpargen_path = f"{save_path}/ligpargen{extra_name}"
        print(f"ligpargen path: {ligpargen_path}")
        if not os.path.isdir(ligpargen_path):
            os.makedirs(ligpargen_path)
        
        long_smiles, _ = create_long_smiles(
            i,
            repeats=repeats[ind],
            add_end_Cs=True,
        )
        long_smiles_list.append(long_smiles)

        mol_long = Chem.MolFromSmiles(long_smiles)
        mol_long = Chem.AddHs(mol_long)
        r_long, atom_names_long, atoms_long, bonds_typed_long = generate_atom_types(
            mol_long, 2
        )
        atom_names_long_list.append(atom_names_long)
        atoms_long_list.append(atoms_long)

        (
            ligpargen_repeats,
            smiles_initial,
            mol_initial,
            r,
            atom_names,
            atoms,
            bonds_typed,
        ) = create_ligpargen_short_polymer(
            i,
            add_end_Cs=True,
            reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
            product_index=0,
            atom_names_long=atom_names_long,
        )
        
        mol_initial, _ = create_ligpargen(
            smiles=i,
            repeats=ligpargen_repeats,
            add_end_Cs=True,
            ligpargen_path=ligpargen_path,
            hitpoly_path=hitpoly_path,
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
        )
        # minimize = False
        
        print(f"Saved conformer pdb.")

        if minimize:
            minimize_polymer(
                save_path=save_path,
                long_smiles=long_smiles,
                atoms_long=[atoms_long],
                atoms_short=[atoms],
                atom_names_short=[atom_names],
                atom_names_long=[atom_names_long],
                param_dict=[param_dict],
                lit_charges_save_path=None,
                charges=charges,
                name=name,
                cuda_device=cuda_device,
            )

    if system == "gel":
        box_multiplier = 0.2
    elif system == "liquid":
        box_multiplier = 10
    elif system == "polymer":
        box_multiplier = 1
    else:
        raise ValueError("System must be either gel, liquid or polymer")

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
        polynames=polynames,
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

    if system == "polymer" or system == "gel":
        equilibrate_system_1(
            save_path=save_path,
            final_save_path=final_save_path,
            cuda_device=cuda_device,
        )

        equilibrate_system_2(
            save_path=save_path,
            final_save_path=final_save_path,
            cuda_device=cuda_device,
        )
    else:
        equilibrate_system_liquid1(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_temp=simu_temp,
            cuda_device=cuda_device,
        )
        equilibrate_system_liquid2(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_temp=simu_temp,
            cuda_device=cuda_device,
        )

    if simu_type == "conductivity":
        prod_run_nvt(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_temp=simu_temp,
            mdOutputTime=md_save_time,
            simu_time=simu_length,
            cuda_device=cuda_device,
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
            poly_name=','.join(poly_name),
            hitpoly_path=hitpoly_path,
            htvs_env=htvs_env,
        )

    elif simu_type.lower() == "tg":
        prod_run_tg(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_time=10,
            start_temperature=500,
            end_temperature=100,
            temperature_step=20,
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
        help="Path of the file that contains the smiles strings to be created, if more than one, have them '.' separated, and the second line should be the mole fraction ratios of the smiles strings, ex, 'C1.C2.C3', '0.5,0.3,0.2'",
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
        "-f",
        "--hitpoly_path",
        help="Path towards the HiTPoly folder",
        default="None",
    )
    parser.add_argument("--temperature", help="Simulation temperature", default="430")
    parser.add_argument("--simu_length", help="Simulation length, ns", default="100")
    parser.add_argument(
        "--platform", help="For which platform to build the files for", default="local"
    )
    parser.add_argument(
        "-system",
        "--system",
        help="System to be used for the simulation",
        default="polymer",
    )
    parser.add_argument(
        "-atom_count",
        "--atom_count",
        help="Atom count of the system",
        default="None",
    )
    parser.add_argument(
        "-md_save_time",
        "--md_save_time",
        help="MD save time",
        default="10000",
    )
    parser.add_argument(
        "-polymer_chain_length",
        "--polymer_chain_length",
        help="Polymer chain length",
        default="None",
    )
    parser.add_argument(
        "--simu_type",
        help="What type of simulation to perform, options [conductivity, tg]}",
        default="conductivity",
    )
    parser.add_argument(
        "--htvs_env",
        help="HTVS environment",
        default="htvs",
    )
    args = parser.parse_args()

    if args.hitpoly_path == "None":
        args.hitpoly_path = None

    # The file from which the smiles strings are read should have the following format:
    # [smiles_string_1].[smiles_string_2].[smiles_string_3] 
    # [ratio_1],[ratio_2],[ratio_3]
    # [ratios_type] # either 'mol' or 'weight', default is 'mol'
    with open(args.smiles_path, "r") as f:
        lines = f.readlines()
        smiles = lines[0].split(".")
        if len(smiles) > 1:
            ratios = np.array(ast.literal_eval(lines[1]))
            ratios = (ratios / ratios.sum()).tolist()
            if len(lines) > 2:
                ratios_type = lines[2].strip()
            else:
                ratios_type = 'mol'
            assert len(smiles) == len(ratios)
        else:
<<<<<<< HEAD
            mol_ratios = None
        if len(lines) == 3:
            """
            3rd line is the DFT charge CSV file name for each molecule.
            This line is optional.
            """
            polynames = lines[2].split(",")
=======
            ratios = None
            ratios_type = None
>>>>>>> origin/main
    
    if args.atom_count == "None":
        atom_count = None
    else:
        atom_count = int(args.atom_count)

    if args.polymer_chain_length == "None":
        polymer_chain_length = None
    else:
        polymer_chain_length = int(args.polymer_chain_length)
    if args.charge_type == "LIT":
        print("polynames: ", polynames)
    else:
        polynames = None

    run(
        save_path=args.save_path,
        results_path=args.results_path,
        smiles=smiles,
        charge_scale=float(args.charge_scale),
        salt_type=args.salt_type,
        molality=float(args.molality_salt),
        charges=args.charge_type,
        polynames=polynames,
        lit_charges_save_path=args.lit_charges_save_path,
        system=args.system,
        atom_count=atom_count,
        md_save_time=int(args.md_save_time),
        simu_temp=int(args.temperature),
        simu_length=int(args.simu_length),
        platform=args.platform,
        polymer_chain_length=polymer_chain_length,
        ratios=ratios,
        ratios_type=ratios_type,
        hitpoly_path=args.hitpoly_path,
        simu_type=args.simu_type,
        htvs_env=args.htvs_env,
    )
