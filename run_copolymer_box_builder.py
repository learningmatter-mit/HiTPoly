import argparse
import os
import time
import shutil
import uuid
from hitpoly.writers.copolymer_box_builder import *
from hitpoly.simulations.gromacs_writer import GromacsWriter
from hitpoly.simulations.openmm_scripts import (
    equilibrate_system_1,
    equilibrate_system_2,
    prod_run_nvt,
    write_analysis_script,
    prod_run_tg,
)
from distutils.dir_util import copy_tree
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(5000)

def run(
    results_path = None,
    final_path = None, 
    monomers = ["[Cu]CC(c1ccccc1)[Au]","[Cu]CC(C#N)[Au]", "[Cu]CC=CC[Au]"],
    molality = 1.0,
    charge_scale=0.75,
    polymer_count=1, 
    total_repeats = 15, #total number of monomers in chain
    fractions = [0.8,0.1,0.1],
    polymerization_mode = 'random',
    concentration: list = [0, 0],
    poly_name="PEO",
    charges="LPG",
    add_end_Cs=True,
    hitpoly_path = None,
    htvs_path = None,
    salt_smiles=None,
    salt_paths=None,
    salt_data_paths=None,
    lit_charges_save_path = None,
    reaction = "[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
    box_multiplier=1,
    enforce_generation=False,
    salt=True,
    simu_type="conductivity",
    simu_temp=430,
    simu_length=100,
    platform="supercloud",
    num_blocks=None,
    arms = None,
    blend_mode = False,
    polymer_chain_length = 1000,
    single_ion_conductor=True,     
    anion="N",                    
    cation="Na"   
):

    # Set default paths if not provided
    home_dir = os.path.expanduser("~")
    if results_path is None:
        results_path = os.path.join(home_dir, "results")
    if final_path is None:
        final_path = os.path.join(results_path, "final_results")
    if hitpoly_path is None:
        hitpoly_path = os.path.join(home_dir, "HiTPoly")
    if htvs_path is None:
        htvs_path = os.path.join(home_dir, "htvs")

    # don't forget to export the path to your packmol in the bashrc
    packmol_path = os.environ["packmol"]
    #packmol_path = "/home/trios/packmol-20.14.4/packmol"

    htvs_details = {
        "geom_config_name": "nvt_conf_generation_ligpargen_lammps",
        "calc_method_name": "dft_wb97xd3_def2tzvp",
        "species_group_name": "singleion_trios", 
    }

    # Define a dictionary to map atomic numbers to atomic symbols
    number_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        # Add more elements as needed
    }

    if charges == "LIT":
        # TEMP ONLY FOR PEO
        lit_charges_save_path = (
            "/home/trios/HiTPoly/data/forcefield_files/PEO_KOZ_charges"
        )
        if not lit_charges_save_path:
            raise ValueError("A path for literature/custom chargers must be defined")

    if salt:
        if not salt_smiles and not salt_paths and not salt_data_paths:
            salt_path = f"{hitpoly_path}/data/pdb_files"

            salt_smiles = [
                f"[{cation}+]", "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",]

            salt_paths = [
                f"{salt_path}/geometry_file_{cation}.pdb",
                f"{salt_path}/geometry_file_TFSI.pdb",
            ]

            salt_data_paths = [
                f"{hitpoly_path}/data/forcefield_files/lammps_{cation}_q100.data",
                f"{hitpoly_path}/data/forcefield_files/lammps_TFSI_q100.data",
            ]

        elif not salt_smiles or not salt_paths or not salt_data_paths:
            raise ValueError(
                "Must have an input all of these: salt_smiles, salt_paths and salt_data_paths or none."
            )
    else:
        salt_paths = []
        salt_data_paths = []
        salt_smiles = []

    # Define folder structure
    date_name = time.strftime("%y%m%d_H%HM%M")
    conc_name = int(np.array(concentration).mean())
    extra_naming = f"{poly_name}_N{total_repeats}_PC{polymer_count}_C{conc_name}_q0{int(charge_scale*100)}_q{charges}_"
    
    if not salt:
        extra_naming = f"{poly_name}_N{total_repeats}_PC{polymer_count}_q{charges}_"
    folder_name = f"{extra_naming}{date_name}"
    save_path = f"{results_path}/{folder_name}"
    ligpargen_path = f"{save_path}/ligpargen"

    # Ensure directories exist
    os.makedirs(ligpargen_path, exist_ok=True)
    os.makedirs(final_path, exist_ok=True)

    print(f"Results path: {results_path}")
    print(f"Save path: {save_path}")
    print(f"LigParGen path: {ligpargen_path}")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isdir(ligpargen_path):
        os.makedirs(ligpargen_path)

    concentration, repeats = get_concentration_from_molality(
        save_path=save_path,
        molality=molality,
        polymer_count=polymer_count,
        monomers=monomers,
        fractions=fractions,
        add_end_Cs=add_end_Cs,
        polymerization_mode=polymerization_mode,
        num_blocks=num_blocks,
        polymer_chain_length=polymer_chain_length,
        arms=arms,
        cation=cation,
        blend_mode=True if blend_mode else False
    )

    print(f"Final concentration: {concentration}")

    print("Creating Oligomers")
    oligomers_dict, oligomers_list = create_oligomers(monomers, polymerization_mode, arms=arms)
    print("Dictionary of Oligomers:", oligomers_dict)
    print("List of Oligomers:", oligomers_list)

    print("Creating Chains")
    long_smiles_list = []
    filenames = []

    MAX_RETRIES = 10  # Adjust as needed

    long_smiles_list = []
    filenames = []
    chains_to_pack = {}

    if blend_mode:
        print("Generating blend of homopolymers...")

        for idx, (monomer, total_repeats) in enumerate(zip(monomers, repeats)):
            num_chains = int(round(fractions[idx] * polymer_count))  # Split chains
            print(f"Generating {num_chains} chains of monomer {idx}: {monomer}")

            success = False
            retries = 0
            while not success and retries < MAX_RETRIES:
                try:
                    long_smiles, _ = create_long_smiles(
                        save_path=save_path,
                        smiles=[monomer],
                        fractions=[1.0],
                        total_repeats=total_repeats,
                        add_end_Cs=add_end_Cs,
                        reaction=reaction,
                        product_index=product_index,
                        polymerization_mode="homopolymer"
                    )

                    filename, mol, minimize = create_conformer_pdb(
                        path_to_save=save_path,
                        smiles=long_smiles,
                        name=f"polymer_conformation_{idx}",
                        enforce_generation=False
                    )

                    if filename is None or mol is None:
                        retries += 1
                        continue

                    long_smiles_list.append(long_smiles)
                    filenames.append(filename)
                    chains_to_pack[filename] = num_chains
                    success = True

                except Exception as e:
                    print(f"Error: {e}")
                    retries += 1

            if not success:
                raise RuntimeError(f"Failed to generate homopolymer {idx}")

    elif polymerization_mode == "homopolymer":
        print("Generating multiple chains of a single homopolymer...")
        print(f"monomers: {monomers}")
        print(f"fractions: {fractions}")
        print(f"repeats: {repeats}")


        success = False
        retries = 0
        while not success and retries < MAX_RETRIES:
            try:
                long_smiles, _ = create_long_smiles(
                    save_path=save_path,
                    smiles=monomers,
                    fractions=fractions,
                    total_repeats=repeats,
                    add_end_Cs=add_end_Cs,
                    reaction=reaction,
                    product_index=product_index,
                    polymerization_mode="homopolymer"
                )

                filename, mol, minimize = create_conformer_pdb(
                    path_to_save=save_path,
                    smiles=long_smiles,
                    name="polymer_conformation",
                    enforce_generation=False
                )

                if filename is None or mol is None:
                    print("Retrying due to catastrophic failure.")
                    retries += 1
                    continue

                long_smiles_list = [long_smiles]
                filenames = [filename]
                chains_to_pack[filename] = polymer_count
                success = True

            except Exception as e:
                print(f"Error in homopolymer generation: {str(e)}. Retrying...")
                retries += 1

        if not success:
            raise RuntimeError("Failed to generate homopolymer after multiple attempts.")
            
    elif polymerization_mode == "random":
        # stochastic polymer generation
        for i in range(polymer_count):
            success = False
            retries = 0
            while not success and retries < MAX_RETRIES:
                try:
                    long_smiles, _ = create_long_smiles(
                        save_path,
                        monomers,
                        fractions,
                        repeats,
                        add_end_Cs,
                        reaction,
                        product_index,
                        polymerization_mode
                    )

                    if long_smiles in long_smiles_list:
                        print(f"Duplicate SMILES: {long_smiles}. Retrying...")
                        retries += 1
                        continue

                    filename, mol, minimize = create_conformer_pdb(
                        path_to_save=save_path,
                        smiles=long_smiles,
                        name=f"polymer_conformation_{i}",
                        enforce_generation=False
                    )

                    if filename is None or mol is None:
                        print("Retrying due to failure.")
                        retries += 1
                        continue

                    long_smiles_list.append(long_smiles)
                    filenames.append(filename)
                    success = True

                except Exception as e:
                    print(f"Random gen error: {str(e)}. Retrying...")
                    retries += 1

            if not success:
                print(f"Skipping chain {i} due to repeated failures.")

    else:
        # All other non-homopolymer, non-blend modes (block, alternating, star, etc.)
        success = False
        retries = 0
        while not success and retries < MAX_RETRIES:
            try:
                long_smiles, _ = create_long_smiles(
                    save_path,
                    monomers,
                    fractions,
                    repeats,
                    add_end_Cs,
                    reaction,
                    product_index,
                    polymerization_mode,
                    num_blocks=num_blocks if polymerization_mode == 'block' else None,
                    arms=arms if polymerization_mode == 'star' else None,
                )

                filename, mol, minimize = create_conformer_pdb(
                    path_to_save=save_path,
                    smiles=long_smiles,
                    name="polymer_conformation",
                    enforce_generation=False
                )

                if filename is None or mol is None:
                    print("Retrying due to catastrophic failure.")
                    retries += 1
                    continue

                long_smiles_list.append(long_smiles)
                filenames.append(filename)
                success = True

            except ValueError as e:
                print(f"Error in generation: {str(e)}. Retrying...")
                retries += 1

            except Exception as e:
                print(f"Unexpected failure: {str(e)}. Retrying...")
                retries += 1

        if not success:
            raise RuntimeError("Failed to generate non-random polymer.")

    # Printout for all modes
    print(f"Final FILENAMES HERE: {filenames}")
    print(f"Final long smiles: {long_smiles_list}")
    print(f"Final length filenames: {len(filenames)}")
    print(f"Final length long smiles: {len(long_smiles_list)}")

    # Save the full raw Python list of long SMILES
    with open(os.path.join(save_path, "final_long_smiles.txt"), "w") as f:
        f.write(str(long_smiles_list))
    print(f"Saved raw SMILES list to {os.path.join(save_path, 'final_long_smiles.txt')}")


    #Generate combined source dictionary which runs ligpargen on each oligomer in a loop and gets atom types from 
    # opology dataset to then run param dictionary and combine everything.
    combined_param_dict, atoms_short, atom_names_short = create_combined_param_dict(oligomers_list, ligpargen_path, hitpoly_path, platform)

    #Update the parameter dictionary to use the combined parameters
    param_dict = combined_param_dict

    # Combine oligomer SMILES and long SMILES into a single list
    combined_smiles_list = long_smiles_list + oligomers_list

    print(f"COMBINED SMILES {combined_smiles_list}")

    #Create a new dataset with all oligomers and the long molecule
    train_args = hitpolyArgs()
    train_molecule_data = [
        TopologyBuilder(
            smiles=[i], train_args=train_args, load_geoms=False, lj_from_file=False
        )
        for i in combined_smiles_list
    ]
    train_args.discrete_flag = True  # Set the discrete_flag to True
    train_dataset = TopologyDataset(data=train_molecule_data, train_args=train_args)

    #Initialize lists to store attributes for all long molecules
    all_atom_names_long = []
    all_atoms_long = []
    all_bonds_typed_long = []
    all_angles = []
    all_atoms_symbols = []

    #Collect attributes for each long molecule in the dataset
    #index all molecules in train_dataset first to ensure all molecules are unique
    smiles_to_index = {molecule.smiles[0]: idx for idx, molecule in enumerate(train_dataset.data)}

    for long_smiles in long_smiles_list:
        if long_smiles in smiles_to_index:
            molecule = train_dataset.data[smiles_to_index[long_smiles]]
            atom_names_long = molecule.atom_type.tolist()
            atoms_long = molecule.atomic_nums
            bonds_typed_long = molecule.bonds.tolist()
            angles = molecule.angles.tolist()
            
            all_atom_names_long.append(atom_names_long)
            all_atoms_long.append(atoms_long)
            all_bonds_typed_long.append(bonds_typed_long)
            all_angles.append(angles)
        else:
            raise ValueError(f"SMILES {long_smiles} not found in train_dataset")
                
    #Convert atomic numbers to atomic symbols for each sublist in all_atoms_long
    all_atoms_symbols = [[number_to_symbol[number] for number in sublist] for sublist in all_atoms_long]

    # Print to verify
    print(f"Atomic symbols for long molecules: {all_atoms_symbols}")


    # Now, all_atom_names_long, all_atoms_long, all_bonds_typed_long, and all_angles
    # contain the attributes for each long molecule in the dataset
    print(f"Atom Names (Long Molecules): {all_atom_names_long}")
    print(f"Atoms (Long Molecules): {all_atoms_long}")
    print(f"Bonds Typed (Long Molecules): {all_bonds_typed_long}")
    print(f"Angles (Long Molecules): {all_angles}")

    if single_ion_conductor:
        #This works for homopolymer just pull the one smiles from oligomer list
        my_smiles=oligomers_list[0]
    
        #patch the parameter dictionary with the updated charges from dft
        patched_param_dict = patch_params_with_dft_charges_from_db(
            smiles=my_smiles,
            original_param_dict=combined_param_dict,
            htvs_path=htvs_path,
            htvs_details=htvs_details
        )
        
        #reset parm_dict to patched!
        param_dict = patched_param_dict
    
        # Count the number of anionic [N-] groups in one polymer from the SMILES
        n_anions_per_polymer = long_smiles_list[0].count(f"[{anion}-]")
    
        concentration, _ = get_concentration_from_charge_neutrality(
            atom_names_long=all_atom_names_long,
            param_dict=param_dict,
            polymer_count=polymer_count,
            cation_charge=charge_scale,  # Usually 0.75
            single_ion_conductor=single_ion_conductor,
            anionic_atom_count_per_polymer=n_anions_per_polymer
        )

    # Determine polymer_count list correctly
    if blend_mode:
        # Blend: different structures with different counts
        polymer_count = [chains_to_pack[os.path.basename(f)] for f in filenames]

    elif polymerization_mode in ["homopolymer", "block", "alternating", "star"]:
        # Single structure replicated N times
        if len(filenames) != 1:
            raise ValueError(f"{polymerization_mode} mode expects exactly one filename.")
        polymer_count = [polymer_count]  # replicate this one chain N times

    else:
        # Random architecture: each chain is unique, used once
        polymer_count = [1] * len(filenames)

    #Create box and force field files
    create_box_and_ff_files_openmm(
        short_smiles=oligomers_list,
        save_path=save_path,
        long_smiles=long_smiles_list, #list of all the long smiles in the box
        filename=filenames,
        polymer_count=polymer_count,  # Set the polymer count as needed
        concentration=concentration,  # Set the concentration as needed
        packmol_path=packmol_path,  # Set the path to Packmol
        atoms_long=all_atoms_symbols, #list of lists with the atoms of each of the polymers packed
        atoms_short=atoms_short,
        atom_names_short=atom_names_short,
        atom_names_long=all_atom_names_long, #list of list with atom names of each polymer packed
        param_dict=param_dict,
        lit_charges_save_path=lit_charges_save_path,
        charges=charges,
        charge_scale=charge_scale,  # Set the charge scale as needed
        htvs_path=htvs_path,
        htvs_details=htvs_details,
        salt_smiles=salt_smiles,  # Set the salt SMILES as needed
        salt_paths=salt_paths,  # Set the salt paths as needed
        salt_data_paths=salt_data_paths,  # Set the salt data paths as needed
        box_multiplier=box_multiplier,  # Set the box multiplier as needed
        salt=salt,  # Set whether to include salt
        single_ion_conductor = single_ion_conductor,
    )
    #NEEDS DEBUGGING
    if polymerization_mode in ["block", "alternating"]:
        write_atom_names_rdf_from_pdb(
            f"{save_path}/packed_box.pdb",
            f"{save_path}/atom_names_rdf.txt"
        )

        write_atom_labels_from_log(
            atom_names_file=f"{save_path}/atom_names_rdf.txt",
            log_file=f"{save_path}/final_polymer_details.txt",
            output_file=f"{save_path}/atom_names_all.rdf.txt"
        )

    final_save_path = f"{save_path}/openmm_saver"
    if not os.path.isdir(final_save_path):
        os.makedirs(final_save_path)

    if simu_type == "conductivity":
    
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
            simu_time=simu_length,
        )
    
        write_analysis_script(
            save_path=save_path,
            results_path=results_path,
            repeat_units=repeats,
            cation=cation,
            anion=anion,
            platform=platform,
            simu_temperature=simu_temp,
            prod_run_time=simu_length,
            xyz_output=25,
            ani_name_rdf=None
        )

    elif simu_type == "tg":
        prod_run_tg(
            save_path=save_path,
            final_save_path=final_save_path,
            simu_time=simu_length,
    )


#Parse args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxbuilder for OpenMM simulations")
    parser.add_argument(
        "-p", "--results_path", help="Path to where the new directory will be created", default = "None"
    )
    parser.add_argument(
        "-ff", "--final_path", help="Path to where the folders should be transferred", default="None"
    )
    parser.add_argument(
        "-m1", "--monomer1", help="SMILES string of the first monomer", default="COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC([Au])=C[Cu]"
    )
    parser.add_argument(
        "-m2", "--monomer2", help="SMILES string of the second monomer", default="COCCOCCOCCOCC(C[Cu])O[Au]" #default="[Cu]CC(C#N)[Au]
    )
    parser.add_argument(
        "-m3", "--monomer3", help="SMILES string of the third monomer", default="None" #default="[Cu]CC=CC[Au]"
    )
    # SMILES list input
    parser.add_argument(
        "-sm", "--smiles", nargs='+', help="List of SMILES strings for the monomers", default = ["FC(CO[Au])C1CC1CC[Cu]","CC(C[Au])O[Cu]"]
    )
    # Fractions list input
    parser.add_argument(
        "-fr", "--fractions", nargs='+', type=float, help="List of fractions for the monomers", default = [0.2,0.8]
    )

    parser.add_argument(
        "-M",
        "--molality_salt",
        help="Molality of salt in mol/kg",
        default="1",
    )
    
    parser.add_argument(
        "-cs", "--charge_scale", help="Value to scale the charges of the salts", default="0.75"
    )
    parser.add_argument(
        "-pc", "--polymer_count", help="Number of polymer chains or molecules to be packed", default="1"
    )
    parser.add_argument(
        "-tr", "--total_repeats", help="Total number of repeats in the polymer", default="20"
    )
    parser.add_argument(
        "-f1", "--fraction_monomer1", help="Fraction of monomer1 in the polymer", default="0.5" #default="0.8"
    )
    parser.add_argument(
        "-f2", "--fraction_monomer2", help="Fraction of monomer2 in the polymer", default="0.3" #default="0.1"
    )
    parser.add_argument(
        "-pm", "--polymerization_mode", help="Mode for polymerization (e.g., random)", default="random" #random, homopolymer, alternating, block
    )
    parser.add_argument(
        "-c", "--concentration", help="Amount of LiTFSI pairs in the solution", default="10"
    )
    parser.add_argument(
        "-pn", "--poly_name", help="Name for the folder where all files will be saved", default="test"
    )
    parser.add_argument(
        "-ct", "--charge_type", help="Type of charges to select (e.g., LPG)", default="LPG"
    )
    parser.add_argument(
        "-ecs", "--end_carbons", help="If end carbons should be added when creating the polymer", default="True"
    )
    parser.add_argument(
        "-fnet", "--hitpoly_path", help="Path towards the HiTPoly folder", default="None"
    )
    parser.add_argument(
        "-htvs", "--htvs_path", help="Path towards the HTVS folder", default="None"
    )
    parser.add_argument(
        "-react", "--reaction", help="Reaction that creates the polymer", default="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]"
    )
    parser.add_argument(
        "-pi", "--product_index", help="Product index to use for the reaction", default="0"
    )
    parser.add_argument(
        "-box", "--box_multiplier", help="PBC box size multiplier for packmol", default="20"
    )
    parser.add_argument(
        "-conf", "--enforce_generation", help="Whether to force rdkit to create a conformation", default="False"
    )
    parser.add_argument(
        "--simu_type", help="Type of simulation to perform (e.g., conductivity, tg)", default="conductivity"
    )
    parser.add_argument(
        "--salt", help="Should salt be added to the simulation", default="True"
    )
    parser.add_argument(
        "--temperature", help="Simulation temperature", default="393"
    )
    parser.add_argument("--simu_length", help="Simulation length, ns", default="100")
    parser.add_argument(
        "--platform", help="For which platform to build the files for", default="supercloud"
    )

    parser.add_argument(
        "-nb", "--num_blocks", help="Number of blocks for block polymerization", default= "None"
    )

    parser.add_argument(
        "-arms", "--arms", type=int, help="Number of arms for star polymerization", default=3
    )
    parser.add_argument(
        "-blend", "--blend_mode", help="Blending Homopolymers", default="False"
    )

    parser.add_argument(
        "-polymer_chain_length", "--polymer_chain_length", type=int, help="Number of atoms per chain", default=1000
    )

    parser.add_argument(
        "--single_ion_conductor",
        help="Whether the system is a single-ion conductor",
        default="True",
    )

    parser.add_argument(
        "--anion",
        help="Symbol for the anion in the salt (e.g. Al, TFSI)",
        default="Al",
    )

    parser.add_argument(
        "--cation",
        help="Symbol for the cation in the salt (e.g. Na, Li)",
        default="Na",
    )


    args = parser.parse_args()

    print(f"SMILES: {args.smiles}")
    print(f"Fractions: {args.fractions}")


    # Parse boolean values properly
    if args.single_ion_conductor.lower() == "false":
        args.single_ion_conductor = False
    else:
        args.single_ion_conductor = True

    if args.end_carbons == "false" or args.end_carbons == "False":
        add_end_Cs = False
    else:
        add_end_Cs = True

    if args.enforce_generation == "False":
        args.enforce_generation = False
    else:
        args.enforce_generation = True

    if args.salt == "False":
        args.salt = False
    else:
        args.salt = True

    if args.num_blocks == "None":
        num_blocks = None
    else:
        num_blocks = int(args.num_blocks)


    # Convert paths to None if "None" is provided
    hitpoly_path = None if args.hitpoly_path == "None" else args.hitpoly_path
    htvs_path = None if args.htvs_path == "None" else args.htvs_path
    final_path = None if args.final_path == "None" else args.final_path
    results_path = None if args.results_path == "None" else args.results_path

    # Ensure that monomer2 and monomer3 are either a valid SMILES string or None
    monomer2 = None if args.monomer2 == "None" else args.monomer2
    monomer3 = None if args.monomer3 == "None" else args.monomer3

    

    # Run the function with parsed arguments
    run(
        results_path=results_path,
        final_path=final_path,
        monomers=args.smiles,  # Pass the list of SMILES
        molality=int(args.molality_salt),
        charge_scale=float(args.charge_scale),
        polymer_count=int(args.polymer_count),
        total_repeats=int(args.total_repeats),
        fractions=args.fractions,  # Pass the list of fractions
        polymerization_mode=args.polymerization_mode,
        concentration=[int(args.concentration), int(args.concentration)],
        poly_name=args.poly_name,
        charges=args.charge_type,
        add_end_Cs=add_end_Cs,
        hitpoly_path=hitpoly_path,
        htvs_path=htvs_path,
        reaction=args.reaction,
        product_index=int(args.product_index),
        box_multiplier=float(args.box_multiplier),
        enforce_generation=args.enforce_generation,
        salt=args.salt,
        simu_type=args.simu_type,
        simu_temp=int(args.temperature),
        simu_length=int(args.simu_length),
        platform=args.platform,
        num_blocks=num_blocks, 
        arms=int(args.arms),
        blend_mode = args.blend_mode.lower() == "true",
        polymer_chain_length=int(args.polymer_chain_length),
        single_ion_conductor=args.single_ion_conductor,
        anion=args.anion,
        cation=args.cation
    ) 
