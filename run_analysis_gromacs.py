import os

os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from hitpoly.analysis.trajectory_analysis import (
    get_coords_PDB_msd,
    get_coords_PDB_rdf,
    unwrap_all,
    save_params,
    read_xyz,
    plot_calc_diffu,
    plot_calc_corr,
    plot_calc_rdf,
    save_gromacs_params,
)
from hitpoly.analysis.coordination_analysis import (
    get_structure_list,
    get_coord_environment_convex,
    return_atomtypes_numberneighbors,
    return_distance_coordination,
    make_barplot_coordination_atomtypes,
    make_violinplot_distance_coordinationnumber,
)
import argparse
import time
import numpy as np


def run(
    folder,
    cat_name: list,
    ani_name: list,
    simu_time,
    diffu_calc_start_time,
    save_freq,
    temperature,
    name,
    corr_analysis=True,
    gromacs=True,
    platform="local",
    poly_name: list = None,
    repeat_units=None,
):
    if not os.path.isdir(f"{folder}/results"):
        os.mkdir(f"{folder}/results")
    if platform == "local":
        pre_folder = folder
    else:
        pre_folder = f"{folder}/tmpdir"

    if gromacs:
        if not os.path.isfile(f"{folder}/results/simu_output.pdb"):
            os.chdir(folder)
            os.system(
                "gmx_mpi convert-trj -f simu/nvt_prod.trr -s simu/nvt_prod.tpr -o results/simu_output.pdb"
            )
    folder = f"{folder}/results"

    cat_names = []
    for ind, i in enumerate(cat_name):
        cat_names.append([i, "CA" + str(ind + 1)])
    ani_names = []
    for ind, i in enumerate(ani_name):
        ani_names.append([i, "AN" + str(ind + 1)])
    if poly_name:
        poly_names = []
        for ind, i in enumerate(poly_name):
            poly_names.append([i, "PL" + str(ind + 1)])
        atom_name_list = cat_names + ani_names + poly_names
    else:
        poly_names = []
        atom_name_list = cat_names + ani_names
    print("folder:", folder)

    if gromacs:
        if os.path.exists(f"{folder}/cell_dimensions.npy"):
            cell = np.load(f"{folder}/cell_dimensions.npy")
        else:
            cell = save_gromacs_params(folder)
    else:
        cell = save_params(folder)

    if not os.path.exists(f"{folder}/atom_names_msd.txt"):
        frame_count, simu_time = get_coords_PDB_msd(
            folder,
            pre_folder,
            atom_name_list,
            save_interval=save_freq,
            cell=cell,
            repeat_units=repeat_units,
        )
    else:
        with open(f"{folder}/frame_count.txt", "r") as f:
            frame_count = int(f.readlines()[0])
        simu_time = (frame_count - 1) * save_freq / 1000
        print("MSD coords and atom names already exists!")

    if not os.path.exists(f"{folder}/atom_names_rdf.txt"):
        get_coords_PDB_rdf(
            folder=folder,
            frame_count=frame_count,
            save_interval=save_freq,
        )
    else:
        print("RDF coords and atom names already exists!")

    xyz_msd, atom_names_msd, atom_names_long_msd = read_xyz(
        folder, "atom_names_msd.txt", "xyz_wrapped_msd.txt"
    )

    with open(f"{folder}/xyz_length.txt", "w") as f:
        f.write(str(xyz_msd.shape) + "\n")
        f.write(str(xyz_msd.shape[0] * save_freq / 1000) + "ns")

    # print(f"Running analysis for {folder}")
    xyz_msd_unwrp = unwrap_all(xyz_msd, cell)
    cat_ani_index = [ind for ind, j in enumerate(atom_names_long_msd) if "PL1" not in j]
    xyz_msd_corr = xyz_msd_unwrp[:, cat_ani_index].copy()

    for i in poly_names:
        for j in list(set(atom_names_long_msd)):
            if i[1] in j:
                if i[0] != j.split("-")[0]:
                    i[0] = j.split("-")[0]

    plot_calc_diffu(
        xyz=xyz_msd_unwrp,
        folder=folder,
        save_freq=save_freq,
        diffu_time=diffu_calc_start_time,
        cat_name=[i[0] + "-" + i[1] for i in cat_names],
        ani_name=[i[0] + "-" + i[1] for i in ani_names],
        cell=cell,
        temperature=temperature,
        atom_names=atom_names_long_msd,
        name=name,
        poly_name=[i[0] + "-" + i[1] for i in poly_names],
    )

    atom_names_long_msd = [i for i in atom_names_long_msd if "PL1" not in i]
    plot_calc_corr(
        xyz=xyz_msd_corr,
        folder=folder,
        save_freq=save_freq,
        cat_name=[i[0] + "-" + i[1] for i in cat_names],
        ani_name=[i[0] + "-" + i[1] for i in ani_names],
        cell=cell,
        temperature=temperature,
        atom_names=atom_names_long_msd,
        name=name,
    )

    one_name = ["Li-CA1"]

    two_names = [
        [
            "O-PL1",
            "O-AN1",
            "N-AN1",
            "S-PL1",
            "N-PL1",
            "Br-PL1",
            "P-PL1",
            "Si-PL1",
        ],
        ["O-PL1"],
        ["S-PL1", "N-PL1", "Br-PL1", "P-PL1", "Si-PL1"],
        ["O-AN1"],
        ["N-AN1"],
    ]
    names = ["solv_all", "O_poly", "others_poly", "O_tfsi", "N_tfsi"]

    for frame in ["beginning", "end"]:
        xyz_rdf, atom_names_rdf, atom_names_long_rdf, residue_ids = read_xyz(
            folder,
            "atom_names_rdf.txt",
            f"xyz_wrapped_rdf_{frame}.txt",
            include_resid=True,
        )
        temp_names = ["_".join([name, frame]) for name in names]
        plot_calc_rdf(
            xyz_rdf_unwrp=xyz_rdf,
            folder=folder,
            one_name=one_name,
            two_names=two_names,
            cell=cell,
            atom_names_long_rdf=atom_names_long_rdf,
            names=temp_names,
            temperature=temperature,
        )

        # convex hull coordination analysis
        atom_names_rdf = np.array(atom_names_rdf)
        atom_names_long_rdf = np.array(atom_names_long_rdf)

        structurelist = get_structure_list(
            atom_names_rdf=atom_names_rdf,
            xyz_rdf=xyz_rdf,
            res_id=residue_ids,
            box_dim=cell[0][0],
        )

        data_coordination = {
            i: {
                j: {"nlist": nlist, "dist": dist, "numer_coord_O": len(nlist)}
                for j, (nlist, dist) in enumerate(
                    (
                        get_coord_environment_convex(
                            li_idx, structurelist[i], return_dist=True
                        )
                        for li_idx in structurelist[i].indices_from_symbol("Li")
                    )
                )
            }
            for i in range(len(structurelist))
        }

        coord_env_atomtypes, number_atoms_coord_env = return_atomtypes_numberneighbors(
            data=data_coordination
        )
        distance_coord = return_distance_coordination(
            data=data_coordination, folder=folder
        )

        make_barplot_coordination_atomtypes(
            data=data_coordination,
            coord_env=coord_env_atomtypes,
            num_neighs=number_atoms_coord_env,
            distance_coord=distance_coord,
            folder=folder,
            name=frame,
            temperature=temperature,
        )

        make_violinplot_distance_coordinationnumber(
            distance_coord=distance_coord,
            folder=folder,
            name=frame,
            temperature=temperature,
        )

    if gromacs:
        os.remove(f"{folder}/simu_output.pdb")

    print(f"Analysis done for {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running MSD analysis on PDB trajectories"
    )
    parser.add_argument("-p", "--folder", help="Path towards the trajectory folder")
    parser.add_argument(
        "-c",
        "--cat",
        default="Li",
        help="Name of the cation atom, can input comma separated list, example Li,Na for two cations",
    )
    parser.add_argument(
        "-a",
        "--ani",
        default="N",
        help="Name of the anion atom, can input comma separated list, example N,N for two N containing anions",
    )
    parser.add_argument(
        "--poly",
        default="O",
        help="Name of the polymer atom, can input comma separated list, example O,O for two O containing polymers",
    )
    parser.add_argument(
        "-t", "--simu_t", default="200", help="Simulation time in ns, default 200"
    )
    parser.add_argument(
        "-d",
        "--diffu_t",
        default="75",
        help="Time to start diffusivity analysis (linear regime only) in ns, default 75",
    )
    parser.add_argument(
        "-f", "--save_freq", default="5", help="Frame save frequency in ps, default 5"
    )
    parser.add_argument(
        "-temp",
        "--temperature",
        default="353",
        help="The temperature of the simulation, default 353",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="None",
        help="Title name for the diffusivity plot, default None",
    )
    parser.add_argument(
        "--gromacs",
        action="store_true",
    )
    parser.add_argument(
        "--platform",
        default="local",
    )
    parser.add_argument(
        "--repeat_units", help="Amount of monomer repeat units in backbone"
    )

    args = parser.parse_args()
    if args.name == "None":
        args.name = None
    if args.poly == "None":
        args.poly = None
    else:
        poly_name = args.poly.split(",")
    cat_name = args.cat.split(",")
    ani_name = args.ani.split(",")

    start_time = time.time()
    run(
        folder=args.folder,
        cat_name=cat_name,
        ani_name=ani_name,
        simu_time=int(args.simu_t),
        diffu_calc_start_time=int(args.diffu_t),
        save_freq=int(args.save_freq),
        temperature=int(args.temperature),
        name=args.name,
        gromacs=args.gromacs,
        platform=args.platform,
        poly_name=poly_name,
        repeat_units=int(args.repeat_units),
    )
    print("length of analysis [m]:", (time.time() - start_time) / 60)
