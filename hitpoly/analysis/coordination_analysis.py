import pickle as pkl
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial import ConvexHull
from pymatgen.core.periodic_table import Element
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Structure, Lattice
import matplotlib
import matplotlib.colors as mcolors


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


def get_structure_list(atom_names_rdf, xyz_rdf, box_dim, res_id, cubic_box=True):
    if type(box_dim) == float:
        # cubic box
        lattice = Lattice.cubic(box_dim)
    elif cubic_box:
        lattice = Lattice.cubic(box_dim)
    else:
        # non-cubic box
        lattice = Lattice(box_dim)

    structure_list = []
    print("total number of frames:", len(xyz_rdf))
    for i in range(len(xyz_rdf)):
        if i % 200 == 0:
            print(i)
        structure_list.append(
            Structure(
                lattice=lattice,
                species=atom_names_rdf,
                coords=xyz_rdf[i],
                coords_are_cartesian=True,
                labels=res_id,
            )
        )
    return structure_list


def get_coord_environment_convex(Li_index, s, return_dist=False, ani_name_rdf=None):
    """ANION
    Li_index : index of cation in the pymatgen structure object
    structure : pymatgen structure object
    returns a list of pmg Neighbor objects in the order of distance from target cation
    """
    site = s[Li_index]
    nlist = s.get_neighbors(site, 6)
    nlist.sort(key=lambda x: x.nn_distance)
    anion_elements = []
    for ani in ani_name_rdf:
        anion_elements.append(Element(ani))
        
    # find atoms that don't coordinate to Li (carbons and F/S from TFSI)
    nlist_non_H_O = [i for i in nlist if i.specie not in anion_elements]
    nlist_new = []
    for n in nlist:
        if n.specie == Element("S"):
            if n.label.startswith("AN"):
                nlist_non_H_O.append(n)
            else:
                nlist_new.append(n)
        else:
            nlist_new.append(n)

    # find all the anions that coordinate to cation
    nlist_O = [i for i in nlist_new if i.specie in anion_elements]
    nearest_non_H_O = nlist_non_H_O[0]
    convex_nlist_O = nlist_O[:4]
    for neighb_index, neighb in enumerate(
        nlist_O[4:]
    ):  # this is where we start 3D shape
        og_convex = ConvexHull([i.coords for i in convex_nlist_O])
        new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
        new_non_H_convex = ConvexHull(
            [i.coords for i in convex_nlist_O]
            + [nearest_non_H_O.coords]
            + [neighb.coords]
        )
        exist_non_H_O = False
        for k in nlist_non_H_O:
            if k.nn_distance < neighb.nn_distance:
                exist_non_H_O = True
        if (
            len(new_convex.vertices) == len(og_convex.vertices) + 1
            and len(new_convex.vertices) != len(new_non_H_convex.vertices)
            and neighb.nn_distance < 1.75 * convex_nlist_O[0].nn_distance
        ):
            convex_nlist_O.append(neighb)
        else:
            break

    if return_dist:
        max_dist = convex_nlist_O[-1].distance(site)
        return convex_nlist_O, max_dist
    return convex_nlist_O


def return_distance_coordination(data, folder):
    len_datapoints = 0
    for i in range(len(data)):
        val_len = len(data[i])
        len_datapoints += val_len

    distance_coord = np.zeros((len_datapoints, 2))
    counter = 0
    for i in range(len(data)):
        for ind in data[i].keys():
            spec_list = [str(i.specie) for i in data[i][ind]["nlist"]]
            distance_coord[counter] = [int(len(spec_list)), data[i][ind]["dist"]]
            counter += 1

    with open(f"{folder}/coordination_environments.pkl", "wb") as f:
        pkl.dump(distance_coord, f)

    return distance_coord


def return_atomtypes_numberneighbors(data):
    # this returns the atomtypes coordinating to lithium
    # and the number of neighbors for the lithium that exist
    coord_env = set()
    num_neighs = set()

    for i in range(len(data)):
        for j in range(len(data[i])):
            num_neighbors = len(data[i][j]["nlist"])
            num_neighs.add(num_neighbors)
            for neighbor in data[i][j]["nlist"]:
                label_prefix = neighbor.label.split(" ")[0]
                # if label_prefix == "AN1":
                coord_env.add(f"{str(neighbor.specie)}-{label_prefix}")
                # elif label_prefix == "PL1":
                #     coord_env.add(f"{str(neighbor.specie)}-PL")

    coord_env = list(coord_env)
    num_neighs = list(num_neighs)

    return coord_env, num_neighs


def make_barplot_coordination_atomtypes(
    data,
    coord_env,
    num_neighs,
    distance_coord,
    folder,
    name,
    temperature,
):
    # dict of all the coordination numbers with all the lithiums and their coordinating atoms
    num_neighs_dict = {f"num_neighs_{n}": [] for n in num_neighs}
    length = len(coord_env)
    # returns array fors each li with the amount of atoms coordinating to it, each column is an atom type
    # this list matches with coord_env that contains the atom type
    for i in range(len(data)):
        for j in range(len(data[i])):
            count_env = [0] * length
            for l in range(len(coord_env)):
                for k in range(len(data[i][j]["nlist"])):
                    if str(data[i][j]["nlist"][k].specie) == coord_env[l].split("-")[0]:
                        if (
                            data[i][j]["nlist"][k].label[:3]
                            == coord_env[l].split("-")[1]
                        ):
                            count_env[l] += 1
            for n in num_neighs:
                if sum(count_env) == n:
                    num_neighs_dict[f"num_neighs_{n}"].append(count_env)

    # Counting the how many unique anions and polymers are in each coord shell
    # and the mean distances to the polymers and to the anions
    unique_solv = {}
    for i in range(1,40):
        unique_solv[i] = []
        
    for item1 in data.values():
        for item2 in item1.values():
            poly_solv = []
            anion_solv = []
            for solv in item2['nlist']:
                if 'PL' in solv.label.split()[0]:
                    poly_solv.append([int(solv.label.split()[1]), solv.nn_distance])
                else:
                    anion_solv.append([int(solv.label.split()[1]), solv.nn_distance])
            if poly_solv and anion_solv:
                unique_solv[len(item2['nlist'])].append(
                    [len(set([i[0] for i in poly_solv])), sum([i[1] for i in poly_solv])/len(poly_solv),
                    len(set([i[0] for i in anion_solv])), sum([i[1] for i in anion_solv])/len(anion_solv)]
                )
            if not anion_solv:
                unique_solv[len(item2['nlist'])].append(
                    [len(set([i[0] for i in poly_solv])), sum([i[1] for i in poly_solv])/len(poly_solv),
                    0,np.nan]
                )
            if not poly_solv:
                unique_solv[len(item2['nlist'])].append(
                    [0, np.nan,
                    len(set([i[0] for i in anion_solv])), sum([i[1] for i in anion_solv])/len(anion_solv)]
                )
                
                
    freq_counter = 0
    for val in unique_solv.values():
        freq_counter += len(val)

    with open(f"{folder}/unique_coordinators.txt", "w") as f:
        f.write('Coord num,prevalence,unique chains,mean chain distance,\
chain prevalence,unique anions,mean anion distance,anion prevalence\n')
        for coord_num in unique_solv.keys():
            if unique_solv[coord_num]:
                coord_num_prevalence = len(np.array(unique_solv[coord_num]))/freq_counter
                unique_chains = np.nanmean(np.array(unique_solv[coord_num])[:,0])
                mean_chain_dist = np.nanmean(np.array(unique_solv[coord_num])[:,1])
                poly_prevalence_in_coord_env = np.count_nonzero(~np.isnan(np.array(unique_solv[coord_num])[:,0]))/len(np.array(unique_solv[coord_num]))
                unique_anions = np.nanmean(np.array(unique_solv[coord_num])[:,2])
                mean_anion_dist = np.nanmean(np.array(unique_solv[coord_num])[:,3])
                anion_prevalence_in_coord_env = np.count_nonzero(~np.isnan(np.array(unique_solv[coord_num])[:,2]))/len(np.array(unique_solv[coord_num]))
                print(coord_num,coord_num_prevalence,unique_chains,mean_chain_dist,poly_prevalence_in_coord_env,unique_anions,mean_anion_dist,anion_prevalence_in_coord_env)
                f.write(f"{str(coord_num)},{str(coord_num_prevalence)},{str(unique_chains)},\
{str(mean_chain_dist)},{str(poly_prevalence_in_coord_env)},{str(unique_anions)},{str(mean_anion_dist)},{str(anion_prevalence_in_coord_env)}\n")

    # return all the coordination environment based on the atom types
    coord_env_atoms = np.zeros((len(num_neighs), len(coord_env)))
    for i in range(len(num_neighs_dict)):
        coord_ev_val = np.mean(num_neighs_dict[f"num_neighs_{num_neighs[i]}"], axis=0)
        coord_env_atoms[i] = coord_ev_val

    # ensure the plot always has same ordering and coloring for atom type
    neighbors_average = {}
    for i in range(len(coord_env)):
        neighbors_average[coord_env[i]] = coord_env_atoms[:, i]
    neighbors_average = dict(sorted(neighbors_average.items()))
    colors_list = []
    for label, spec in neighbors_average.items():
        # For reproducibility and consistent color mapping, get a unique sorted list of labels outside the loop
        # Suppose neighbors_average_keys is a sorted list of all possible labels (e.g. list(neighbors_average.keys()))
        # Here we cache the color assignment on first run
        if "label_color_map" not in locals():
            palette = sns.color_palette("tab10", n_colors=len(neighbors_average.keys()))
            neighbors_average_keys = sorted(neighbors_average.keys())
            label_color_map = {lab: mcolors.to_hex(palette[i]) for i, lab in enumerate(neighbors_average_keys)}

        colors_list.append(label_color_map[label])

    coord_number, freq = np.unique(distance_coord[:, 0], return_counts=True)
    total_sum = np.sum(freq)
    freq_percent = (freq / total_sum) * 100
    freq_percent = freq_percent.tolist()

    # only keep coordination environments with at least 1 percent occurence
    freq_relevant = []
    neighbors_relevant = {}
    num_neighs_relevant = []
    relevant_ids = []

    for i in range(len(freq_percent)):
        if freq_percent[i] > 1:
            freq_relevant.append(freq_percent[i])
            num_neighs_relevant.append(num_neighs[i])
            relevant_ids.append(i)
    for j in neighbors_average.keys():
        relevant = []
        for id in relevant_ids:
            relevant.append(neighbors_average[j][id])
        neighbors_relevant[j] = np.array(relevant)

    distances_mean = np.arange(len(coord_number))
    for i in range(len(distance_coord)):
        for j in range(len(coord_number)):
            if distance_coord[i][0] == coord_number[j]:
                distances_mean[j] += distance_coord[i][1]
    distances_mean = distances_mean / freq

    # write files with important features
    relevant_id = np.argmax(freq_relevant)
    max_coord = num_neighs_relevant[relevant_id]
    max_coord_dict = {
        "majority_coordination": max_coord,
        "mean_distance": distances_mean[relevant_id],
    }
    for key in neighbors_relevant.keys():
        max_coord_dict[key] = neighbors_relevant[key][relevant_id]
    with open(f"{folder}/coordination_majority.txt", "w") as f:
        keys = ",".join(max_coord_dict.keys())
        f.write(keys + "\n")
        values = ",".join(
            f"{round(float(value), 2):.2f}"
            if isinstance(value, (int, float))
            else str(value)
            for value in max_coord_dict.values()
        )
        f.write(values + "\n")

    with open(f"{folder}/coordination_everything.txt", "w") as f:
        keys = ",".join(neighbors_relevant.keys())
        f.write(f"coordination_number,frequency,mean_distance," + f"{keys}\n")
        for i in range(len(num_neighs_relevant)):
            vals = [neighbors_relevant[key][i] for key in neighbors_relevant.keys()]
            vals_str = ",".join([str(val) for val in vals])
            f.write(
                f"{num_neighs_relevant[i]}, {freq_relevant[i]},{distances_mean[relevant_ids[i]]}, {vals_str}\n"
            )

    figsize = (6, 5)
    fig_scalingfactor = figsize[1] / 5
    fontsize = 20 * fig_scalingfactor
    plt.rcParams.update({"font.size": fontsize})
    labelsize = 20 * fig_scalingfactor
    titelsize = labelsize * 1.2
    pad = labelsize / 3
    tickwidth = 2.5 * fig_scalingfactor
    maj_tick_size = 6 * fig_scalingfactor
    min_tick_size = 3 * fig_scalingfactor
    dpi = 400

    print(neighbors_relevant, num_neighs_relevant, freq_relevant)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bottom = np.zeros(len(num_neighs_relevant))
    counter = 0
    for idx, (label, spec) in enumerate(neighbors_relevant.items()):
        bars = ax.bar(
            num_neighs_relevant,
            spec,
            label=label,
            bottom=bottom,
            color=colors_list[idx],
        )
        if idx == len(list(neighbors_relevant.keys())) - 1:
            for i, (bar, value) in enumerate(zip(bars, spec)):
                if value >= 0.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y(),
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=labelsize * 0.8,
                    )
                ax.text(
                    bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() + bars[i].get_y() + 0.1,
                    f"{freq_relevant[counter]:.0f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=labelsize * 0.8,
                )
                counter += 1
        else:
            for bar, value in zip(bars, spec):
                if value >= 0.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y(),
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=labelsize * 0.8,
                    )
        bottom += spec

    plot_name = "_".join(folder.split("/")[-3].split("_")[:6])
    ax.set_title(
        f"Cluster pop at {name} of simu, {plot_name}, T={temperature}",
        fontsize=int(2 * labelsize / 3),
    )
    ax.legend(fontsize=labelsize * 0.8)
    ax.set_xlim(min(num_neighs_relevant) - 1, max(num_neighs_relevant) + 1)
    ax.set_ylim(0, max(num_neighs_relevant) + 1)
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.set_xlabel("Coordination number to Li-cation", fontsize=labelsize)
    ax.set_ylabel("Coordination by atom type", fontsize=labelsize)

    plt.tight_layout()
    plt.savefig(f"{folder}/coordination_atomtypes_{name}.svg", dpi=dpi)
    plt.savefig(f"{folder}/coordination_atomtypes_{name}.png", dpi=dpi)
    plt.show()


def make_violinplot_distance_coordinationnumber(
    distance_coord,
    folder,
    name,
    temperature,
):
    df_violin = pd.DataFrame(
        distance_coord, columns=["coordination_number", "distance"]
    )
    df_violin["coordination_number"] = df_violin["coordination_number"].astype(int)

    figsize = (6, 5)
    fig_scalingfactor = figsize[1] / 5
    fontsize = 20 * fig_scalingfactor
    plt.rcParams.update({"font.size": fontsize})
    labelsize = 20 * fig_scalingfactor
    titelsize = labelsize * 1.2
    pad = labelsize / 3
    tickwidth = 2.5 * fig_scalingfactor
    maj_tick_size = 6 * fig_scalingfactor
    min_tick_size = 3 * fig_scalingfactor
    dpi = 400

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.violinplot(
        x="coordination_number",
        y="distance",
        data=df_violin,
        color="skyblue",
        linecolor="navy",
        density_norm="count",
        linewidth=1,
        inner_kws=dict(box_width=3),
    )

    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.set_xlabel("Coordination Number", fontsize=labelsize)
    ax.set_ylabel("$d_{\mathrm{furthest~ anion}} ~ /  \AA$", fontsize=labelsize)

    plot_name = "_".join(folder.split("/")[-3].split("_")[:6])
    ax.set_title(
        f"Cluster pop at {name} of simu, {plot_name}, T={temperature}",
        fontsize=int(2 * labelsize / 3),
    )

    plt.tight_layout()
    plt.savefig(f"{folder}/distance_coordinations_{name}.svg", dpi=dpi)
    plt.savefig(f"{folder}/distance_coordinations_{name}.png", dpi=dpi)
    plt.show()
