import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from tqdm import tqdm

from scipy.signal import correlate
from scipy.optimize import curve_fit

from openmm import XmlSerializer
from openmm.app import PDBFile
import itertools
import time
import pickle as pkl
from ffnet.utils.constants import ELEMENT_TO_MASS

FS_TO_NS = 1e-6
ELEC = 1.60217662 * (1e-19)  # C
KB = 1.3806452e-23  # J/K

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


def lin_func(x, m, y0):
    return y0 + m * x


def exponential(x, a, b, c):
    return np.exp(-(x - c) / a) + b


def lin_exp(x, a1, a2, c):
    return c * np.exp(-x / a1) + (1.0 - c) * np.exp(-x / a2)


def lin_exp3(x, a1, a2, a3, b, c):
    return b * np.exp(-x / a1) + c * np.exp(-x / a2) + (1.0 - b - c) * np.exp(-x / a3)


def mini_dist(pos1, pos2, cell_vecs):
    abs_dist = np.abs(pos1 - pos2) % (cell_vecs)
    pbc_dist = np.stack([np.abs(abs_dist - cell_vecs), abs_dist]).min(axis=0)
    return np.linalg.norm(pbc_dist, axis=-1)


def CN(rij, r0, n=6, m=12):
    frac = rij / r0
    sij = (1.0 - np.power(frac, n)) / (1.0 - np.power(frac, m))
    nans = np.where(np.isnan(sij) == True)
    sij[nans] = 0.5
    return sij


def RDF(dists, dx, Na, Nb, V):
    data = dists.flatten()
    data = data[data != 1e10]
    maxv = data.max()
    dx2 = dx / 2
    nbins = int(np.ceil(maxv / dx))
    binc = np.arange(nbins) * dx + dx2
    hist = np.histogram(data, bins=nbins, range=(0, maxv))[0].astype(float)
    hist /= dists.shape[0]
    gr = hist * (V / (4.0 * np.pi * Na * Nb))
    gr /= np.power(binc, 2) * dx

    crd = np.zeros(shape=(nbins,), dtype=float)
    for binn in range(nbins):
        crd[binn] = hist[: (binn + 1)].sum() / Na

    return binc, gr, crd


def find_slope1_indeces(timesteps_delTs, total_corr):
    slope_list = []
    iteration_window = 15
    steps_to_skip = len(timesteps_delTs) // 5  # Skip the first 20%
    for ind, i in enumerate(range(len(timesteps_delTs))):
        if ind < steps_to_skip or ind > len(timesteps_delTs) - iteration_window:
            continue
        # Iterates through windows of lenght {iteration_window}
        diffu_inx = np.arange(i, i + iteration_window)
        if np.any(np.round(total_corr[diffu_inx], decimals=8)):
            if np.all(total_corr[diffu_inx] > 0):
                popt, pcov = curve_fit(
                    lin_func,
                    np.log10(timesteps_delTs[diffu_inx]),
                    np.log10(total_corr[diffu_inx]),
                    p0=[0.0, 1.0],
                )
            elif np.all(total_corr[diffu_inx] < 0):
                popt, pcov = curve_fit(
                    lin_func,
                    np.log10(timesteps_delTs[diffu_inx]),
                    np.log10(-1 * total_corr[diffu_inx]),
                    p0=[0.0, 1.0],
                )
            else:
                print("both pos and negative for loglog, so assign 1e5")
                slope_list.append(1e5)
                continue
        else:
            popt, pcov = curve_fit(
                lin_func,
                np.log10(timesteps_delTs[diffu_inx]),
                np.zeros_like(total_corr[diffu_inx]),
                p0=[0.0, 1.0],
            )
        slope_list.append(popt[0])
    slope_val = min(slope_list, key=lambda x: abs(x - 1))
    slope_ind = slope_list.index(slope_val)
    diffu_inx = np.arange(
        slope_ind + steps_to_skip, slope_ind + steps_to_skip + iteration_window
    )
    return diffu_inx, slope_list[slope_ind]


def binning_2D(data1, data2, dx1, dx2, density=True):
    min1, max1 = data1.min(), data1.max()
    min2, max2 = data2.min(), data2.max()
    nbins1 = int(np.ceil((max1 - min1) / dx1))
    nbins2 = int(np.ceil((max2 - min2) / dx2))
    count_mat = np.zeros(shape=(nbins1, nbins2))
    for val1, val2 in zip(
        data1.flatten(), data2.flatten()
    ):  # , data1.mask.flatten(), data2.mask.flatten()): , mask1, mask2
        # if not (mask1 or mask2):
        idx1 = int(np.floor((val1 - min1) / dx1))
        idx2 = int(np.floor((val2 - min2) / dx2))
        count_mat[idx1, idx2] += 1

    if density:
        count_mat /= count_mat.sum() * dx1 * dx2

    centers1 = np.arange(nbins1) * dx1 + dx1 / 2 + min1
    centers2 = np.arange(nbins2) * dx2 + dx2 / 2 + min2
    x_2d, y_2d = np.meshgrid(centers1, centers2)

    return count_mat.T, x_2d, y_2d


def pmf_2D(data1, data2, dx1, dx2, T=300):
    density_mat, x_2d, y_2d = binning_2D(data1, data2, dx1, dx2, density=True)
    R = 8.314 / 1000.0  # kJ/mol K
    A = -R * T * np.log(density_mat)
    A = np.ma.masked_array(A, mask=np.isinf(A))

    return A, x_2d, y_2d


def find_mic(D, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) D"""
    # Calculate the 4 unique unit cell diagonal lengths
    diags = np.sqrt(
        (
            np.dot(
                [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1],
                ],
                cell,
            )
            ** 2
        ).sum(1)
    )

    # calculate 'mic' vectors (D) and lengths (D_len) using simple method
    Dr = np.dot(D, np.linalg.inv(cell))
    D = np.dot(Dr - np.round(Dr) * pbc, cell)
    D_len = np.sqrt((D**2).sum(1))
    # return mic vectors and lengths for only orthorhombic cells,
    # as the results may be wrong for non-orthorhombic cells
    if (max(diags) - min(diags)) / max(diags) > 1e-9:
        return D, D_len

    # The cutoff radius is the longest direct distance between atoms
    # or half the longest lattice diagonal, whichever is smaller
    cutoff = min(max(D_len), max(diags) / 2.0)
    # The number of neighboring images to search in each direction is
    # equal to the ceiling of the cutoff distance (defined above) divided
    # by the length of the projection of the lattice vector onto its
    # corresponding surface normal. a's surface normal vector is e.g.
    # b x c / (|b| |c|), so this projection is (a . (b x c)) / (|b| |c|).
    # The numerator is just the lattice volume, so this can be simplified
    # to V / (|b| |c|). This is rewritten as V |a| / (|a| |b| |c|)
    # for vectorization purposes.
    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    n = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) / (V * latt_len)), dtype=int)

    # Construct a list of translation vectors. For example, if we are
    # searching only the nearest images (27 total), tvecs will be a
    # 27x3 array of translation vectors. This is the only nested loop
    # in the routine, and it takes a very small fraction of the total
    # execution time, so it is not worth optimizing further.
    tvecs = []
    for i in range(-n[0], n[0] + 1):
        latt_a = i * cell[0]
        for j in range(-n[1], n[1] + 1):
            latt_ab = latt_a + j * cell[1]
            for k in range(-n[2], n[2] + 1):
                tvecs.append(latt_ab + k * cell[2])
    tvecs = np.array(tvecs)

    # Translate the direct displacement vectors by each translation
    # vector, and calculate the corresponding lengths.
    D_trans = tvecs[np.newaxis] + D[:, np.newaxis]
    D_trans_len = np.sqrt((D_trans**2).sum(2))

    # Find mic distances and corresponding vector(s) for each given pair
    # of atoms. For symmetrical systems, there may be more than one
    # translation vector corresponding to the MIC distance; this finds the
    # first one in D_trans_len.
    D_min_len = np.min(D_trans_len, axis=1)
    D_min_ind = D_trans_len.argmin(axis=1)
    D_min = D_trans[range(len(D_min_ind)), D_min_ind]

    return D_min, D_min_len


def unwrap(this_xyz, last_xyz, cell):
    #     this_n = np.array(this_nxyz)[:,:1]
    #     this_xyz = np.array(this_nxyz)[:,:]
    #     last_xyz = np.array(last_nxyz)[:,:]
    dr = this_xyz - last_xyz
    #     new_dr = mic(dr,cell)
    new_dr, _ = find_mic(dr, cell)
    if not np.array_equal(dr, new_dr):
        new_xyz = new_dr + last_xyz
    else:
        new_xyz = this_xyz
    #     new_nxyz = np.hstack((this_n,new_xyz))

    return new_xyz


def wrap_xyz(xyz, cell):
    """
    Wrapping xyz coordinates so that all the atoms fit within
     the box dimensions
    """
    for i, cur in enumerate(xyz):
        for j, cur_atom in enumerate(cur):
            # wrap the coordinates into the cell box
            image = cur_atom // cell.diagonal()
            cur_atom = cur_atom - cell.diagonal() * image
            xyz[i][j] = cur_atom
    return xyz


def unwrap_all(ion_pos, cell):
    ion_pos_new = np.full_like(ion_pos, 0)
    ion_pos_new[0, :, :] = ion_pos[0, :, :]
    for indi, i in enumerate(ion_pos[1:]):
        ion_pos_new[indi + 1] = unwrap(ion_pos[indi + 1], ion_pos_new[indi], cell)
    return ion_pos_new


def torch_nbr_list(
    xyz, cutoff, cell, directed=True, pbc_flag=True, requires_large_offsets=True
):
    """Pytorch implementations of nbr_list for minimum image convention, the offsets are only limited to 0, 1, -1:
    it means that no pair interactions is allowed for more than 1 periodic box length. It is so much faster than
    neighbor_list algorithm in ase.
    It is similar to the output of neighbor_list("ijS", atomsobject, cutoff) but a lot faster
    Args:
        atomsobject (TYPE): Description
        cutoff (float): cutoff for
        device (str, optional): Description
        requires_large_offsets: to get offsets beyond -1,0,1
    Returns:
        i, j, cutoff: just like ase.neighborlist.neighbor_list
    """

    if pbc_flag:
        # check if sufficiently large to run the "fast" nbr_list function
        # otherwise, default to the "robust" nbr_list function below for small cells
        if np.all(2 * cutoff < cell.diagonal()):
            # fidn the distance from one to another
            dis_mat = xyz[None, :, :] - xyz[:, None, :]
            cell_dim = cell.diagonal()
            # cell_dim = torch.Tensor(np.array(atomsobject.get_cell())).diag().to(device)
            if requires_large_offsets:
                shift = np.round(np.divide(dis_mat, cell_dim))
                offsets = -shift
            else:
                offsets = np.greater_equal(-dis_mat, 0.5 * cell_dim).astype(
                    float
                ) + np.less(dis_mat, -0.5 * cell_dim).astype(float)

            dis_mat = dis_mat + offsets * cell_dim
            dis_sq = np.sum(np.power(dis_mat, 2), axis=-1)
            mask = (dis_sq < cutoff**2) & (dis_sq != 0)
            # print('min:', np.min(np.sqrt(dis_sq[dis_sq!=0])), np.max(np.sqrt(dis_sq[dis_sq!=0])))
            nbr_list = mask.nonzero()
            offsets = offsets[nbr_list[0], nbr_list[1], :]
        # nbr_list[:, 1], :].detach().to("cpu").numpy()
        else:
            raise NameError("only works if cutoff << cell_param for now")
    else:
        raise NameError("only works if pbc for now")

    if not directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]
    # i, j = nbr_list[:, 0].detach().to("cpu").numpy(
    # ), nbr_list[:, 1].detach().to("cpu").numpy()
    i, j = nbr_list[0], nbr_list[1]

    if pbc_flag:
        offsets = offsets
    else:
        offsets = np.zeros((nbr_list.shape[0], 3))

    return i, j, offsets


# def get_molecules(nbr_list, all_ion_atoms):
#     """
#     Args:
#         nbr_list: list of atoms that are within some cutoff from one another
#         all_ion_atoms: array indeces of all ions
#     Returns:
#         list of arrays, where each array has the indeces of the molecules within the same cluster
#     """
#     clusters = np.zeros(len(all_ion_atoms), dtype=int)
#     for i, cur in enumerate(all_ion_atoms):
#         # the most recent added cluster index (counter)
#         mm = max(clusters)
#         # oxy_neighbors contains all the atoms that the current atom is in the cutoff of
#         oxy_neighbors = nbr_list[nbr_list[:, 0] == cur].reshape(-1)
#         # if this has no neighbors, then the current atom will belong to a new cluster
#         if len(oxy_neighbors) == 0:
#             clusters[i] = mm + 1
#             continue
#         # if all the neighbors are in unassigned clusters and the current atom has already been assigned,
#         # assign the neighbors to the current cluster
#         if (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
#             clusters[oxy_neighbors] = clusters[i]
#         # if all the neighbors are in unassigned clusters and the curent cluster is also unassigned
#         # assign the current atom and all the neighbors to a new cluster
#         elif (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
#             clusters[oxy_neighbors] = mm + 1
#             clusters[i] = mm + 1
#         # if not all the neighbors are in unassigned clusters (so if at least one neighbor has been assigned)
#         # and the current atom has not been assigned yet, assign the current atom and all neighbors to
#         # the lowest cluster name (choosing the minimum cluster name as default is arbitrary,
#         # as long as all have same)
#         elif not (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
#             clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
#             clusters[oxy_neighbors] = min(
#                 clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
#             )
#         # if not all the neighbors are in unassigned clusters and current atom has been assigned,
#         elif not (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
#             # temporary is the first you only take the clusters that are neighbors
#             # then you only take the nonzero of those neighboring clusters
#             # then you only take the ones that are not the minimum of those nonzero neighboring clusters
#             tmp = clusters[oxy_neighbors]
#             tmp = tmp[tmp != 0]
#             tmp = tmp[tmp != min(tmp)]
#             # set the current cluster to the minimum of these nonzero, neighboring clusters
#             clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
#             # then, set all the neighboring atoms to this same minimum cluster name
#             clusters[oxy_neighbors] = min(
#                 clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
#             )
#             # temp contains all the nonzero neighboring cluster types
#             # for each of these nonzero neighboring cluster types, change them to the overall minimum same clustering name
#             for tr in tmp:
#                 clusters[np.where(clusters == tr)[0]] = min(
#                     clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
#                 )
#     molecules = []
#     for i in range(1, max(clusters) + 1):
#         if np.size(np.where(clusters == i)[0]) == 0:
#             continue
#         molecules.append(np.where(clusters == i)[0])
#     return molecules

##FIXED CLUSTER MAPPING

def get_molecules(nbr_list, all_ion_atoms):
    """
    Args:
        nbr_list: Nx2 array of pairs of atoms that are neighbors
                  indices are relative to the subset of selected atoms
        all_ion_atoms: array of indices (relative to the selected atom array)
                       identifying which atoms are ions

    Returns:
        list of arrays, where each array has the indices (local in all_ion_atoms)
        of atoms belonging to the same cluster
    """

    # Map from absolute (local-to-selected-atoms) index → local index within all_ion_atoms
    abs_to_local = {abs_idx: local_idx for local_idx, abs_idx in enumerate(all_ion_atoms)}

    clusters = np.zeros(len(all_ion_atoms), dtype=int)

    for i, cur in enumerate(all_ion_atoms):

        mm = clusters.max()

        # Find neighbors of current atom
        oxy_neighbors_abs = nbr_list[nbr_list[:, 0] == cur, 1]

        # Keep only neighbors that are also in all_ion_atoms
        oxy_neighbors_abs = [n for n in oxy_neighbors_abs if n in abs_to_local]

        # Remap neighbor indices to local indices in all_ion_atoms
        oxy_neighbors = np.array([abs_to_local[n] for n in oxy_neighbors_abs], dtype=int)

        if len(oxy_neighbors) == 0:
            clusters[i] = mm + 1
            continue

        if (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
            clusters[oxy_neighbors] = clusters[i]

        elif (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
            clusters[oxy_neighbors] = mm + 1
            clusters[i] = mm + 1

        elif not (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            clusters[oxy_neighbors] = clusters[i]

        elif not (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
            tmp = clusters[oxy_neighbors]
            tmp = tmp[tmp != 0]
            tmp = tmp[tmp != min(tmp)]

            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            clusters[oxy_neighbors] = clusters[i]

            for tr in tmp:
                clusters[clusters == tr] = clusters[i]

    molecules = []
    for i in range(1, clusters.max() + 1):
        indices = np.where(clusters == i)[0]
        if indices.size > 0:
            molecules.append(indices)

    return molecules




def get_molecule_population_matrix(
    folder,
    xyz,
    cell,
    atom_names,
    atom_names_list,
    cutoff=3.25,
):
    """
    Function to get the population matrix of clusters across the whole
    simulation range
    Args:
        xyz - xyz coordinates of the simulation
        cell - 3x3 matrix of the cell size (PBC)
        atom_names - long atom names loaded from the read_xyz function
        cutoff - cutoff for measuring clustering (3.25 from France-Lanord and Molinari papers)
    Return:
        all_mol - repeating numpy array where every three arrays is the next timestep
            clustering information, 1st line is the cation amount in cluster, 2nd line
            is the anion amount in cluster, 3rd line is the cluster amount for those cation
            anion amounts
    """
    print("DEBUG entered get_molecule_population_matrix")

    print("DEBUG FULL atom_names list:")
    for idx, name in enumerate(atom_names):
        print(f"{idx}: {name}")

        print("DEBUG atom_names_list:", atom_names_list)

    xyz_msd = wrap_xyz(xyz, cell)
    salt_types = [i[1] for i in atom_names_list]
    salt_atoms = [i[0] for i in atom_names_list]
    atom_inxs = [
        cur.split("-")[1] in salt_types and cur.split("-")[0] in salt_atoms
        # cur.split("-")[1] in ["PL1", "CA1"] and cur.split("-")[0] in ["N", "Li"]
        for cur in atom_names
    ]

    cations = [cur.split("-")[1] == salt_types[0] for cur in np.array(atom_names)[atom_inxs]]
    cations = np.nonzero(cations)[0]
    orig_cations = [cur.split("-")[1] == salt_types[0] for cur in np.array(atom_names)]
    orig_cations = np.nonzero(orig_cations)[0]

    an1 = [cur.split("-")[1] == salt_types[1] for cur in np.array(atom_names)[atom_inxs]]
    # an1 = [cur.split("-")[1] == "PL1" for cur in np.array(atom_names)[atom_inxs]]
    an1 = np.nonzero(an1)[0]
    orig_an1 = [
        cur.split("-")[1] == salt_types[1] and cur.split("-")[0] == salt_atoms[1]
        # cur.split("-")[1] == "PL1" and cur.split("-")[0] == "N"
        for cur in np.array(atom_names)
    ]
    orig_an1 = np.nonzero(orig_an1)[0]

    all_ion_atoms = np.concatenate([an1.reshape(-1), cations])
    max_num = max(len(an1), len(cations))

    print("DEBUG all_ion_atoms shape:", all_ion_atoms.shape)
    print("DEBUG all_ion_atoms values:", all_ion_atoms)

    all_mol = []
    max_mols = 0
    for cur_frame in tqdm(xyz_msd):
        # for cur_frame in xyz_rdf[0:1]:
        # popmatrix will be cation, an1, an2
        popmatrix = np.zeros((max_num, max_num), dtype=float)
        edge_from, edge_to, offsets = torch_nbr_list(
            cur_frame[atom_inxs],
            cutoff,
            cell,
            directed=True,
            requires_large_offsets=True,
        )
        nbr_list = np.stack([edge_from, edge_to], axis=1)

         # Print neighbor list shape and max indices
        print("DEBUG nbr_list shape:", nbr_list.shape)
        if nbr_list.size > 0:
            print("DEBUG max index in nbr_list:", nbr_list.max())

        molecules = get_molecules(nbr_list, all_ion_atoms)

        for molecule in molecules:
            num_cat = len(set(molecule).intersection(set(cations)))
            num_an1 = len(set(molecule).intersection(set(an1.reshape(-1))))
            popmatrix[num_cat, num_an1] += 1
        all_mol.append(
            np.vstack(
                [
                    np.stack(np.nonzero(popmatrix), axis=1).transpose(1, 0),
                    popmatrix[np.nonzero(popmatrix)],
                ]
            )
        )
        max_mols = max(all_mol[-1].shape[1], max_mols)
    for cur in range(len(all_mol)):
        if all_mol[cur].shape[1] < max_mols:
            # Cation, anion, values
            all_mol[cur] = np.hstack(
                [
                    all_mol[cur],
                    np.zeros((all_mol[cur].shape[0], max_mols - all_mol[cur].shape[1])),
                ]
            )
    all_mol = np.array(all_mol, dtype=int)
    if folder:
        all_mol_csv = pd.DataFrame(all_mol.reshape(-1, all_mol.shape[-1]))
        all_mol_csv.to_csv(f"{folder}/clusters_1cat_2ani_3count.csv")

    return all_mol

##current from ffnet
def get_coords_PDB_msd(
    folder,
    pre_folder,
    atom_name_list,
    save_interval=5,  # ps
    repeat_units=None,
    cell=None,
):
    """
    THIS IS GOING TO BE COM'D NOW BIIIIIAAAA***
    """
    xyz = []
    atom_names = []
    frames = 0

    polymer_msd = False
    for i in atom_name_list:
        if "PL1" in i:
            polymer_msd = True
    if polymer_msd:
        pdb = PDBFile(f"{pre_folder}/polymer_conformation.pdb")
        atoms_poly = [i.element._symbol for i in pdb.topology.atoms()]
        poly_atom_ind = [
            i for i, e in enumerate(atoms_poly) if e in ["O", "S", "N", "Si", "Br"]
        ]
        # Either works with defining repeat_units as an argument to run_analysis
        # or reading a repeats.txt file that from 09/14/24 is saved from builder files
        # or having Jurgis naming conventions
        if not repeat_units:
            if os.path.exists(f"{pre_folder}/repeats.txt"):
                with open(f"{pre_folder}/repeats.txt", "r") as f:
                    repeat_units = int(f.readlines()[0])
            else:
                repeat_units = int(pre_folder.split("/")[-2].split("_")[1][1:])
        poly_counter = len(poly_atom_ind) // repeat_units
        atom_name_list[-1] = [atoms_poly[poly_atom_ind[0]], "PL1"]
    COMS = []
    cur_frame = []
    prev_frame = []
    masses = []
    with open(f"{folder}/simu_output.pdb", "r") as f:
        poly_ind = 0
        for ind, line in enumerate(f):
            if "MODEL" in line:
                if prev_frame:
                    prev_frame = np.array(prev_frame)
                    cur_frame = np.array(cur_frame)
                    cur_frame = unwrap(cur_frame, prev_frame, cell)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()
                if frames == 1:
                    cur_frame = np.array(cur_frame)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()

                prev_frame = cur_frame
                cur_frame = []
                frames += 1

            for atom, mol in atom_name_list:
                if mol in line and mol != "PL1":
                    values = line.split()
                    if atom.lower() == values[-1].lower() or values[3] == "CA1":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        xyz.append(np.array([x, y, z], dtype=np.float32))
                        if frames < 2:
                            # Harcoding CA1 as the only cation
                            if values[3] == "CA1":
                                atom_names.append(atom + f",{mol}")
                            else:
                                atom_names.append(values[-1] + f",{mol}")
                if mol == "PL1" and mol in line:
                    values = line.split()
                    if values[-1].lower() in ["o", "s", "n", "si", "br"]:
                        if poly_ind % poly_counter == 0:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            xyz.append(np.array([x, y, z], dtype=np.float32))
                            if frames < 2:
                                atom_names.append(values[-1] + f",{mol}")
                        poly_ind += 1

            if "PL1" in line or "CA1" in line or "AN1" in line:
                values = line.split()
                if "TER" == values[0]:
                    continue
                else:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cur_frame.append(np.array([x, y, z], dtype=np.float32))

                    if frames < 2:
                        if values[3] == "CA1":
                            masses.append(ELEMENT_TO_MASS["Li"])
                        else:
                            masses.append(ELEMENT_TO_MASS[values[-1]])

        # Calculating the COM of the final frame
        prev_frame = np.array(prev_frame)
        cur_frame = np.array(cur_frame)
        cur_frame = unwrap(cur_frame, prev_frame, cell)
        masses = np.array(masses)
        total_mass = np.sum(masses)
        COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)

    xyz = np.array(xyz)
    xyz = xyz.reshape(len(COMS), len(atom_names), 3)
    for ind, com in enumerate(COMS):
        xyz[ind] = xyz[ind] - com
    xyz = xyz.reshape(-1, 3)

    with open(f"{folder}/xyz_wrapped_msd.txt", "w") as f:
        for xyz in xyz:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/atom_names_msd.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")
    with open(f"{folder}/frame_count.txt", "w") as f:
        f.write(f"{frames-1}")  # This frames-1 might be completely unnecessary actually

    simu_time = (frames - 1) * save_interval / 1000
    print(
        f"{frames} MSD frames have been saved at {folder}, simulation time {simu_time}"
    )
    return frames - 1, simu_time

#this isn't functional yet, kyujong also needs to confirm that analysis is valid for taking the frame in our intervals
def get_coords_PDB_coordinating_atoms(
    folder,
    atom_name_list,
):
    xyz = []
    atom_names = []
    frames = 0

    COMS = []
    cur_frame = []
    prev_frame = []
    masses = []

    with open(f"{folder}/simu_output.pdb", "r") as f:
        poly_ind = 0
        for ind, line in enumerate(f):
            if "MODEL" in line:
                if prev_frame:
                    prev_frame = np.array(prev_frame)
                    cur_frame = np.array(cur_frame)
                    cur_frame = unwrap(cur_frame, prev_frame, cell)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()
                if frames == 1:
                    cur_frame = np.array(cur_frame)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()

                prev_frame = cur_frame
                cur_frame = []
                frames += 1

            for atom, mol in atom_name_list:
                if mol in line and mol != "PL1": #cation and anion
                    values = line.split()
                    if atom.lower() == values[-1].lower() or values[3] == "CA1":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        xyz.append(np.array([x, y, z], dtype=np.float32))
                        if frames < 2:
                            # Harcoding CA1 as the only cation
                            if values[3] == "CA1":
                                atom_names.append(atom + f",{mol}")
                            else:
                                atom_names.append(values[-1] + f",{mol}")
                if mol == "PL1" and mol in line: #polymers
                    values = line.split()
                    if values[-1].lower() in ["o", "s", "n", "si", "br"]:
                        
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            xyz.append(np.array([x, y, z], dtype=np.float32))
                            if frames < 2:
                                atom_names.append(values[-1] + f",{mol}")
                poly_ind += 1
            
            if "PL1" in line or "CA1" in line or "AN1" in line:
                values = line.split()
                if "TER" == values[0]:
                    continue
                else:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cur_frame.append(np.array([x, y, z], dtype=np.float32))

                    if frames < 2:
                        if values[3] == "CA1":
                            masses.append(ELEMENT_TO_MASS["Li"])
                        else:
                            masses.append(ELEMENT_TO_MASS[values[-1]])

        # Calculating the COM of the final frame
        prev_frame = np.array(prev_frame)
        cur_frame = np.array(cur_frame)
        cur_frame = unwrap(cur_frame, prev_frame, cell)
        masses = np.array(masses)
        total_mass = np.sum(masses)
        COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)

    xyz = np.array(xyz)
    xyz = xyz.reshape(len(COMS), len(atom_names), 3)
    for ind, com in enumerate(COMS):
        xyz[ind] = xyz[ind] - com
    xyz = xyz.reshape(-1, 3)

    with open(f"{folder}/xyz_wrapped_coordinating_atoms.txt", "w") as f:
        for xyz in xyz:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")                       
    with open(f"{folder}/atom_names_coordinating_atoms.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")

def get_coords_PDB_rdf_openmm(
    folder,
    frame_count,  # ns
    save_interval=5,  # ps
    frame_save_amount=100,
):
    """
    Saves unwrapped 100 xyz frames at the beginning of the simulation
    at the middle of it and at the end. This is for the simu_output.pdb from OpenMM
    """
    xyz_1 = []
    xyz_2 = []
    xyz_3 = []
    atom_names = []
    curr_frame = 0

    frame_intervals = [frame_save_amount, int(frame_count / 2), frame_count]

    exclude_tags = ["TER", "ENDMDL", "TITLE", "REMARK", "CRYST1", "MODEL"]
    with open(f"{folder}/simu_output.pdb", "r") as f:
        for ind, line in enumerate(f):
            if "MODEL" in line:
                curr_frame += 1
                continue
            if not any(ext in line for ext in exclude_tags):
                if (
                    curr_frame > frame_intervals[0] - frame_save_amount
                    and curr_frame < frame_intervals[0]
                ):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xyz_1.append(np.array([x, y, z], dtype=np.float32))
                    if curr_frame < 2:
                        values = line.split()
                        if values[3] == "CA1":
                            atom_names.append(
                                "Li" + f",{line[17:20]},{line[23:26].strip()}"
                            )
                        else:
                            atom_names.append(
                                values[-1] + f",{line[17:20]},{line[23:26].strip()}"
                            )
                elif (
                    curr_frame > frame_intervals[1] - frame_save_amount
                    and curr_frame < frame_intervals[1]
                ):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xyz_2.append(np.array([x, y, z], dtype=np.float32))
                elif (
                    curr_frame > frame_intervals[2] - frame_save_amount
                    and curr_frame < frame_intervals[2]
                ):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xyz_3.append(np.array([x, y, z], dtype=np.float32))
                elif curr_frame > frame_intervals[2]:
                    break

    with open(f"{folder}/xyz_wrapped_rdf_beginning.txt", "w") as f:
        for xyz in xyz_1:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/xyz_wrapped_rdf_middle.txt", "w") as f:
        for xyz in xyz_2:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/xyz_wrapped_rdf_end.txt", "w") as f:
        for xyz in xyz_3:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/atom_names_rdf.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")
    print(f"RDF frames have been saved at {folder}")


def get_coords_PDB_rdf_full_openmm(
    folder,
):
    atom_names = []
    frames = 0
    with open(f"{folder}/xyz_unwrapped_rdf_all.txt", "w") as outfile:
        with open(f"{folder}/simu_output.pdb", "r") as f:
            exclude_tags = ["TER", "ENDMDL", "TITLE", "REMARK", "CRYST1", "MODEL"]
            for ind, line in enumerate(f):
                if 'CONECT' in line:
                    break
                if "MODEL" in line:
                    frames += 1
                    continue

                if not any(ext in line for ext in exclude_tags):
                    values = line.split()
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        outfile.write(f"{x},{y},{z}\n")
                    except:
                        print('current line number and line:', ind)
                        print(line)
                        print(values)
                        raise ValueError('could not convert string to float:', line[30:38])
                    if frames < 2:
                        # Harcoding CA1 as the only cation
                        if values[3] == "CA1":
                            atom_names.append(
                                "Li" + f",{line[17:20]},{line[23:26].strip()}"
                            )
                        else:
                            atom_names.append(
                                values[-1] + f",{line[17:20]},{line[23:26].strip()}"
                            )
    with open(f"{folder}/atom_names_rdf_all.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")
    print(
        f"{frames} PDB frames have been saved at {folder} for xyz_unwrapped_rdf_all.txt"
    )




#this isn't functional yet, kyujong also needs to confirm that analysis is valid for taking the frame in our intervals
def get_coords_PDB_coordinating_atoms(
    folder,
    atom_name_list,
):
    xyz = []
    atom_names = []
    frames = 0

    COMS = []
    cur_frame = []
    prev_frame = []
    masses = []

    with open(f"{folder}/simu_output.pdb", "r") as f:
        poly_ind = 0
        for ind, line in enumerate(f):
            if "MODEL" in line:
                if prev_frame:
                    prev_frame = np.array(prev_frame)
                    cur_frame = np.array(cur_frame)
                    cur_frame = unwrap(cur_frame, prev_frame, cell)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()
                if frames == 1:
                    cur_frame = np.array(cur_frame)
                    masses = np.array(masses)
                    total_mass = np.sum(masses)
                    COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)
                    cur_frame = cur_frame.tolist()

                prev_frame = cur_frame
                cur_frame = []
                frames += 1

            for atom, mol in atom_name_list:
                if mol in line and mol != "PL1": #cation and anion
                    values = line.split()
                    if atom.lower() == values[-1].lower() or values[3] == "CA1":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        xyz.append(np.array([x, y, z], dtype=np.float32))
                        if frames < 2:
                            # Harcoding CA1 as the only cation
                            if values[3] == "CA1":
                                atom_names.append(atom + f",{mol}")
                            else:
                                atom_names.append(values[-1] + f",{mol}")
                if mol == "PL1" and mol in line: #polymers
                    values = line.split()
                    if values[-1].lower() in ["o", "s", "n", "si", "br"]:
                        
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            xyz.append(np.array([x, y, z], dtype=np.float32))
                            if frames < 2:
                                atom_names.append(values[-1] + f",{mol}")
                poly_ind += 1
            
            if "PL1" in line or "CA1" in line or "AN1" in line:
                values = line.split()
                if "TER" == values[0]:
                    continue
                else:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cur_frame.append(np.array([x, y, z], dtype=np.float32))

                    if frames < 2:
                        if values[3] == "CA1":
                            masses.append(ELEMENT_TO_MASS["Li"])
                        else:
                            masses.append(ELEMENT_TO_MASS[values[-1]])

        # Calculating the COM of the final frame
        prev_frame = np.array(prev_frame)
        cur_frame = np.array(cur_frame)
        cur_frame = unwrap(cur_frame, prev_frame, cell)
        masses = np.array(masses)
        total_mass = np.sum(masses)
        COMS.append(np.sum(cur_frame.T * masses, axis=1) / total_mass)

    xyz = np.array(xyz)
    xyz = xyz.reshape(len(COMS), len(atom_names), 3)
    for ind, com in enumerate(COMS):
        xyz[ind] = xyz[ind] - com
    xyz = xyz.reshape(-1, 3)

    with open(f"{folder}/xyz_wrapped_coordinating_atoms.txt", "w") as f:
        for xyz in xyz:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")                       
    with open(f"{folder}/atom_names_coordinating_atoms.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")


def save_params(folder_path):
    # Find the final_state file
    for i in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, i)) and "final_state_box" in i:
            final_state_name = i

    with open(f"{folder_path}/{final_state_name}") as input:
        final_state = XmlSerializer.deserialize(input.read())

    cell = np.zeros((3, 3))
    for indi, i in enumerate(final_state.getPeriodicBoxVectors()._value):
        for indj, j in enumerate(i):
            cell[indi, indj] = j
    if final_state.getPeriodicBoxVectors().unit.get_name() == "nanometer":
        cell = cell * 10

    return cell


def save_gromacs_params(folder):
    cell = None
    with open(f"{folder}/simu_output.pdb", "r") as f:
        for line in f:
            if "CRYST" in line:
                cell = line
            if "MODEL" in line:
                break
    if not cell:
        raise ValueError("Cell dimensions were not found")

    cell = np.diag(np.array(cell.split()[1:4], dtype=np.float32))
    np.save(f"{folder}/cell_dimensions", cell)
    with open(f"{folder}/box_len.txt", 'w') as outfile:
        outfile.write(str(cell[0][0]))
    return cell


def read_xyz(
    folder_path,
    atom_name,
    xyz_name,
    include_resid=False,
):
    with open(f"{folder_path}/{atom_name}", "r") as f:
        atom_names = []
        atom_names_long = []
        res_ids = []
        for line in f:
            value = line.split(",")
            atom_names.append(value[0])
            if include_resid:  # can only be used with atom_names_rdf.txt
                res_ids.append(f"{value[1]} {int(value[2])}")
            atom_names_long.append(value[0] + "-" + value[1].strip())
    with open(f"{folder_path}/{xyz_name}", "r") as f:
        xyz = []
        for line in f:
            values = line.split(",")
            xyz.append(np.array(values[0:3], dtype=np.float64))
    xyz = np.concatenate(xyz).reshape(-1, len(atom_names), 3)
    if include_resid:
        return xyz, atom_names, atom_names_long, res_ids
    else:
        return xyz, atom_names, atom_names_long


def plot_calc_diffu(
    xyz,
    folder,
    save_freq,
    diffu_time,
    cat_name: list,
    ani_name: list,
    cell,
    temperature,
    atom_names,
    name=None,
    solv_name=[],
    poly_name: list = [],
    atom_names_list: list = [],
):
    """
    save_freq - ps
    """
    print("DEBUG: Unique atom names found in trajectory:")
    print(set(atom_names))
    print("DEBUG: Cation names being searched:", cat_name)
    print("DEBUG: Anion names being searched:", ani_name)
    print("DEBUG: Polymer names being searched:", poly_name)
    print("DEBUG: Solvent names being searched:", solv_name)


    cat_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == cat] for cat in cat_name
    ]
    ani_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == ani] for ani in ani_name
    ]
    solv_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == solv] for solv in solv_name
    ]
    poly_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == poly] for poly in poly_name
    ]

    num_frames = xyz.shape[0]
    t = np.arange(num_frames // 2) * save_freq
    np.save(f"{folder}/MSD_multi_origin_t", t)

    # Diffusivity / MSD
    multi_origin_step = xyz.shape[0] * save_freq // 1000 // 5

    figsize = (5, 4)
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

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=False, sharey=False)

    cat_color = ["darkorange", "saddlebrown", "darkred"]
    ani_color = ["blue", "navy", "darkviolet"]
    solv_color = ["dimgrey", "darkgrey", "silver"]
    poly_color = ["forestgreen", "green", "darkolivegreen"]
    #####################################
    # Cation
    # zero origin
    D_cat_list = []
    D_cat_err_list = []
    m_cat_loglog = []
    for ind, cat_idxs in enumerate(cat_idxs_list):
        start_cat = xyz[0, cat_idxs]
        displ_vec = xyz[:, cat_idxs] - start_cat
        MSDs = np.power(displ_vec, 2).sum(axis=2)
        # np.save(f"{folder}/MSD_single_origin_{cat_name}.npy", MSDs)

        displ_vec = xyz[: num_frames // 2, cat_idxs] - start_cat
        MSDs = [np.power(displ_vec, 2).sum(axis=2)]
        # other origins
        for origin in tqdm(
            np.arange(multi_origin_step, num_frames // 2, multi_origin_step)
        ):
            start_cat = xyz[origin, cat_idxs]
            displ_vec = xyz[origin : origin + num_frames // 2, cat_idxs] - start_cat
            this_SD = np.power(displ_vec, 2).sum(axis=2)
            # MSDs = np.concatenate((MSDs, this_SD), axis=1)
            MSDs.append(this_SD)

        MSDs = np.array(MSDs)
        MSD_multi_origin = MSDs.mean(axis=2).mean(axis=0)
        np.save(f"{folder}/MSD_multi_origin_{cat_name[ind]}.npy", MSD_multi_origin)
        if xyz.shape[0] * save_freq // 1000 <= 10:
            slope_time_window = 100
            time_width = slope_time_window // (t[1] - t[0])
            # select windows around 0.1ns for the smaller simulations
        else:
            # Selecting always 150 windows if the simulation is long enough
            time_width = len(t) // 150
        diffu_inx, m_loglog = find_slope1_indeces(
            t[::time_width], MSD_multi_origin[::time_width]
        )
        print("cation slope time:", t[::time_width][diffu_inx] / 1000)
        popt, pcov = curve_fit(
            lin_func,
            t[::time_width][diffu_inx],
            MSD_multi_origin[::time_width][diffu_inx],
            p0=[0.0, 1.0],
        )
        m_cat_loglog.append(m_loglog)

        D_cat = (popt[0] / 6) * 1e-4  # conversion from Angstrom^2/ps to cm^2/s
        D_cat_list.append(D_cat)
        D_cat_err = (pcov[0, 0] / 6) * 1e-4
        D_cat_err_list.append(D_cat_err)
        ax.plot(
            t / 1000.0,
            MSD_multi_origin,
            linewidth=2,
            color=cat_color[ind],
            label="$D_\mathrm{%s}$ = %1.3e cm$^2$/s"
            % (f"{cat_name[ind]}", D_cat_list[ind]),
        )
        ax.plot(
            t / 1000.0, lin_func(t, *popt), linewidth=1, linestyle="--", color="black"
        )
        mask = (t >= t[::time_width][diffu_inx][0]) & (
            t <= t[::time_width][diffu_inx][-1]
        )
        ax.plot(
            t[mask] / 1000.0,
            MSD_multi_origin[mask],
            linewidth=2,
            color="black",
            label="Cat LogLog slope = %1.3e" % (m_loglog),
        )
    del displ_vec, MSDs, MSD_multi_origin
    #####################################
    # Anion
    # zero origin
    D_ani_list = []
    D_ani_err_list = []
    m_ani_loglog = []
    for ind, ani_idxs in enumerate(ani_idxs_list):
        if len(ani_idxs) == 0:
            print("there are none of anion", ind, "in this simulation")
            continue
        start_ani = xyz[0, ani_idxs]
        displ_vec = xyz[:, ani_idxs] - start_ani
        MSDs = np.power(displ_vec, 2).sum(axis=2)
        # np.save(f"{folder}/MSD_single_origin_{ani_name}.npy", MSDs)

        displ_vec = xyz[: num_frames // 2, ani_idxs] - start_ani
        MSDs = [np.power(displ_vec, 2).sum(axis=2)]
        # other origins
        for origin in tqdm(
            np.arange(multi_origin_step, num_frames // 2, multi_origin_step)
        ):
            start_ani = xyz[origin, ani_idxs]
            displ_vec = xyz[origin : origin + num_frames // 2, ani_idxs] - start_ani
            this_SD = np.power(displ_vec, 2).sum(axis=2)
            # MSDs = np.concatenate((MSDs, this_SD), axis=1)
            MSDs.append(this_SD)

        MSDs = np.array(MSDs)
        MSD_multi_origin = MSDs.mean(axis=2).mean(axis=0)
        np.save(f"{folder}/MSD_multi_origin_{ani_name[ind]}.npy", MSD_multi_origin)
        diffu_inx, m_loglog = find_slope1_indeces(
            t[::time_width], MSD_multi_origin[::time_width]
        )
        popt, pcov = curve_fit(
            lin_func,
            t[::time_width][diffu_inx],
            MSD_multi_origin[::time_width][diffu_inx],
            p0=[0.0, 1.0],
        )
        m_ani_loglog.append(m_loglog)
        D_ani = (popt[0] / 6) * 1e-4  # conversion from Angstrom^2/ps to cm^2/s
        D_ani_list.append(D_ani)
        D_ani_err = (pcov[0, 0] / 6) * 1e-4
        D_ani_err_list.append(D_ani_err)
        ax.plot(
            t / 1000.0,
            MSD_multi_origin,
            linewidth=2,
            color=ani_color[ind],
            label="$D_\mathrm{%s}$ = %1.3e cm$^2$/s"
            % (f"{ani_name[ind]}", D_ani_list[ind]),
        )
        ax.plot(
            t / 1000.0, lin_func(t, *popt), linewidth=1, linestyle="--", color="black"
        )
        mask = (t >= t[::time_width][diffu_inx][0]) & (
            t <= t[::time_width][diffu_inx][-1]
        )
        ax.plot(
            t[mask] / 1000.0,
            MSD_multi_origin[mask],
            linewidth=2,
            color="black",
            label="Ani LogLog slope = %1.3e" % (m_loglog),
        )
    del displ_vec, MSDs, MSD_multi_origin
    #####################################
    # Solvent
    # zero origin
    D_solv_list = []
    D_solv_err_list = []
    m_solv_loglog = []
    for ind, solv_idxs in enumerate(solv_idxs_list):
        if len(solv_idxs) == 0:
            print("there are none of solvent", ind, "in this simulation")
            continue
        start_solv = xyz[0, solv_idxs]
        displ_vec = xyz[:, solv_idxs] - start_solv
        MSDs = np.power(displ_vec, 2).sum(axis=2)
        # np.save(f"{folder}/MSD_single_origin_{solv_name}.npy", MSDs)

        displ_vec = xyz[: num_frames // 2, solv_idxs] - start_solv
        MSDs = [np.power(displ_vec, 2).sum(axis=2)]
        # other origins
        for origin in tqdm(
            np.arange(multi_origin_step, num_frames // 2, multi_origin_step)
        ):
            start_solv = xyz[origin, solv_idxs]
            displ_vec = xyz[origin : origin + num_frames // 2, solv_idxs] - start_solv
            this_SD = np.power(displ_vec, 2).sum(axis=2)
            # MSDs = np.concatenate((MSDs, this_SD), axis=1)
            MSDs.append(this_SD)

        MSDs = np.array(MSDs)
        MSD_multi_origin = MSDs.mean(axis=2).mean(axis=0)
        np.save(f"{folder}/MSD_multi_origin_{solv_name[ind]}.npy", MSD_multi_origin)

        diffu_inx, m_loglog = find_slope1_indeces(
            t[::time_width], MSD_multi_origin[::time_width]
        )
        popt, pcov = curve_fit(
            lin_func,
            t[::time_width][diffu_inx],
            MSD_multi_origin[::time_width][diffu_inx],
            p0=[0.0, 1.0],
        )
        m_solv_loglog.append(m_loglog)
        D_solv = (popt[0] / 6) * 1e-4  # conversion from Angstrom^2/ps to cm^2/s
        D_solv_list.append(D_solv)
        D_solv_err = (pcov[0, 0] / 6) * 1e-4
        D_solv_err_list.append(D_solv_err)
        ax.plot(
            t / 1000.0,
            MSD_multi_origin,
            linewidth=2,
            color=solv_color[ind],
            label="$D_\mathrm{%s}$ = %1.3e cm$^2$/s"
            % (f"{solv_name[ind]}", D_solv_list[ind]),
        )
        ax.plot(
            t / 1000.0, lin_func(t, *popt), linewidth=1, linestyle="--", color="black"
        )
        del displ_vec, MSDs, MSD_multi_origin

    D_poly_list = []
    D_poly_err_list = []
    m_poly_loglog = []
    for ind, poly_idxs in enumerate(poly_idxs_list):
        start_poly = xyz[0, poly_idxs]
        displ_vec = xyz[:, poly_idxs] - start_poly
        MSDs = np.power(displ_vec, 2).sum(axis=2)
        # np.save(f"{folder}/MSD_single_origin_{solv_name}.npy", MSDs)

        displ_vec = xyz[: num_frames // 2, poly_idxs] - start_poly
        MSDs = [np.power(displ_vec, 2).sum(axis=2)]
        # other origins
        for origin in tqdm(
            np.arange(multi_origin_step, num_frames // 2, multi_origin_step)
        ):
            poly_solv = xyz[origin, poly_idxs]
            displ_vec = xyz[origin : origin + num_frames // 2, poly_idxs] - poly_solv
            this_SD = np.power(displ_vec, 2).sum(axis=2)
            # MSDs = np.concatenate((MSDs, this_SD), axis=1)
            MSDs.append(this_SD)

        MSDs = np.array(MSDs)
        MSD_multi_origin = MSDs.mean(axis=2).mean(axis=0)
        np.save(f"{folder}/MSD_multi_origin_{poly_name[ind]}.npy", MSD_multi_origin)

        diffu_inx, m_loglog = find_slope1_indeces(
            t[::time_width], MSD_multi_origin[::time_width]
        )
        popt, pcov = curve_fit(
            lin_func,
            t[::time_width][diffu_inx],
            MSD_multi_origin[::time_width][diffu_inx],
            p0=[0.0, 1.0],
        )
        m_poly_loglog.append(m_loglog)
        D_poly = (popt[0] / 6) * 1e-4  # conversion from Angstrom^2/ps to cm^2/s
        D_poly_list.append(D_poly)
        D_poly_err = (pcov[0, 0] / 6) * 1e-4
        D_poly_err_list.append(D_poly_err)
        ax.plot(
            t / 1000.0,
            MSD_multi_origin,
            linewidth=2,
            color=poly_color[ind],
            label="$D_\mathrm{%s}$ = %1.3e cm$^2$/s"
            % (f"{poly_name[ind]}", D_poly_list[ind]),
        )
        ax.plot(
            t / 1000.0, lin_func(t, *popt), linewidth=1, linestyle="--", color="black"
        )
        with open(f"{folder}/poly_diffu_{poly_name[ind]}.txt", "w") as f:
            f.write(str(D_poly))

        del displ_vec, MSDs, MSD_multi_origin

    ax.tick_params(
        axis="y",
        length=maj_tick_size,
        width=tickwidth,
        labelsize=labelsize,
        pad=pad,
        direction="in",
    )
    ax.tick_params(
        axis="x",
        length=maj_tick_size,
        width=tickwidth,
        labelsize=labelsize,
        pad=pad,
        direction="in",
    )

    for key in ax.spines.keys():
        ax.spines[key].set_linewidth(tickwidth)

    ax.set_xlabel("$t$ / ns", fontsize=labelsize)
    ax.set_ylabel("MSD($t$) / \AA$^2$", fontsize=labelsize)
    if not name:
        name = "_".join(folder.split("/")[-3].split("_")[:6])
        title_name = f"{name}, T{int(temperature)}"
    else:
        title_name = f"{name}, T{int(temperature)}"
    ax.set_title(f"{title_name}", fontsize=titelsize)

    volume = cell[0, 0] * cell[1, 1] * cell[2, 2]
    cat_idxs_len = len([i for j in cat_idxs_list for i in j])
    diffu_total = 0
    # nernst einstein conductivity is sum of num_charge_carriers*D_{charge_carrier}
    conductivity = 0
    with open(f"{folder}/cond.txt", "w") as f:
        for i, D_cat in enumerate(D_cat_list):
            print(
                "diffusivity of",
                cat_name[i],
                D_cat,
                "with this many:",
                len(cat_idxs_list[i]),
            )
            f.write(
                f"diffusivity of {cat_name[i]} is {D_cat}, with this many ions: {len(cat_idxs_list[i]), 'with linearity:', {m_cat_loglog[i]}}\n"
            )
            conductivity += D_cat * len(cat_idxs_list[i])
        for i, D_ani in enumerate(D_ani_list):
            print(
                "diffusivity of",
                ani_name[i],
                D_ani,
                "with this many:",
                len(ani_idxs_list[i]),
            )
            f.write(
                f"diffusivity of {ani_name[i]} is {D_ani}, with this many ions: {len(ani_idxs_list[i]), 'with linearity:', {m_ani_loglog[i]}}\n"
            )
            conductivity += D_ani * len(ani_idxs_list[i])
        for i, D_solv in enumerate(D_solv_list):
            print(
                "diffusivity of",
                solv_name[i],
                D_solv,
                "with this many:",
                len(solv_idxs_list[i]),
            )
            f.write(
                f"diffusivity of {solv_name[i]} is {D_solv}, with this many ions: {len(solv_idxs_list[i]), 'with linearity:', {m_solv_loglog[i]}}\n"
            )
        for i, D_poly in enumerate(D_poly_list):
            print(
                "diffusivity of",
                poly_name[i],
                D_poly,
                "with this many:",
                len(poly_idxs_list[i]),
            )
            f.write(
                f"diffusivity of {poly_name[i]} is {D_poly}, with this many polymer segments: {len(poly_idxs_list[i]), 'with linearity:', {m_poly_loglog[i]}}\n"
            )
        conductivity = (ELEC**2) / (volume * KB * temperature) * (conductivity) * 1e24
        print("conductivity is [S/cm]:", conductivity)

        for ind, cat_log in enumerate(m_cat_loglog):
            f.write(f"loglog_cat_{ind} - " + str(float(cat_log)) + "\n")
        for ind, ani_log in enumerate(m_ani_loglog):
            f.write(f"loglog_ani_{ind} - " + str(float(ani_log)) + "\n")
        for ind, poly_log in enumerate(m_poly_loglog):
            f.write(f"loglog_poly_{ind} - " + str(float(poly_log)) + "\n")

        for ind, cat_err in enumerate(D_poly_err_list):
            f.write(f"diffu_err_cat_{ind} - " + str(float(cat_err)) + "\n")
        for ind, ani_err in enumerate(D_ani_err_list):
            f.write(f"diffu_err_ani_{ind} - " + str(float(ani_err)) + "\n")
        for ind, poly_err in enumerate(D_poly_err_list):
            f.write(f"diffu_err_poly_{ind} - " + str(float(poly_err)) + "\n")

        f.write(f"Conductivity - {str(conductivity)}")

    ax.text(
        0.6,
        0.08,
        "Conductivity = {:1.2e} $S$/$cm$".format(conductivity),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=labelsize,
    )

    plt.legend(fontsize=int(1 / 2 * labelsize))
    plt.tight_layout()
    plt.savefig(f"{folder}/Diffusitivty_fit.svg", dpi=dpi)
    plt.savefig(f"{folder}/Diffusitivty_fit.png", dpi=dpi)
    plt.show()
    plt.close()
    print("getting population matrix:")
    print(f"atom_names_list being inserted {atom_names_list}")
    all_mol = get_molecule_population_matrix(
        folder=folder,
        xyz=xyz,
        cell=cell,
        atom_names=atom_names,
        atom_names_list=atom_names_list,
        cutoff=3.25,
    )

    plot_clusters_cond(
        folder=folder,
        all_mol=all_mol,
        cell=cell,
        D_cat=D_cat_list[0],
        D_ani=D_ani_list[0],
        temperature=temperature,
    )


def plot_clusters_cond(
    folder,
    all_mol,
    cell,
    D_cat,
    D_ani,
    temperature,
):
    """
    Plotting the cluster population statistics across the simulation time and
        saves the cNE conductivity across the simulation time (as a change of
        cluster population matrix)
    Args:
        Folder - path to where to save figures
        All_mol - cluster population array from get_molecule_population_matrix
        D_cat - cation difffusivity from NE
        D_ani - anion difffusivity from NE
    """
    figsize = (5, 4)
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

    # Create the imshow of the clusters in the simulation
    max_amount = all_mol[:, :2, :].flatten().max()
    imshow_matr = np.zeros((max_amount + 1, max_amount + 1))
    indeces_to_sample = [0, all_mol.shape[0] // 2, all_mol.shape[0] - 100]
    names = ["beginning", "middle", "end"]
    cNE_cond = []
    cNE_tn = []
    neutral_clusters = []
    positive_clusters = []
    negative_clusters = []
    freelithium = []
    freetfsi = []
    for sample_time, name in zip(indeces_to_sample, names):
        for t in all_mol[int(sample_time) : int(sample_time) + 100]:
            for i in range(all_mol.shape[-1]):
                indi = int(t[0, i])
                indj = int(t[1, i])
                imshow_matr[indi, indj] += t[2, i]

        imshow_matr = imshow_matr / 100
        imshow_matr = np.round(imshow_matr, 2)

        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=False, sharey=False)
        imshow_matr = imshow_matr[::-1, :]
        im = ax.imshow(imshow_matr)

        for i in range(imshow_matr.shape[0]):
            for j in range(imshow_matr.shape[1]):
                text = ax.text(
                    j,
                    i,
                    imshow_matr[i, j],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=int(2 * labelsize / 3),
                )
        ax.set_ylabel("Cation", fontsize=labelsize)
        ax.set_xlabel("Anion", fontsize=labelsize)
        ax.set_xticks(np.arange(0, max_amount + 1, 1))
        ax.set_yticks(np.arange(max_amount + 1))
        ax.set_yticklabels(np.arange(max_amount + 1)[::-1])
        ax.tick_params(axis="both", labelsize=int(2 * labelsize / 3))
        plot_name = "_".join(folder.split("/")[-3].split("_")[:6])
        ax.set_title(
            f"Cluster pop at {name} of simu, {plot_name}, T={temperature}",
            fontsize=int(2 * labelsize / 3),
        )
        plt.tight_layout()
        if folder:
            plt.savefig(f"{folder}/cluster_pop_{name}.svg", dpi=dpi)
            plt.savefig(f"{folder}/cluster_pop_{name}.png", dpi=dpi)
        plt.show()
        plt.close()

        # this part writes the amount of clusters with neutral, positive or negative charge
        numbers_clusters = np.sum(imshow_matr)

        freelithium_occur = 0
        neutral_occur = 0
        positive_occur = 0
        negative_occur = 0
        freetfsi_occur = 0
        for i in range(imshow_matr.shape[0]):
            t = imshow_matr.shape[0] - 1 - i
            for j in range(imshow_matr.shape[1]):
                if j == 0 and t > j:
                    freelithium_occur += imshow_matr[i][j] / numbers_clusters
                elif t == 0 and j > t:
                    freetfsi_occur += imshow_matr[i][j] / numbers_clusters
                elif t > j:
                    positive_occur += imshow_matr[i][j] / numbers_clusters
                elif t == j:
                    neutral_occur += imshow_matr[i][j] / numbers_clusters
                elif j > t:
                    negative_occur += imshow_matr[i][j] / numbers_clusters
        neutral_clusters.append(neutral_occur)
        positive_clusters.append(positive_occur)
        negative_clusters.append(negative_occur)
        freelithium.append(freelithium_occur)
        freetfsi.append(freetfsi_occur)

        doubSum, alphasum, numerator = 0, 0, 0
        volume = np.prod((cell.diagonal() * 1e-10))
        for ncat, cur_cat in enumerate(imshow_matr[::-1]):
            for nani, cur_pop in enumerate(cur_cat):
                D_ij = 0
                z_ij = ncat - nani
                if ncat > nani:
                    D_ij = D_cat
                elif nani > ncat:
                    D_ij = D_ani

                numerator += (z_ij * ncat) * cur_pop.item() * D_ij * 1e-4
                doubSum += (z_ij**2) * cur_pop.item() * D_ij * 1e-4
                alphasum += cur_pop.item() * (ncat + nani)
        cNE0 = (ELEC**2 / (volume * KB * temperature)) * doubSum  # S/m
        cNE0 = cNE0 / 100  # S/cm
        cNE_cond.append(cNE0)
        cNE_tn.append(numerator / doubSum)
        print("Clustered Nernst Einstein [S/cm]:", cNE0)
    if folder:
        with open(f"{folder}/cond_cNE.txt", "w") as f:
            f.write("simu_interval,conductivity,transference_number\n")
            for name, cond, tn in zip(names, cNE_cond, cNE_tn):
                f.write(f"{name},{cond},{tn}\n")

        with open(f"{folder}/count_clusters.txt", "w") as f:
            f.write(
                "free_cation,positive_clusters,neutral_clusters,free_anion,negative_clusters\n"
            )
            for lit, pos, neutr, tfsi, neg in zip(
                freelithium,
                positive_clusters,
                neutral_clusters,
                freetfsi,
                negative_clusters,
            ):
                f.write(f"{lit},{pos},{neutr},{tfsi},{neg}\n")


def plot_calc_rdf(
    xyz_rdf_unwrp,
    folder,
    one_name: list,
    two_names: list,
    cell,
    atom_names_long_rdf,
    names: list = None,
    temperature=353,
    plot_names: list = None,
):
    coord_numbers = []
    save_names = []
    for i, two_name in enumerate(two_names):
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

        axs2 = []
        min_idx = 70
        idxs_one = []
        idxs_two = []

        for ind, atom in enumerate(atom_names_long_rdf):
            for name in one_name:
                if name == atom:
                    idxs_one.append(ind)
            # if one_name == two_name:
            for name in two_name:
                if name == atom:
                    idxs_two.append(ind)

        if len(idxs_one) == 0 or len(idxs_two) == 0:
            print(
                "the combination of",
                one_name,
                "and",
                two_name,
                "is not in this simulation",
            )
            continue
        # fig, subplot_dict = plt.subplot(mosaic_str, figsize=figsize, sharex=True, sharey=False)
        fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=False, sharey=False)
        num_frames = xyz_rdf_unwrp.shape[0]
        cell_lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
        idxs = np.array(list(itertools.product(idxs_one, idxs_two)))
        dist_mat = np.zeros(shape=(num_frames, idxs.shape[0]), dtype=np.float32)
        for ii, idx in enumerate(tqdm(idxs)):
            if idx[0] == idx[1]:
                dist_mat[:, ii] = 1e10
                continue
            dist_mat[:, ii] = mini_dist(
                xyz_rdf_unwrp[:, idx[0]], xyz_rdf_unwrp[:, idx[1]], cell_lengths
            )
        dist_mat = dist_mat.reshape((num_frames, len(idxs_one), len(idxs_two)))
        dx = 0.05
        V = cell_lengths[0] * cell_lengths[1] * cell_lengths[2]
        if one_name == two_name:
            centers, gr, coord = RDF(
                dist_mat, dx=dx, Na=len(idxs_one), Nb=len(idxs_two) - 1, V=V
            )
        else:
            centers, gr, coord = RDF(
                dist_mat, dx=dx, Na=len(idxs_one), Nb=len(idxs_two), V=V
            )
        center_val = centers[min_idx]
        coord_val = coord[min_idx]
        coord_numbers.append(coord_val)
        if names:
            save_names.append(names[i])
        else:
            save_names.append("_".join(two_name))

        axs.plot(centers, gr, linewidth=3, label=r"$g(r)$", color="blue")

        ax02 = axs.twinx()
        axs2.append(ax02)
        ax02.plot(
            centers, coord, linewidth=3, label=r"$\int dr g(r)$", color="darkorange"
        )

        axs.text(
            0.375,
            0.9,
            "CN(%1.1f \AA): %1.2f" % (center_val, coord_val),
            transform=axs.transAxes,
            fontsize=labelsize * 0.8,
        )
        axs.set_ylim([0, gr.max() * 1.1])

        if not plot_names:
            temp_name = "_".join(folder.split("/")[-3].split("_")[:6])
            title_name = f"{temp_name}, T{int(temperature)}, {names[i]}"
        else:
            title_name = f"{plot_names[i]}, T{int(temperature)}, {names[i]}"
        axs.set_title(f"{title_name}", fontsize=titelsize)

        for ax in [axs]:
            ax.set_xlim([0, 10])
            ax.axhline(1.0, linewidth=2, color="black")
            ax.axvline(3.5, linewidth=1, color="grey", linestyle="--")

            ax.tick_params(
                axis="y",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
                labelcolor="blue",
            )
            ax.tick_params(
                axis="x",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
            )

            for key in ax.spines.keys():
                ax.spines[key].set_linewidth(tickwidth)

        for ax in axs2:
            ax.tick_params(
                axis="y",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
                labelcolor="darkorange",
            )
            ax.set_ylim([0, 10])

        plt.tight_layout()
        if names:
            plt.savefig(f"{folder}/RDF_{names[i]}.svg", dpi=dpi)
            plt.savefig(f"{folder}/RDF_{names[i]}.png", dpi=dpi)
        else:
            plt.savefig(f"{folder}/RDF_{i}.svg", dpi=dpi)
            plt.savefig(f"{folder}/RDF_{i}.png", dpi=dpi)
        plt.show()

    if folder:
        duration = names[0].split("_")[-1]
        with open(f"{folder}/coord_vals_{duration}.txt", "w") as f:
            f.write("names,coordination_number\n")
            for name, coord in zip(names, coord_numbers):
                f.write(f"{name},{coord},\n")


def plot_calc_corr(
    xyz,
    folder,
    save_freq,
    cat_name: list,
    ani_name: list,
    cell,
    temperature,
    atom_names,
    num_windows=100,
    name=None,
):
    num_frames = xyz.shape[0]

    figsize = (10, 4)
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

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=False)

    cat_color = ["darkorange", "saddlebrown", "darkred"]
    ani_color = ["blue", "navy", "darkviolet"]

    # treat all anions the same for now, so cannot get ani1-ani2 cross,
    # but that is not needed for now, just the total wheeler newmann conductivity
    cat_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == cat] for cat in cat_name
    ]
    ani_idxs_list = [
        [ind for ind, i in enumerate(atom_names) if i == ani] for ani in ani_name
    ]
    for ind, cat_idxs in enumerate(cat_idxs_list):
        num_cations = len(cat_idxs)

        t = np.arange(num_frames // 2) * save_freq
        displ_vec = xyz[1:] - xyz[:-1]
        corr_matrix = compute_fft_onsager(displ_vec)
        total_ani = corr_matrix[ani_idxs_list[0]][:, ani_idxs_list[0], :].transpose(
            2, 0, 1
        )[: num_frames // 2]
        total_cat = corr_matrix[cat_idxs_list[0]][:, cat_idxs_list[0], :].transpose(
            2, 0, 1
        )[: num_frames // 2]
        catani = corr_matrix[cat_idxs_list[0]][:, ani_idxs_list[0], :].transpose(
            2, 0, 1
        )[: num_frames // 2]
        cat_selfs = get_self(total_cat)
        ani_selfs = get_self(total_ani)
        cat_cross = get_cross(total_cat)
        ani_cross = get_cross(total_ani)
        total_ani = total_ani.sum(axis=(1, 2))
        total_cat = total_cat.sum(axis=(1, 2))
        catani = catani.sum(axis=(1, 2))
        ### ends here

        if xyz.shape[0] * save_freq // 1000 <= 10:
            slope_time_window = 100
            time_width = slope_time_window // (t[1] - t[0])
            # make it so that the slope windwos are every 0.1ns in find_slope1
        else:
            slope_time_window = 1000
            time_width = len(t) // 150

        diffu_inxs = []
        m_loglogs = []
        corr_slopes = []
        diffu_errs = []

        # Calculating the slopes for the correlations
        for corr in [total_cat, total_ani, catani]:
            corr = corr / (num_cations**2)
            diffu_inx, m_loglog = find_slope1_indeces(
                timesteps_delTs=t[::time_width], total_corr=corr[::time_width]
            )

            print(
                "Total corr slope time:", t[::time_width][diffu_inx] / slope_time_window
            )
            popt, pcov = curve_fit(
                lin_func,
                t[::time_width][diffu_inx],
                corr[::time_width][diffu_inx],
                p0=[0.0, 1.0],
            )
            diffu_inxs.append(diffu_inx)
            m_loglogs.append(m_loglog)
            corr_slopes.append(popt)
            diffu_errs.append(pcov)

        print("saving correlations to:", f"{folder}/correlations.pkl")
        with open(f"{folder}/correlations.pkl", "wb") as pklfile:
            pkl.dump(
                {
                    "time_intervals": t,
                    "save_freq": save_freq,
                    "self_cat": cat_selfs,
                    "cross_cat": cat_cross,
                    "self_ani": ani_selfs,
                    "cross_ani": ani_cross,
                    "total": total_cat + total_ani,
                    "catani": catani,
                },
                pklfile,
            )

        volume = cell[0, 0] * cell[1, 1] * cell[2, 2]
        # Calculating the onsanger coefficients for each part of the matrix
        l_cat = 1 / (6 * volume * 1e-24 * KB * temperature) * (corr_slopes[0][0] * 1e-4)
        l_ani = 1 / (6 * volume * 1e-24 * KB * temperature) * (corr_slopes[1][0] * 1e-4)
        l_catani = (
            1 / (6 * volume * 1e-24 * KB * temperature) * (corr_slopes[2][0] * 1e-4)
        )

        condWN = (ELEC**2) * (
            l_cat * num_cations**2
            + l_ani * num_cations**2
            - 2 * l_catani * num_cations**2
        )
        with open(f"{folder}/conductivity_wh.txt", "w") as f:
            f.write("conductivity - " + str(float(condWN)) + "\n")
            f.write("slope_cat - " + str(float(corr_slopes[0][0] * 1e-4)) + "\n")
            f.write("slope_ani - " + str(float(corr_slopes[1][0] * 1e-4)) + "\n")
            f.write("slope_catani - " + str(float(corr_slopes[2][0] * 1e-4)) + "\n")
            f.write("loglog_cat - " + str(float(m_loglogs[0])) + "\n")
            f.write("loglog_ani - " + str(float(m_loglogs[1])) + "\n")
            f.write("loglog_catani - " + str(float(m_loglogs[2])) + "\n")
            f.write("err_cat - " + str(float(diffu_errs[0][0, 0])) + "\n")
            f.write("err_ani - " + str(float(diffu_errs[1][0, 0])) + "\n")
            f.write("err_catani - " + str(float(diffu_errs[2][0, 0])) + "\n")

        # Calculating the transference number (L++ - L+-)/(L++ + L-- - 2*L+-)
        cat_transf = (l_cat * num_cations**2 - l_catani * num_cations**2) / (
            l_cat * num_cations**2
            + l_ani * num_cations**2
            - 2 * l_catani * num_cations**2
        )
        with open(f"{folder}/cat_transference.txt", "w") as f:
            f.write(str(float(cat_transf)))

        for ax in axs:
            ax.plot(
                t / 1000.0,
                total_cat + total_ani + catani,
                linewidth=1,
                color="red",
                label="Total correlation",
            )
            ax.plot(
                t / 1000.0,
                catani,
                linewidth=1,
                color="purple",
                label="Cat-Ani correlation",
            )

        axs[0].plot(
            t / 1000.0,
            cat_selfs,
            linewidth=1,
            color=cat_color[ind],
            label="Cat self correlation",
        )
        axs[0].plot(
            t / 1000.0,
            cat_cross,
            linewidth=1,
            color=cat_color[ind + 1],
            label="Cat cross correlation",
        )
        axs[0].plot(
            t / 1000.0,
            ani_selfs,
            linewidth=1,
            color=ani_color[ind],
            label="Ani self correlation",
        )
        axs[0].plot(
            t / 1000.0,
            ani_cross,
            linewidth=1,
            color=ani_color[ind + 1],
            label="Ani cross correlation",
        )

        # For the fitting of slopes figure
        ### CATION
        axs[1].plot(
            t / 1000.0,
            total_cat,
            linewidth=1,
            color=cat_color[ind],
            label="Cat total correlation",
        )
        axs[1].plot(
            t / 1000.0,
            lin_func(t, *corr_slopes[0] * num_cations**2),
            linewidth=1,
            linestyle="--",
            color="black",
        )
        mask = (t >= t[::time_width][diffu_inxs[0]][0]) & (
            t <= t[::time_width][diffu_inxs[0]][-1]
        )
        axs[1].plot(
            t[mask] / 1000.0,
            total_cat[mask],
            linewidth=1,
            color="black",
            label="Cat LogLog slope = %1.3e" % (m_loglogs[0]),
        )
        ### ANION
        axs[1].plot(
            t / 1000.0,
            total_ani,
            linewidth=1,
            color=ani_color[ind],
            label="Ani total correlation",
        )
        axs[1].plot(
            t / 1000.0,
            lin_func(t, *corr_slopes[1] * num_cations**2),
            linewidth=1,
            linestyle="--",
            color="black",
        )
        mask = (t >= t[::time_width][diffu_inxs[1]][0]) & (
            t <= t[::time_width][diffu_inxs[1]][-1]
        )
        axs[1].plot(
            t[mask] / 1000.0,
            total_ani[mask],
            linewidth=1,
            color="black",
            label="Ani LogLog slope = %1.3e" % (m_loglogs[1]),
        )
        ### CATANI
        axs[1].plot(
            t / 1000.0,
            lin_func(t, *corr_slopes[2] * num_cations**2),
            linewidth=1,
            linestyle="--",
            color="black",
        )
        mask = (t >= t[::time_width][diffu_inxs[2]][0]) & (
            t <= t[::time_width][diffu_inxs[2]][-1]
        )
        axs[1].plot(
            t[mask] / 1000.0,
            catani[mask],
            linewidth=1,
            color="black",
            label="CatAni LogLog slope = %1.3e" % (m_loglogs[2]),
        )
        for ax in axs:
            ax.legend(fontsize=int(1 / 2 * labelsize))
            ax.tick_params(
                axis="y",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
            )
            ax.tick_params(
                axis="x",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
            )

            for key in ax.spines.keys():
                ax.spines[key].set_linewidth(tickwidth)

            ax.set_xlabel("$t$ / ns", fontsize=labelsize)
            ax.set_ylabel("MSD($t$) / \AA$^2$", fontsize=labelsize)
            ax.set_xlabel("$t$ / ns", fontsize=labelsize)
            ax.set_ylabel("total correlation($t$) / \AA$^2$", fontsize=labelsize)

        axs[1].text(
            0.6,
            0.08,
            "Conductivity = {:1.2e} $S$/$cm$".format(condWN),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=labelsize,
        )

        if not name:
            name = folder.split("/")[-1]
            title_name = f"{name}, T{int(temperature)}"
        else:
            title_name = f"{name}, T{int(temperature)}"
        fig.suptitle(f"{title_name}", fontsize=titelsize)
        plt.tight_layout()
        plt.savefig(f"{folder}/Correlation_fit.svg", dpi=dpi)
        plt.savefig(f"{folder}/Correlation_fit.png", dpi=dpi)
        plt.show()


def get_dcf(delta_x):
    """dcf will calculate the distance correlation matrix for each window of length delT, this means that the
    [[r_cat1 dot r_cat1, r_cat1 dot r_cat2, r_cat1 dot r_ani1, r_cat1 dot r_ani2],
     [r_cat2 dot r_cat1, r_cat2 dot r_cat2, r_cat2 dot r_ani1, r_cat2 dot r_ani2],
     [r_ani1 dot r_cat1, r_ani1 dot r_cat2, r_ani1 dot r_ani1, r_ani1 dot r_ani2],
     [r_ani2 dot r_cat1, r_ani2 dot r_cat2, r_ani2 dot r_ani1, r_ani2 dot r_ani2]] for each window... diagonal=self"""
    return np.stack(
        [np.matmul(cur_frame, np.transpose(cur_frame, (1, 0))) for cur_frame in delta_x]
    )


def get_self(dcf):
    return np.array([np.sum(np.diagonal(cur_frame)) for cur_frame in dcf])
    # return np.array([np.diagonal(cur_frame).mean() for cur_frame in dcf])


def get_total(dcf):
    return np.array([np.sum(cur_frame) for cur_frame in dcf])
    # return np.array([np.diagonal(cur_frame).mean() for cur_frame in dcf])


def get_cross(dcf):  # distinct
    return np.array(
        [(np.sum(cur_frame) - np.sum(np.diagonal(cur_frame))) for cur_frame in dcf]
    )
    # return np.array([(np.triu(cur_frame, diagonal=1)+np.tril(cur_frame, diagonal=-1)).mean() for cur_frame in dcf])


def compute_fft_onsager(complete_disp_matrix):
    """
    complete_disp_matrix (shape: (number of frames X number of particles X dimensions));
    each value is displacement over one time window (instantaneous disp)
    returns a particle-based covariance matrix as a function of time
    dimensions : (number of particles X number of particles X number of frames)
    """

    def autocorrelation_fft(x):
        N = x.shape[0]
        F = np.fft.fft(x, n=2 * N)
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real
        n = N * np.ones(N) - np.arange(N)
        return res / n

    # Function to compute self-MSD for a single ion using FFT
    def one_ion_msd_fft(r, dt_indices):
        n_step, dim = r.shape
        r_square = np.square(r)
        r_square = np.append(r_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
        S1_component = np.zeros((dim, n_step))  # (dim, n_step)
        r_square_sum = 2 * np.sum(r_square, axis=0)  # (3)
        for i in range(n_step):
            r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
            S1_component[:, i] = r_square_sum / (n_step - i)
        S1 = np.sum(S1_component, axis=0)

        # Compute S2 using FFT (autocorrelation)
        S2_component = np.array(
            [autocorrelation_fft(r[:, i]) for i in range(r.shape[1])]
        )  # (dim, N)
        S2 = np.sum(S2_component, axis=0)
        # Return the MSD and its components for each time interval
        return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[
            :, dt_indices
        ]

    def crosscorrelation_fft(x1, x2):
        N = x1.shape[0]
        F1 = np.fft.fft(x1, n=2 * N)
        F2 = np.fft.fft(x2, n=2 * N)
        PSD = F1 * F2.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real
        n = N * np.ones(N) - np.arange(N)
        return res / n

    # New function to compute cross-MSD (distinct particles) using FFT
    def cross_ion_msd_fft(r1, r2, dt_indices):
        n_step, dim = r1.shape

        # Compute the cross-term S1

        r3_square = r1 * r2
        r3_square = np.append(r3_square, np.zeros((1, dim)), axis=0)  # (n_step+1)
        S1_component = np.zeros((dim, n_step))  # (dim, n_step)
        r3_square_sum = 2 * np.sum(r3_square, axis=0)
        for i in range(n_step):
            r3_square_sum = r3_square_sum - r3_square[i - 1] - r3_square[n_step - i]
            S1_component[:, i] = r3_square_sum / (n_step - i)

        S1 = np.sum(S1_component, axis=0)
        # Cross-correlation S2 using FFT
        S2_component1 = np.array(
            [crosscorrelation_fft(r1[:, i], r2[:, i]) for i in range(r1.shape[1])]
        )  # (dim, N)
        S2_component2 = np.array(
            [crosscorrelation_fft(r2[:, i], r1[:, i]) for i in range(r1.shape[1])]
        )  # (dim, N)
        S2_1 = np.sum(S2_component1, axis=0)
        S2_2 = np.sum(S2_component2, axis=0)

        return (S1 - S2_1 - S2_2)[dt_indices], (
            S1_component - S2_component1 - S2_component2
        )[:, dt_indices]

    n_ions = complete_disp_matrix.shape[1]
    # Main loop to compute the self- and cross-MSD for all particle pairs
    n_dt = complete_disp_matrix.shape[0]
    msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
    msd_component_by_pairs = np.empty([3, 0, n_dt])  # Shape of 3 * n_pairs * n_dt
    displacements_final_diffusion_ions = complete_disp_matrix.transpose(1, 0, 2)
    dt_indices = np.arange(0, n_dt)
    msd_matrix = []
    for i in tqdm(range(n_ions)):
        msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
        # print(f"working on column {i}th")
        for j in range(n_ions):  # Loop over distinct particle pairs
            ith = displacements_final_diffusion_ions[i, :, :].transpose()
            jth = displacements_final_diffusion_ions[j, :, :].transpose()
            ith = np.cumsum(ith.transpose(), axis=0)
            jth = np.cumsum(jth.transpose(), axis=0)

            # print(mask_tensor)
            if i > j:
                msd_ij = np.zeros(n_dt)
                msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                continue
            if i == j:
                # Self-correlation (diagonal terms)
                msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
            else:
                # Cross-correlation (off-diagonal terms)

                msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)

            msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
        msd_matrix.append(msd_by_pairs)  # , axis=0)

    msd_matrix = np.stack(msd_matrix)

    # Average MSD and MSD components over all pairs
    msd = np.average(msd_by_pairs, axis=0)
    msd_component = np.average(msd_component_by_pairs, axis=1)
    msd_matrix = np.where(msd_matrix, msd_matrix, msd_matrix.transpose(1, 0, 2))

    return msd_matrix
