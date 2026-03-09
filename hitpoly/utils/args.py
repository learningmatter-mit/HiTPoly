import os
from tap import Tap
from typing_extensions import Literal
from typing import List

RESULTS_PATH = "/home/jurgis/HiTPoly/results"


class hitpolyArgs(Tap):
    geom_path: str = None  # Path for geometries

    extra_results_path: str = None  # Meant for saving single molecule trainings under a smiles folder or when doing hyperparam optimization

    results_path: str = RESULTS_PATH  # Path for where the results have to be saved in

    ff_type: str = "opls"

    device: str = "cpu"
    """
    Options 'cpu' or 'cuda:0', maybe something else, depends on the machine.
    """

    exclude_dihedrals_in_pairs: bool = False
    """
    If True then 1-4 neighbors (end atoms of dihedrals) are removed from the pairwise interactions (together with bonds and angles)
    """

    discrete_flag: bool = False
    """
    True - training using discrete atom types to directly optimize parameters (adapted from AuTopology)
    False - training using atomic embeddings to predict parameters (learn weights of parameter prediction)
    """

    discrete_neighborhood_depth: int = 3
    """
    The number of nearest neighbor convolutions used to determine discrete atomic environments based on
    message passing
    """

    init_params: bool = False
