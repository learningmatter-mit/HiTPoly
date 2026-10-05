"""
DFT (CHELPG) charges from the htvs database, for single-ion-conductor components.

Only needed when SETTINGS['charges'] == 'DFT' or for SIC blends; Django is imported lazily
so the rest of the pipeline runs without the database.
"""
import copy
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem

from hitpoly.data.builder import TopologyBuilder, TopologyDataset
from hitpoly.utils.args import hitpolyArgs

DEFAULT_CALC_METHOD = "dft_wb97xd3_def2tzvp"
SPECIES_GROUP = "singleion_trios"


def _setup_django(htvs_path):
    import django

    for p in (htvs_path, f"{htvs_path}/djangochem/"):
        if p and p not in sys.path:
            sys.path.append(p)
    os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()


def get_avg_atomic_charges_df(smiles, htvs_path=None, htvs_details=None, expected_total_charge=-3.0, tolerance=1e-4):
    """
    Mean and std of CHELPG charges per atom over every database geometry of `smiles`.
    Geometries whose charges have the wrong length, or do not sum to expected_total_charge
    (-3 for an SIC trimer with one anion per unit), are skipped.
    Returns (DataFrame[atomic_number, mean_charge, std_charge], n_geometries).
    """
    _setup_django(htvs_path)
    from pgmols.models import Calc, Geom

    htvs_details = dict(htvs_details or {})
    htvs_details.setdefault("calc_method_name", DEFAULT_CALC_METHOD)

    geoms = Geom.objects.filter(
        species__group__name=SPECIES_GROUP,
        species__smiles=smiles,
        parentjob__config__name=htvs_details["geom_config_name"],
        childjobs__status="done",
    ).all()
    if not geoms.exists():
        raise ValueError(f"No geometries found for the given SMILES: {smiles}")

    atomic_numbers = [int(row[0]) for row in geoms[0].xyz]
    num_atoms = len(atomic_numbers)

    charges_qs = Calc.objects.filter(
        method__name=htvs_details["calc_method_name"],
        parentjob__group__name=SPECIES_GROUP,
        geoms__id__in=list(geoms.values_list("id", flat=True)),
        props__atomiccharges__has_key="chelpg",
    ).values_list("props__atomiccharges__chelpg", flat=True)
    if not charges_qs.exists():
        raise ValueError(f"No CHELPG charges found for these geometries with SMILES: {smiles}")

    valid = []
    for i, charge_list in enumerate(charges_qs):
        if len(charge_list) != num_atoms:
            print(f"  - Geom {i+1} skipped: {len(charge_list)} charges, expected {num_atoms}.")
            continue
        arr = np.array(charge_list)
        if not np.isclose(arr.sum(), expected_total_charge, atol=tolerance):
            print(f"  - Geom {i+1} skipped: total charge {arr.sum():.6f}, expected {expected_total_charge}.")
            continue
        valid.append(arr)
    if not valid:
        raise ValueError(f"Validation failed for all geometries for SMILES: {smiles}")

    charge_matrix = np.array(valid)
    print(f"Averaged CHELPG charges over {len(valid)} valid geometries.")
    df = pd.DataFrame({
        "atomic_number": atomic_numbers,
        "mean_charge": charge_matrix.mean(axis=0),
        "std_charge": charge_matrix.std(axis=0),
    })
    return df, len(valid)


def patch_params_with_dft_charges_from_db(
    smiles=None,
    original_param_dict=None,
    htvs_path=None,
    htvs_details=None,
    smiles_dft=None,
    long_smiles_list=None,
    neighborhood_depth=2,
):
    """
    Return a copy of original_param_dict with atom charges replaced by DFT CHELPG charges
    of the DFT oligomer `smiles_dft`, averaged per atom type. The oligomer is typed jointly
    with `long_smiles_list` so the type IDs match the long chains. Returns None on failure.
    (`smiles` is the legacy name for `smiles_dft`.)
    """
    smiles_dft = smiles_dft or smiles
    if smiles_dft is None:
        raise ValueError("Provide `smiles_dft` (or legacy `smiles=`) for DFT charge retrieval.")

    def _canon(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"Bad SMILES: {s}")
        return Chem.MolToSmiles(m, canonical=True)

    def _elem(z):
        return Chem.GetPeriodicTable().GetElementSymbol(int(z))

    can_dft = _canon(smiles_dft)
    can_longs = [_canon(s) for s in (long_smiles_list or [])]
    print(f"--- DFT charges for {can_dft} (joint typing with {len(can_longs)} long SMILES) ---")

    # 1) Joint typing
    train_args = hitpolyArgs()
    train_args.discrete_flag = True
    train_args.discrete_neighborhood_depth = neighborhood_depth
    builders = [TopologyBuilder(smiles=[s], train_args=train_args, load_geoms=False) for s in [can_dft] + can_longs]
    dataset = TopologyDataset(data=builders, train_args=train_args)
    Z_dft = dataset.data[0].atomic_nums
    types_dft = [int(t) for t in dataset.data[0].atom_type.tolist()]
    long_types = {int(t) for d in dataset.data[1:] for t in d.atom_type.tolist()}

    # 2) DFT charges
    try:
        db_df, n_geoms = get_avg_atomic_charges_df(can_dft, htvs_path=htvs_path, htvs_details=htvs_details)
    except Exception as e:
        print(f"[FATAL] DB fetch failed: {e}")
        return None
    if db_df["atomic_number"].astype(int).tolist() != list(Z_dft):
        print("[ERROR] Atom ordering mismatch between hitpoly typing (DFT oligomer) and DB (CHELPG). Aborting.")
        return None

    # 3) Per-type simple mean over all atoms
    buckets = defaultdict(list)
    for t, q in zip(types_dft, db_df["mean_charge"].astype(float)):
        buckets[t].append(q)
    type_charges = {t: float(np.mean(qs)) for t, qs in buckets.items()}

    # 4) Patch
    new_param_dict = copy.deepcopy(original_param_dict)
    if not isinstance(new_param_dict.get("atoms"), dict):
        print("[ERROR] 'original_param_dict' missing 'atoms' dict.")
        return None
    rows = []
    for k, props in new_param_dict["atoms"].items():
        try:
            t = int(k)
        except (TypeError, ValueError):
            continue
        prev = props.get("charges", [None])[0] if props.get("charges") else None
        if t in type_charges:
            props["charges"] = [type_charges[t]]
        rows.append((t, prev, props.get("charges", [None])[0], t in type_charges))

    print(f"Patched {sum(r[3] for r in rows)} atom types from {n_geoms} geometries:")
    print(pd.DataFrame(rows, columns=["TypeInt", "PrevCharge", "AssignedCharge", "Updated"])
          .sort_values("TypeInt").to_string(index=False))
    missing = sorted(long_types - set(types_dft))
    if missing:
        print(f"Types in long chains but not in the DFT oligomer (keep LigParGen charges): {missing}")
    return new_param_dict
