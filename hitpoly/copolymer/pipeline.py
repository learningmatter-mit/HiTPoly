"""
Pipeline driver: describe a system as a dict, then build its box and run MD.

    from hitpoly.copolymer.pipeline import build_system, run_simulation
    system = dict(kind="copolymer", name="P5Alt_5050_Li", monomers=[POLY6, PEO],
                  architecture="alternating")
    build = build_system(system, settings={"cation": "Li"})
    run_simulation(build)

kinds:  homopolymer  monomers=[A]
        copolymer    monomers=[A, B, ...], architecture="random"|"block"|"alternating",
                     fractions=[fA, fB, ...] (not for alternating), num_blocks=4
        blend        monomers=[A, B, ...], chains_per_component=[...], repeats_per_component=[...],
                     sic_components=[i, ...] for single-ion conductors
        star         monomers=[A], arms=3

run_system() is the resumable entry point used by run_copolymer.py: it reuses a complete
box with the same name and skips the MD if it already finished.
"""
import glob
import json
import os
import random
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from hitpoly.copolymer.box import create_box_and_ff_files_openmm, salt_setup
from hitpoly.copolymer.chains import create_conformer_pdb, create_long_smiles, create_oligomers, get_atom_count
from hitpoly.copolymer.concentration import (
    get_concentration_from_charge_neutrality_blend,
    get_concentration_from_molality,
)
from hitpoly.copolymer.ligpargen import NUMBER_TO_SYMBOL, create_combined_param_dict

HITPOLY_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Physics and protocol of the published copolymer dataset; change deliberately.
# 393 K rather than the experimental 353 K compensates for OPLS-AA underestimating chain dynamics.
DEFAULT_SETTINGS = dict(
    cation="Li",                 # "Li" or "Na"
    anion="N",                   # element tag of the tethered anion site, used for SIC counting
    molality=1.0,                # mol salt per kg polymer (non-SIC systems)
    polymer_count=30,            # chains per box (homopolymer / copolymer / star)
    polymer_chain_length=1000,   # target atoms per chain
    use_total_repeats=False,     # True -> use total_repeats instead of polymer_chain_length
    total_repeats=None,
    charge_scale=0.75,           # salt charge scaling
    charges="LPG",               # "LPG" = LigParGen | "DFT" = database charges (SIC only)
    neighborhood_depth=2,        # atom-type neighborhood depth
    add_end_Cs=True,
    box_multiplier=20,
    random_seed=None,            # int -> reproducible random sequences
    simu_type="conductivity",    # "conductivity" or "tg"
    simu_temp=393,               # K
    simu_length=100,             # ns
    xyz_output=25,               # ps between frames read by the analysis script
    platform="supercloud",       # "supercloud", "local" or "perlmutter"
    hitpoly_env="htvs",          # conda env the analysis script activates
)

HTVS_DETAILS = {
    "geom_config_name": "nvt_conf_generation_ligpargen_lammps",
    "calc_method_name": "dft_wb97xd3_def2tzvp",
    "species_group_name": "singleion_trios",
}

SYSTEM_DEFAULTS = dict(
    name=None, architecture=None, fractions=None, num_blocks=4, arms=3,
    chains_per_component=None, repeats_per_component=None,
    sic_components=None, oligomer_repeats=None,
)
MAX_RETRIES = 10
TFSI_ATOMS = 15


def default_paths():
    home = os.path.expanduser("~")
    return dict(
        results_path=os.path.join(home, "results"),
        hitpoly_path=HITPOLY_PATH,
        htvs_path=os.path.join(home, "htvs"),
        packmol_path=os.environ.get("PACKMOL", os.path.join(home, "packmol-20.14.4", "packmol")),
    )


def resolve_system(system):
    """Validate a system dict and fill in the derived fields the pipeline needs."""
    s = {**SYSTEM_DEFAULTS, **system}
    kind, mons = s.get("kind"), list(s.get("monomers") or [])
    if not mons:
        raise ValueError("'monomers' cannot be empty.")
    if not s["name"]:
        raise ValueError("Give the system a 'name' (used for the output folder).")
    s["monomers"] = mons
    n = len(mons)

    if kind == "homopolymer":
        if n != 1:
            raise ValueError("A homopolymer takes exactly one monomer; use kind='blend' for several.")
        s.update(mode="homopolymer", fractions=[1.0], blend=False)
    elif kind == "star":
        if n != 1:
            raise ValueError("A star polymer takes exactly one monomer.")
        if not s["arms"] or s["arms"] <= 0:
            raise ValueError("A star polymer needs arms > 0.")
        s.update(mode="star", fractions=[1.0], blend=False)
    elif kind == "copolymer":
        arch = s["architecture"]
        if arch not in ("random", "block", "alternating"):
            raise ValueError("Copolymer 'architecture' must be 'random', 'block', or 'alternating'.")
        if n < 2:
            raise ValueError("A copolymer needs at least two monomers.")
        fr = list(s["fractions"] or [1.0 / n] * n)
        if len(fr) != n or abs(sum(fr) - 1.0) > 1e-6:
            raise ValueError(f"'fractions' must have one entry per monomer and sum to 1 (got {fr}).")
        if arch == "alternating" and any(abs(f - 1.0 / n) > 1e-6 for f in fr):
            raise ValueError("Alternating copolymers are equimolar; omit 'fractions' or make them equal.")
        if arch == "block" and s["num_blocks"] % n:
            raise ValueError(f"'num_blocks' ({s['num_blocks']}) must be a multiple of the number of monomers ({n}).")
        s.update(mode=arch, fractions=fr, blend=False)
    elif kind == "blend":
        for key in ("chains_per_component", "repeats_per_component"):
            if not s[key] or len(s[key]) != n:
                raise ValueError(f"A blend needs '{key}' with one entry per monomer.")
        s.update(mode="homopolymer", fractions=[1.0] * n, blend=True)
    else:
        raise ValueError("'kind' must be 'homopolymer', 'copolymer', 'blend', or 'star'.")

    s["sic"] = bool(s["sic_components"])
    if s["sic"] and kind != "blend":
        raise ValueError("Single-ion conductors run as kind='blend' (a single SIC component is a 1-entry blend).")
    if s["oligomer_repeats"] is None:
        s["oligomer_repeats"] = [3] * n if kind == "blend" else [3]
    return s


def compute_concentration(s, st, work_dir="."):
    return get_concentration_from_molality(
        save_path=work_dir,
        molality=st["molality"],
        polymer_count=st["polymer_count"],
        monomers=s["monomers"],
        fractions=s["fractions"],
        add_end_Cs=st["add_end_Cs"],
        polymerization_mode=s["mode"],
        num_blocks=s["num_blocks"],
        cation=st["cation"],
        anion=st["anion"],
        polymer_chain_length=st["polymer_chain_length"],
        total_repeats=st["total_repeats"],
        use_total_repeats=st["use_total_repeats"],
        arms=s["arms"],
        blend_mode=s["blend"],
        num_chains_per_component=s["chains_per_component"],
        num_repeats_per_component=s["repeats_per_component"],
        single_ion_conductor=s["sic"],
        sic_component_indices=s["sic_components"],
    )


def _make_chain(save_path, smiles, fractions, repeats, mode, s, st, pdb_name, build_pdb, existing=()):
    """Build one chain SMILES (and its 3D PDB unless build_pdb=False), retrying on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            long_smiles, counts = create_long_smiles(
                save_path=save_path, smiles=smiles, fractions=fractions,
                total_repeats=repeats, add_end_Cs=st["add_end_Cs"],
                polymerization_mode=mode, num_blocks=s["num_blocks"], arms=s["arms"],
                write_log=build_pdb,
            )
            if long_smiles in existing:
                continue
            if not build_pdb:
                return long_smiles, None, counts
            filename, mol, _ = create_conformer_pdb(path_to_save=save_path, smiles=long_smiles, name=pdb_name)
            if filename is not None and mol is not None:
                return long_smiles, filename, counts
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
    raise RuntimeError(f"Could not build chain '{pdb_name}' after {MAX_RETRIES} attempts.")


def generate_chains(s, st, save_path, repeats, build_pdb=True):
    """
    Returns (long_smiles_list, filenames, chains_per_file, monomer_counts_list).
    Blends get one file per component; random copolymers get polymer_count distinct
    sequences (one chain each); everything else is one chain packed polymer_count times.
    """
    long_smiles_list, filenames, chains_per_file, counts_list = [], [], [], []

    def add(result, n_chains):
        long_smiles, filename, counts = result
        long_smiles_list.append(long_smiles)
        filenames.append(filename)
        chains_per_file.append(n_chains)
        counts_list.append(counts)

    if s["blend"]:
        for i, (mon, n_rep, n_ch) in enumerate(zip(s["monomers"], s["repeats_per_component"], s["chains_per_component"])):
            print(f"  Component {i}: {n_ch} chains x {n_rep} repeats ({mon[:40]})")
            add(_make_chain(save_path, [mon], [1.0], n_rep, "homopolymer", s, st,
                            f"polymer_conformation_{i}", build_pdb), n_ch)
    elif s["mode"] == "random":
        for i in range(st["polymer_count"]):
            print(f"  Random chain {i + 1}/{st['polymer_count']}")
            add(_make_chain(save_path, s["monomers"], s["fractions"], repeats, "random", s, st,
                            f"polymer_conformation_{i}", build_pdb, existing=long_smiles_list), 1)
    else:
        add(_make_chain(save_path, s["monomers"], s["fractions"], repeats, s["mode"], s, st,
                        "polymer_conformation", build_pdb), st["polymer_count"])
    return long_smiles_list, filenames, chains_per_file, counts_list


def summarize_chains(long_smiles_list, chains_per_file, counts_list, concentration):
    print(f"\n{len(long_smiles_list)} unique chain(s), {sum(chains_per_file)} chains in the box, "
          f"ions [cation, free anion] = {concentration}")
    for smi, n, counts in zip(long_smiles_list, chains_per_file, counts_list):
        total = sum(counts.values())
        comp = ", ".join(f"{k[:25]}: {v} ({100 * v / total:.0f}%)" for k, v in counts.items())
        print(f"  x{n:<3d} {get_atom_count(smi)} atoms | {comp}")


def _merge(settings, paths):
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    pt = {**default_paths(), **(paths or {})}
    return st, pt


def build_system(system, settings=None, paths=None, dry_run=False):
    """
    Build everything up to (not including) the MD run. Returns a dict describing the build.
    dry_run=True stops after the chain SMILES and writes only to a temp folder.
    """
    s = resolve_system(system)
    st, paths = _merge(settings, paths)
    if st["random_seed"] is not None:
        random.seed(st["random_seed"])
        np.random.seed(st["random_seed"])

    print(f"=== {s['name']}: {s['kind']} ({s['mode']}), {st['cation']}+ ===")
    concentration, repeats = compute_concentration(s, st, tempfile.mkdtemp())

    if dry_run:
        chains = generate_chains(s, st, tempfile.mkdtemp(), repeats, build_pdb=False)
        summarize_chains(chains[0], chains[2], chains[3], concentration)
        return dict(system=s, settings=st, concentration=concentration, repeats=repeats,
                    long_smiles=chains[0], chains_per_file=chains[2])

    n_chains = sum(s["chains_per_component"]) if s["blend"] else st["polymer_count"]
    folder_name = (
        f"{s['name']}_PC{n_chains}_C{concentration[0]}_q0{int(st['charge_scale'] * 100)}"
        f"_q{st['charges']}_{time.strftime('%y%m%d_H%HM%M')}"
    )
    save_path = os.path.join(paths["results_path"], folder_name)
    ligpargen_path = os.path.join(save_path, "ligpargen")
    os.makedirs(ligpargen_path, exist_ok=True)
    print(f"Save path: {save_path}")

    print("\nCreating oligomers")
    _, oligomers_list = create_oligomers(
        smiles=s["monomers"], polymerization_mode="homopolymer" if s["blend"] else s["mode"],
        arms=s["arms"], oligomer_repeats_list=s["oligomer_repeats"],
    )

    print("\nCreating chains")
    long_smiles_list, filenames, chains_per_file, counts_list = generate_chains(s, st, save_path, repeats)
    summarize_chains(long_smiles_list, chains_per_file, counts_list, concentration)
    with open(os.path.join(save_path, "final_long_smiles.txt"), "w") as f:
        f.write(str(long_smiles_list))

    print("\nRunning LigParGen and building combined parameter dict...")
    param_dict, atoms_short, atom_names_short, train_dataset = create_combined_param_dict(
        oligomers_list, long_smiles_list, ligpargen_path, st["platform"],
        neighborhood_depth=st["neighborhood_depth"],
    )

    smiles_to_index = {mol.smiles[0]: idx for idx, mol in enumerate(train_dataset.data)}
    all_atom_names_long, all_atoms_long = [], []
    for smi in long_smiles_list:
        mol_data = train_dataset.data[smiles_to_index[smi]]
        all_atom_names_long.append(mol_data.atom_type.tolist())
        all_atoms_long.append(mol_data.atomic_nums)

    if s["sic"]:
        from hitpoly.copolymer.dft_charges import patch_params_with_dft_charges_from_db

        for comp_idx in s["sic_components"]:
            print(f"Patching DFT charges for SIC component {comp_idx}...")
            patched = patch_params_with_dft_charges_from_db(
                original_param_dict=param_dict,
                htvs_path=paths["htvs_path"],
                htvs_details=HTVS_DETAILS,
                smiles_dft=oligomers_list[comp_idx],
                long_smiles_list=long_smiles_list,
                neighborhood_depth=st["neighborhood_depth"],
            )
            if patched is None:
                raise RuntimeError(f"DFT charge patching failed for SIC component {comp_idx}.")
            param_dict = patched
        concentration, sic_scale, _ = get_concentration_from_charge_neutrality_blend(
            atom_names_long_list=all_atom_names_long,
            param_dict=param_dict,
            polymer_count_list=list(s["chains_per_component"]),
            cation_charge=st["charge_scale"],
            single_ion_conductor=True,
            sic_component_indices=s["sic_components"],
            anions_per_polymer_list=[smi.count(f"[{st['anion']}-]") for smi in long_smiles_list],
        )
        print(f"[SIC scaling] concentration={concentration}, scale={sic_scale:.6f}")

    print("\nBuilding simulation box and writing force-field files...")
    create_box_and_ff_files_openmm(
        save_path=save_path,
        long_smiles=long_smiles_list,
        short_smiles=oligomers_list,
        filename=filenames,
        polymer_count=chains_per_file,
        concentration=concentration,
        packmol_path=paths["packmol_path"],
        atoms_short=atoms_short,
        atoms_long=[[NUMBER_TO_SYMBOL[z] for z in chain] for chain in all_atoms_long],
        atom_names_short=atom_names_short,
        atom_names_long=all_atom_names_long,
        param_dict=param_dict,
        charge_scale=st["charge_scale"],
        box_multiplier=st["box_multiplier"],
        salt=True,
        single_ion_conductor=s["sic"],
        sic_component_indices=s["sic_components"],
        neighborhood_depth=st["neighborhood_depth"],
        **salt_setup(st["cation"], paths["hitpoly_path"]),
    )

    build = dict(system=s, settings=st, paths=paths, save_path=save_path,
                 concentration=concentration, repeats=repeats,
                 long_smiles=long_smiles_list, chains_per_file=chains_per_file)
    with open(os.path.join(save_path, "build_info.json"), "w") as f:
        json.dump({k: build[k] for k in ("system", "settings", "concentration", "repeats",
                                         "long_smiles", "chains_per_file")}, f, indent=1)
    problem = check_box(save_path)
    if problem:
        raise RuntimeError(f"Box failed the completeness check: {problem}")
    print("Box building complete.")
    return build


# ── Resuming ───────────────────────────────────────────────────────────────────

def _box_contents(save_path):
    """(long_smiles, chains_per_file, concentration, repeats) from build_info.json, or for
    boxes built before it existed, from final_long_smiles.txt and the folder name."""
    info_path = os.path.join(save_path, "build_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        return info["long_smiles"], info["chains_per_file"], info["concentration"], info.get("repeats")
    import ast
    import re

    with open(os.path.join(save_path, "final_long_smiles.txt")) as f:
        long_smiles = ast.literal_eval(f.read())
    name = os.path.basename(save_path.rstrip("/"))
    n_chains = int(re.search(r"_PC(\d+)_", name).group(1))
    n_ions = int(re.search(r"_C(\d+)_", name).group(1))
    chains = [n_chains] if len(long_smiles) == 1 else [1] * len(long_smiles)
    return long_smiles, chains, [n_ions, n_ions], None


def check_box(save_path):
    """None if the box folder is complete, otherwise a description of what is wrong.
    Complete = packed_box.pdb, force_field.xml and force_field_resids.xml exist and the PDB
    has (chain atoms x chains) + cations + 15 x TFSI atoms."""
    for f in ("packed_box.pdb", "force_field.xml", "force_field_resids.xml", "final_long_smiles.txt"):
        if not os.path.exists(os.path.join(save_path, f)):
            return f"missing {f}"
    long_smiles, chains, conc, _ = _box_contents(save_path)
    expected = sum(get_atom_count(smi) * n for smi, n in zip(long_smiles, chains)) + conc[0] + conc[1] * TFSI_ATOMS
    with open(os.path.join(save_path, "packed_box.pdb")) as f:
        found = sum(1 for line in f if line.startswith(("ATOM", "HETATM")))
    if found != expected:
        return f"packed_box.pdb has {found} atoms, expected {expected}"
    return None


def find_box(results_path, name):
    """Newest complete box folder for `name` in results_path, or None."""
    for folder in sorted(glob.glob(os.path.join(results_path, f"{name}_PC*")), key=os.path.getmtime, reverse=True):
        if os.path.isdir(folder) and check_box(folder) is None:
            return folder
    return None


def simulation_finished(save_path):
    # prod_run_nvt writes final_state_box.pdb after its last quarter.
    return os.path.exists(os.path.join(save_path, "final_state_box.pdb"))


def run_simulation(build):
    st, save_path, paths = build["settings"], build["save_path"], build["paths"]
    from hitpoly.simulations.openmm_scripts import (
        equilibrate_system_1,
        equilibrate_system_2,
        prod_run_nvt,
        prod_run_tg,
        write_analysis_script,
    )

    final_save_path = os.path.join(save_path, "openmm_saver")
    os.makedirs(final_save_path, exist_ok=True)
    if st["simu_type"] == "conductivity":
        equilibrate_system_1(save_path=save_path, final_save_path=final_save_path)
        equilibrate_system_2(save_path=save_path, final_save_path=final_save_path)
        prod_run_nvt(save_path=save_path, final_save_path=final_save_path,
                     simu_temp=st["simu_temp"], simu_time=st["simu_length"])
        repeats = build.get("repeats")
        write_analysis_script(
            save_path=save_path, results_path=save_path,
            repeat_units=repeats if isinstance(repeats, list) else [repeats],
            cation=st["cation"], anion=st["anion"], platform=st["platform"],
            simu_temperature=st["simu_temp"], prod_run_time=st["simu_length"],
            hitpoly_path=paths["hitpoly_path"], xyz_output=st["xyz_output"],
            hitpoly_env=st["hitpoly_env"],
        )
    elif st["simu_type"] == "tg":
        prod_run_tg(save_path=save_path, final_save_path=final_save_path, simu_time=st["simu_length"])
    else:
        raise ValueError("simu_type must be 'conductivity' or 'tg'.")


def run_system(system, settings=None, paths=None, simulate=True, status_path=None):
    """
    Build (or reuse) the box for `system`, then run MD unless it already finished.
    Writes one status row (name, save_path, box_built, sim_finished, error) to status_path.
    """
    st, pt = _merge(settings, paths)
    rec = dict(name=system["name"], save_path="", box_built=False, sim_finished=False, error="")

    def write_status():
        if status_path:
            os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
            pd.DataFrame([rec]).to_csv(status_path, index=False)

    try:
        existing = find_box(pt["results_path"], system["name"])
        if existing:
            print(f"Reusing complete box: {existing}")
            long_smiles, chains, conc, repeats = _box_contents(existing)
            build = dict(system=resolve_system(system), settings=st, paths=pt, save_path=existing,
                         concentration=conc, repeats=repeats, long_smiles=long_smiles, chains_per_file=chains)
        else:
            build = build_system(system, st, pt)
        rec.update(save_path=build["save_path"], box_built=True)
        write_status()

        if simulate:
            if simulation_finished(build["save_path"]):
                print("Simulation already finished; skipping MD.")
            else:
                run_simulation(build)
            rec["sim_finished"] = simulation_finished(build["save_path"])
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    write_status()
    return rec


# ── Systems from a CSV ─────────────────────────────────────────────────────────
_ARCH = {"block": "block", "random": "random", "rand": "random",
         "alternating": "alternating", "alt": "alternating"}
_ARCH_TAG = {"block": "Block_4", "random": "Rand", "alternating": "Alt"}


def systems_from_csv(csv_path):
    """
    List of (system, settings_overrides), one per CSV row. Two layouts are accepted:

    1. Case-study lists (e.g. case_study_reruns_P5_P14.csv): columns pair_id, cation,
       structure, fraction_A_B, monomer_A, monomer_B and optionally run_name. Names follow
       the old convention, e.g. P14Alt_5050_Li, where 5050 is the fraction of monomer_A.
    2. General: columns name, kind, monomers ("A;B"), and optionally architecture,
       fractions ("0.8;0.2"), num_blocks, arms, plus any DEFAULT_SETTINGS key as a column.
    """
    df = pd.read_csv(csv_path)
    out = []
    if {"monomer_A", "monomer_B", "structure", "fraction_A_B"} <= set(df.columns):
        for _, r in df.iterrows():
            arch = _ARCH[str(r.structure).lower()]
            fa, fb = (int(x) / 100 for x in str(r.fraction_A_B).split("/"))
            name = f"{r.pair_id}{_ARCH_TAG[arch]}_{str(r.fraction_A_B).replace('/', '')}_{r.cation}"
            if "run_name" in df.columns and r.run_name != name:
                raise ValueError(f"run_name {r.run_name} does not match the naming convention ({name}).")
            system = dict(kind="copolymer", name=name, monomers=[r.monomer_A, r.monomer_B],
                          architecture=arch, fractions=None if arch == "alternating" else [fa, fb])
            out.append((system, {"cation": r.cation}))
        return out

    split = lambda v: [x.strip() for x in str(v).split(";") if x.strip()]
    for _, r in df.iterrows():
        system = dict(kind=r["kind"], name=r["name"], monomers=split(r["monomers"]))
        for key in ("architecture", "num_blocks", "arms"):
            if key in df.columns and pd.notna(r[key]):
                system[key] = r[key] if key == "architecture" else int(r[key])
        if "fractions" in df.columns and pd.notna(r["fractions"]):
            system["fractions"] = [float(x) for x in split(r["fractions"])]
        overrides = {k: _coerce_setting(k, r[k]) for k in DEFAULT_SETTINGS if k in df.columns and pd.notna(r[k])}
        out.append((system, overrides))
    return out


def _coerce_setting(key, value):
    """CSV cells arrive as float/str; match the type of the default in DEFAULT_SETTINGS."""
    default = DEFAULT_SETTINGS[key]
    if isinstance(default, bool):
        return str(value).strip().lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None and isinstance(value, float) and value.is_integer():
        return int(value)
    return value
