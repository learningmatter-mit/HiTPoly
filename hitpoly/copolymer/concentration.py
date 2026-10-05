"""
Chain sizing and salt counts.

get_concentration_from_molality         ion pairs from molality (all non-SIC systems)
get_concentration_from_charge_neutrality_blend  cation count and SIC charge scaling for SIC blends
"""
from hitpoly.copolymer.chains import create_long_smiles, get_atom_count, get_mol_mass


def get_concentration_from_molality(
    save_path,
    molality,
    polymer_count,
    monomers,
    fractions,
    add_end_Cs,
    polymerization_mode,
    num_blocks,
    cation,
    anion,
    polymer_chain_length=1100,
    total_repeats=None,
    use_total_repeats=False,
    arms=None,
    blend_mode=False,
    num_chains_per_component=None,
    num_repeats_per_component=None,
    single_ion_conductor=False,
    sic_component_indices=None,
    max_length_iterations=200,
):
    """
    Returns (concentration, repeat_units).

    Blend + SIC:      concentration = [n_cations, 0]  (charge neutrality; no free TFSI)
    Blend + standard: concentration = [n_ions, n_ions] from molality and total polymer mass
    Single component (homopolymer / random / block / alternating / star):
        repeat_units is chosen so each chain has polymer_chain_length atoms (+-10%),
        unless use_total_repeats=True, and
        concentration = [n_ions, n_ions] = molality x chain mass x polymer_count.
        For stars, repeat_units is per arm.
    """
    print(f"[get_concentration] monomers: {monomers}")

    if blend_mode:
        if not all(isinstance(arg, list) for arg in [monomers, num_chains_per_component, num_repeats_per_component]):
            raise ValueError(
                "In blend_mode, 'monomers', 'num_chains_per_component', and "
                "'num_repeats_per_component' must all be lists."
            )
        if not (len(monomers) == len(num_chains_per_component) == len(num_repeats_per_component)):
            raise ValueError("Blend mode input lists must all have the same length.")

        def homopolymer(s, repeats):
            long_smiles, _ = create_long_smiles(
                save_path=save_path, smiles=[s], fractions=[1.0],
                total_repeats=repeats, add_end_Cs=add_end_Cs,
                polymerization_mode="homopolymer", write_log=False,
            )
            return long_smiles

        if single_ion_conductor:
            if not isinstance(sic_component_indices, list):
                raise ValueError("'sic_component_indices' must be a list of int indices.")
            print("SIC blend: calculating concentration from charge neutrality.")
            total_anions = 0
            for i, (s, repeats, num_chains) in enumerate(
                zip(monomers, num_repeats_per_component, num_chains_per_component)
            ):
                if i in sic_component_indices:
                    anions_per_chain = homopolymer(s, repeats).count(f"[{anion}-]")
                    total_anions += anions_per_chain * num_chains
                    print(f"  Component {i}: {anions_per_chain} anions/chain x {num_chains} chains")
            print(f"  Total anions (= cations needed): {total_anions}")
            return [total_anions, 0], num_repeats_per_component

        print("Standard blend: calculating concentration from molality.")
        total_polymer_mass = 0
        for s, repeats, num_chains in zip(monomers, num_repeats_per_component, num_chains_per_component):
            chain_mass = get_mol_mass(homopolymer(s, repeats))
            total_polymer_mass += chain_mass * num_chains
            print(f"  Component mass contribution: {chain_mass:.2f} g/mol x {num_chains} chains")
        n_ions = round(molality * total_polymer_mass / 1000.0)
        print(f"  Total polymer mass: {total_polymer_mass:.2f} g/mol -> {n_ions} ion pairs")
        return [n_ions, n_ions], num_repeats_per_component

    # Single-component path (homopolymer, copolymer, star)
    def build_chain(repeat_units):
        smiles, _ = create_long_smiles(
            save_path=save_path, smiles=monomers, fractions=fractions,
            total_repeats=repeat_units, add_end_Cs=add_end_Cs,
            polymerization_mode=polymerization_mode, num_blocks=num_blocks,
            arms=arms, write_log=False,
        )
        return smiles

    # Alternating chains need a whole number of A-B(-C...) cycles.
    step = len(monomers) if polymerization_mode == "alternating" else 1
    if polymerization_mode == "star" and (not arms or arms <= 0):
        raise ValueError("Star polymerization requires arms > 0.")

    if use_total_repeats:
        if not total_repeats or total_repeats <= 0:
            raise ValueError("use_total_repeats=True requires a positive 'total_repeats'.")
        repeat_units = total_repeats
    else:
        atoms_per_repeat = sum(get_atom_count(s) * f for s, f in zip(monomers, fractions))
        n_arms = arms if polymerization_mode == "star" else 1
        repeat_units = max(step, round(polymer_chain_length / (atoms_per_repeat * n_arms)))
        lo, hi = 0.9 * polymer_chain_length, 1.1 * polymer_chain_length
        for _ in range(max_length_iterations):
            if repeat_units % step:
                repeat_units += 1
            atom_count = get_atom_count(build_chain(repeat_units))
            if atom_count > hi and repeat_units > step:
                repeat_units -= step
            elif atom_count < lo:
                repeat_units += step
            else:
                break
        else:
            raise RuntimeError(
                f"Could not reach {polymer_chain_length} +-10% atoms per chain; "
                f"last try: {repeat_units} repeats -> {atom_count} atoms."
            )

    chain = build_chain(repeat_units)
    n_ions = round(molality * get_mol_mass(chain) * polymer_count / 1000)
    print(f"  {repeat_units} repeats/chain -> {get_atom_count(chain)} atoms, "
          f"{get_mol_mass(chain):.0f} g/mol x {polymer_count} chains -> {n_ions} ion pairs")
    return [n_ions, n_ions], repeat_units


def get_concentration_from_charge_neutrality_blend(
    atom_names_long_list,
    param_dict,
    polymer_count_list,
    cation_charge=0.75,
    single_ion_conductor=True,
    sic_component_indices=None,
    anions_per_polymer_list=None,
    verbose=True,
):
    """
    Scale the SIC components' charges so the box is neutral with one cation per tethered
    anion. Modifies param_dict['atoms'] in place.
    Returns (concentration, scale_factor, sic_atom_types).
    """
    if isinstance(polymer_count_list, int):
        polymer_count_list = [polymer_count_list]
    if not isinstance(atom_names_long_list, (list, tuple)) or not atom_names_long_list:
        raise ValueError("atom_names_long_list must be a non-empty list of components.")
    n_components = len(atom_names_long_list)

    if isinstance(sic_component_indices, int):
        sic_component_indices = [sic_component_indices]
    sic_component_indices = sic_component_indices or []
    anions_per_polymer_list = anions_per_polymer_list or [0] * n_components
    if len(polymer_count_list) != n_components or len(anions_per_polymer_list) != n_components:
        raise ValueError("polymer_count_list and anions_per_polymer_list must align with atom_names_long_list.")

    if "atoms" not in param_dict:
        raise KeyError("param_dict missing 'atoms' key.")
    atoms_tbl = param_dict["atoms"]

    def avg_charge(entry):
        qs = entry.get("charges", [])
        return (sum(qs) / len(qs)) if qs else 0.0

    charge_lookup = {atom_type: avg_charge(entry) for atom_type, entry in atoms_tbl.items()}

    if not single_ion_conductor or not sic_component_indices:
        if verbose:
            print("[SIC scaling] Skipped (either not SIC or no sic_component_indices).")
        return [0, 0], 1.0, set()

    sic_atom_types = set()
    for idx in sic_component_indices:
        if idx < 0 or idx >= n_components:
            raise IndexError(f"sic_component_indices contains out-of-range index {idx}.")
        sic_atom_types.update(atom_names_long_list[idx])

    total_polymer_charge_sic = sum(
        sum(charge_lookup.get(t, 0.0) for t in atom_names_long_list[i]) * polymer_count_list[i]
        for i in sic_component_indices
    )
    ideal_cation_count = sum(
        anions_per_polymer_list[i] * polymer_count_list[i] for i in sic_component_indices
    )
    target_polymer_charge_sic = -1.0 * ideal_cation_count * cation_charge

    if verbose:
        print(f"[SIC scaling] total unscaled SIC charge: {total_polymer_charge_sic:.6f}")
        print(f"[SIC scaling] total SIC anionic atoms (all chains): {ideal_cation_count}")
        print(f"[SIC scaling] target SIC polymer charge: {target_polymer_charge_sic:.6f}")

    if total_polymer_charge_sic == 0:
        scale_factor = 1.0
    else:
        scale_factor = target_polymer_charge_sic / total_polymer_charge_sic
    if verbose:
        print(f"[SIC scaling] scale_factor (SIC only): {scale_factor:.6f}")

    for atom_type in sic_atom_types:
        entry = atoms_tbl.get(atom_type)
        if not entry or not entry.get("charges"):
            continue
        qs = entry["charges"]
        atoms_tbl[atom_type]["charges"] = [sum(qs) / len(qs) * scale_factor]

    if verbose:
        total_scaled = sum(
            sum(atoms_tbl.get(t, {}).get("charges", [0.0])[0] for t in atom_names_long_list[i])
            * polymer_count_list[i]
            for i in sic_component_indices
        )
        print(f"[SIC scaling] total scaled SIC polymer charge: {total_scaled:.6f}")

    return [ideal_cation_count, 0], scale_factor, sic_atom_types
