"""
Monomer orientation for copolymer chains.

Chains are grown with '[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]': the atom next to [Cu] in one
repeat unit bonds to the atom next to [Au] in the next. Monomer SMILES are written with
arbitrary head/tail orientation, so in a copolymer the A->B or B->A junction can bond two
heteroatoms (e.g. Poly 6 `O=C(CCC[Cu])CCCO[Au]` followed by PEO `[Au]CCO[Cu]` gives an
O-O peroxide), even though every homopolymer junction is fine.

orient_monomers_for_junctions() reverses monomers (swaps their [Cu]/[Au] labels) when that
removes such junctions. Reversing a monomer leaves its own homopolymer junction unchanged,
so homopolymers and already-valid pairs are returned untouched.

check_chain_junctions() is the safety net: it counts heteroatom-heteroatom bonds in a built
chain and raises if the chain has more than its repeat units contain on their own.
"""
import itertools

from rdkit import Chem

HETERO = {"O", "N", "S"}


def flip_monomer(smiles):
    """Reverse a repeat unit's direction by swapping its [Cu] and [Au] labels."""
    return smiles.replace("[Cu]", "[#TMP]").replace("[Au]", "[Cu]").replace("[#TMP]", "[Au]")


def _end_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse monomer SMILES: {smiles}")
    ends = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ("Cu", "Au"):
            ends[atom.GetSymbol()] = atom.GetNeighbors()[0].GetSymbol()
    if set(ends) != {"Cu", "Au"}:
        raise ValueError(f"Monomer must contain exactly one [Cu] and one [Au]: {smiles}")
    return ends


def junction_bonds(smiles_list):
    """(i, j, atom_i, atom_j) for the bond formed when monomer j follows monomer i, for every i != j."""
    ends = [_end_atoms(s) for s in smiles_list]
    return [
        (i, j, ends[i]["Cu"], ends[j]["Au"])
        for i, j in itertools.permutations(range(len(smiles_list)), 2)
    ]


def count_bad_junctions(smiles_list):
    return sum(1 for _, _, a, b in junction_bonds(smiles_list) if a in HETERO and b in HETERO)


def orient_monomers_for_junctions(smiles_list, strict=True):
    """
    Return the monomer list with orientations chosen so that no junction between two
    different monomers bonds two heteroatoms. The first monomer keeps its orientation;
    among equally good choices the one with the fewest reversals wins, so valid inputs
    are returned unchanged.

    If no orientation avoids every bad junction, raises ValueError (strict=True) or
    prints a warning and returns the best orientation found (strict=False).
    """
    smiles_list = list(smiles_list)
    if len(smiles_list) < 2:
        return smiles_list

    best, best_key = smiles_list, None
    for flips in itertools.product([False, True], repeat=len(smiles_list) - 1):
        candidate = [smiles_list[0]] + [
            flip_monomer(s) if f else s for s, f in zip(smiles_list[1:], flips)
        ]
        key = (count_bad_junctions(candidate), sum(flips))
        if best_key is None or key < best_key:
            best, best_key = candidate, key

    if best_key[0] > 0:
        msg = (f"{best_key[0]} heteroatom-heteroatom junction(s) remain for {smiles_list}; "
               f"no monomer orientation avoids them.")
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}")
    return best


def count_heteroatom_bonds(mol):
    """Number of bonds between two atoms in HETERO."""
    return sum(
        1 for b in mol.GetBonds()
        if b.GetBeginAtom().GetSymbol() in HETERO and b.GetEndAtom().GetSymbol() in HETERO
    )


def check_chain_junctions(chain_smiles, unit_counts):
    """
    Raise if the chain has more heteroatom-heteroatom bonds than its repeat units contain
    internally, i.e. if any junction (or end cap) bonded two heteroatoms.

    unit_counts: {monomer SMILES: number of that unit in the chain}.
    """
    chain = Chem.MolFromSmiles(chain_smiles)
    if chain is None:
        raise ValueError("Chain SMILES could not be parsed by RDKit.")
    expected = sum(
        count_heteroatom_bonds(Chem.MolFromSmiles(s)) * n for s, n in unit_counts.items()
    )
    found = count_heteroatom_bonds(chain)
    if found > expected:
        raise ValueError(
            f"Chain has {found} heteroatom-heteroatom bonds but its repeat units only account "
            f"for {expected}: {found - expected} junction(s) bond two heteroatoms (O-O, O-S, N-O...)."
        )
