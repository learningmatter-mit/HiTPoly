"""
Chain-construction tests for hitpoly.copolymer (RDKit only; no LigParGen, OpenMM or GPU).

    python -m pytest tests/test_copolymer_chains.py     # or: python tests/test_copolymer_chains.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem

from hitpoly.copolymer.chains import create_long_smiles, create_oligomers, get_atom_count
from hitpoly.copolymer.concentration import get_concentration_from_molality
from hitpoly.copolymer.junctions import count_bad_junctions, orient_monomers_for_junctions

P6 = "O=C(CCC[Cu])CCCO[Au]"                         # Poly 6 (one ketone per unit)
PEO = "[Au]CCO[Cu]"                                 # no carbonyl
DK = "[Cu]OCCCC(=O)CCCCC(=O)[Au]"                   # DiKeto
TRI = "[Cu]CC(C(=O)O)C(C(=O)O)C(C(=O)O)[Au]"        # Tricarb (3 COOH per unit)

PEROXIDE = Chem.MolFromSmarts("[OX2][OX2]")
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
KETONE = Chem.MolFromSmarts("[CX3](=O)")
COOH = Chem.MolFromSmarts("C(=O)[OX2H1]")


def n_matches(smiles, patt):
    return len(Chem.MolFromSmiles(smiles).GetSubstructMatches(patt))


def chain(smiles, fractions, total, mode, seed=0):
    random.seed(seed)
    return create_long_smiles(smiles=smiles, fractions=fractions, total_repeats=total,
                              polymerization_mode=mode, num_blocks=4, write_log=False)


def test_t1_no_peroxide_junctions():
    for pair in ([P6, PEO], [DK, P6]):
        for mode in ("alternating", "block", "random"):
            s, _ = chain(pair, [0.5, 0.5], 20, mode)
            assert n_matches(s, PEROXIDE) == 0, (pair, mode)
    s, _ = chain([DK, P6], [0.5, 0.5], 20, "alternating")
    assert n_matches(s, ESTER) > 0


def test_t1_unfixed_orientation_would_form_peroxides():
    # The raw CSV orientations are exactly the bad case the fix exists for.
    assert count_bad_junctions([P6, PEO]) > 0
    assert count_bad_junctions([DK, P6]) > 0
    assert count_bad_junctions(orient_monomers_for_junctions([P6, PEO])) == 0
    assert count_bad_junctions(orient_monomers_for_junctions([DK, P6])) == 0


def test_t2_block_composition():
    s, counts = chain([P6, PEO], [0.8, 0.2], 50, "block")
    assert n_matches(s, KETONE) == 40
    assert sorted(counts.values()) == [10, 40]


def test_t3_chain_matches_log():
    cases = [
        ("block", [0.5, 0.5], 62, [31, 31]),
        ("block", [0.8, 0.2], 51, [41, 10]),
        ("block", [0.29, 0.71], 100, [29, 71]),
        ("random", [0.5, 0.5], 62, [31, 31]),
        ("random", [0.8, 0.2], 50, [40, 10]),
        ("alternating", [0.5, 0.5], 62, [31, 31]),
    ]
    for mode, fr, total, expected in cases:
        s, counts = chain([P6, PEO], fr, total, mode)
        assert list(counts.values()) == expected, (mode, fr, total, counts)
        assert n_matches(s, KETONE) == expected[0], (mode, fr, total)
    s, counts = chain([P6], [1.0], 43, "homopolymer")
    assert list(counts.values()) == [43] and n_matches(s, KETONE) == 43


def test_t4_valid_pairs_untouched():
    assert orient_monomers_for_junctions([TRI, P6]) == [TRI, P6]
    for mode in ("block", "random"):
        s, counts = chain([TRI, P6], [0.5, 0.5], 20, mode)
        assert n_matches(s, COOH) // 3 == 10 and n_matches(s, KETONE) - n_matches(s, COOH) == 10


def test_t5_oligomers_parse():
    for smiles, mode in (([DK, P6], "alternating"), ([DK, P6], "block"), ([P6, PEO], "random"),
                         ([P6], "homopolymer"), ([P6], "star")):
        _, oligos = create_oligomers(smiles, mode)
        assert all(Chem.MolFromSmiles(o) is not None for o in oligos), (smiles, mode)
        assert all(n_matches(o, PEROXIDE) == 0 for o in oligos), (smiles, mode)


def _size(monomers, fractions, mode, arms=None):
    conc, repeats = get_concentration_from_molality(
        save_path=".", molality=1.0, polymer_count=30, monomers=monomers, fractions=fractions,
        add_end_Cs=True, polymerization_mode=mode, num_blocks=4, cation="Li", anion="N",
        polymer_chain_length=1000, arms=arms,
    )
    s, _ = create_long_smiles(smiles=monomers, fractions=fractions, total_repeats=repeats,
                              polymerization_mode=mode, num_blocks=4, arms=arms, write_log=False)
    return repeats, get_atom_count(s), conc[0]


def test_t6_sizing():
    assert _size([P6], [1.0], "homopolymer") == (43, 911, 166)
    assert _size([P6, PEO], [0.8, 0.2], "block") == (50, 918, 168)
    assert _size([P6, PEO], [0.5, 0.5], "alternating") == (64, 904, 166)
    assert _size([P6], [1.0], "star", arms=3) == (14, 902, 164)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
