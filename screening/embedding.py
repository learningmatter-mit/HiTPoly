from rdkit import Chem
import math
import torch
import itertools
import numpy as np

def generate_longsmiles(smiles):
    canon_smiles = []
    long_smiles_list = []

    for i in range(len(smiles)):
        smil = smiles[i].replace('[C:2]', 'C')
        smil = smil.replace('[C:1]', 'C')
        smil = smil.replace('[CH:1]', 'C')
        smil = smil.replace('[CH:2]', 'C')
        smil = smil.replace('[CH2:1]', 'C')
        smil = smil.replace('[CH2:2]', 'C')
        smil = smil.replace('[c:1]', 'c')
        smil = smil.replace('[cH:1]', 'c')
        smil = smil.replace('[c:2]', 'c')
        smil = smil.replace('[cH:2]', 'c')
        smil = smil.replace('[c:3]', 'c')
        smil = smil.replace('[cH:3]', 'c')
        smil = smil.replace('[c:4]', 'c')
        smil = smil.replace('[cH:4]', 'c')
        smil = smil.replace('[c:5]', 'c')
        smil = smil.replace('[cH:5]', 'c')
        smil = smil.replace('[c:6]', 'c')
        smil = smil.replace('[cH:6]', 'c')
        smil = smil.replace('[O:1]', 'O')
        smil = smil.replace('[O:2]', 'O')
        smil = smil.replace('[N:1]', 'N')
        smil = smil.replace('[N:2]', 'N')
        smil = smil.replace('[NH:1]', 'N')
        smil = smil.replace('[NH:2]', 'N')
        smil = Chem.CanonSmiles(smil)

        reaction = "[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]"

        long_smiles, _ = create_long_smiles(
            smil,
            repeats=30,
            add_end_Cs=True,
            reaction=reaction,
            product_index=0,
        )

        mol_long = Chem.MolFromSmiles(long_smiles)
        mol_long = Chem.AddHs(mol_long)
        r_long, atom_names_long, atoms_long, bonds_typed_long = generate_atom_types(
            mol_long, 2
        )

        ligpargen_repeats, smiles_initial, mol_initial, r, atom_names, atoms, bonds_typed = create_ligpargen_short_polymer(
            smil,
            add_end_Cs=True,
            reaction=reaction,
            product_index=0,
            atom_names_long=atom_names_long
        )

        smiles_repeat, _ = create_long_smiles(
            smil,
            repeats = ligpargen_repeats,
            add_end_Cs=True,
            reaction=reaction,
            product_index=0,
        )
        
        canon_smiles.append(smil)
        long_smiles_list.append(smiles_repeat)
        
    return canon_smiles, long_smiles_list

def smiles_encoder(smiles):
    smiles_chars = list(set(smiles))

    smi2index = dict((c, i) for i, c in enumerate(smiles_chars))
    index2smi = dict((i, c) for i, c in enumerate(smiles_chars))

    X = np.zeros((len(smiles), len(smiles_chars)))
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
    return X

def create_ligpargen_smiles(
    smiles,
    repeats,
    add_end_Cs,
    reaction,
    product_index,
):
    smiles_initial, repeats = create_long_smiles(
        smile=smiles,
        repeats=repeats,
        add_end_Cs=add_end_Cs,
        reaction=reaction,
        product_index=product_index,
    )

    mol_initial = Chem.MolFromSmiles(smiles_initial)
    mol_initial = Chem.AddHs(mol_initial, addCoords=True)

    return mol_initial, smiles_initial

def create_ligpargen_short_polymer(
    smiles,
    add_end_Cs,
    reaction,
    product_index,
    atom_names_long,
):
    atom_names_long_sorted = sorted(set(atom_names_long))

    for ligpargen_repeats in range(30, 1, -1):
        mol_initial, smiles_initial = create_ligpargen_smiles(
            smiles=smiles,
            repeats=ligpargen_repeats,
            add_end_Cs=add_end_Cs,
            reaction=reaction,
            product_index=product_index,
        )

        r, atom_names, atoms, bonds_typed = generate_atom_types(mol_initial, 2)

        if len(atom_names) < 200 and sorted(set(atom_names)) == atom_names_long_sorted:
            break
        else:
            continue

    return (
        ligpargen_repeats,
        smiles_initial,
        mol_initial,
        r,
        atom_names,
        atoms,
        bonds_typed,
    )

def index_of(input, source):
    source, sorted_index, inverse = np.unique(
        source.tolist(), return_index=True, return_inverse=True, axis=0
    )
    index = torch.cat([torch.tensor(source), input]).unique(
        sorted=True, return_inverse=True, dim=0
    )[1][-len(input) :]
    try:
        index = torch.tensor(sorted_index)[index]
    except:
        print("error in one-hot encoding")
        import IPython

        IPython.embed()
    return index

def create_long_smiles(
    smile,
    repeats=None,
    req_length=None,
    mw=None,
    add_end_Cs=True,  # default is to add Cs
    reaction="[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]",
    product_index=0,
):
    # check if smile is a polymer
    if "Cu" in smile:
        #         calculate required repeats so smiles > 30 atoms long
        if repeats:
            repeats -= 1
        elif req_length:
            num_heavy = Chem.Lipinski.HeavyAtomCount(Chem.MolFromSmiles(smile)) - 2
            repeats = math.ceil(req_length / num_heavy) - 1
        elif mw:
            temp_smiles = smile.replace("[Cu]", "").replace("[Au]", "")
            repeats = math.ceil(
                mw
                / (Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(temp_smiles)) / 1000)
            )
        if repeats > 0:
            try:
                # code to increase length of monomer
                mol = Chem.MolFromSmiles(smile)
                new_mol = mol

                # join repeats number of monomers into polymer
                for i in range(repeats):
                    # join two polymers together at Cu and Au sites
                    rxn = Chem.AllChem.ReactionFromSmarts(reaction)
                    results = rxn.RunReactants((mol, new_mol))
                    assert len(results) == 1 and len(results[0]) == 1, smile
                    new_mol = results[product_index][0]

                new_smile = Chem.MolToSmiles(new_mol)

            except:
                # make smile none if reaction fails
                return "None"

        # if monomer already long enough use 1 monomer unit
        else:
            new_smile = smile

        # caps ends of polymers with carbons
        if add_end_Cs:
            new_smile = (
                new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "")
            )
        else:
            new_smile = (
                new_smile.replace("[Cu]", "").replace("[Au]", "").replace("[Ca]", "")
            )
    else:
        new_smile = smile
        repeats = 0

    # make sure new smile in cannonical
    long_smile = Chem.MolToSmiles(Chem.MolFromSmiles(new_smile))
    return long_smile, repeats + 1

def generate_atom_types(mol, depth):
    neighbors = [[x.GetIdx() for x in y.GetNeighbors()] for y in mol.GetAtoms()]
    atoms = [x.GetSymbol() for x in mol.GetAtoms()]

    bonds = [list(itertools.product(x, [ind])) for ind, x in enumerate(neighbors)]
    bonds = list(itertools.chain(*bonds))
    bonds = torch.LongTensor([list(b) for b in bonds])

    bonds_typed = bonds[bonds[:, 1] > bonds[:, 0]].tolist()

    a = bonds.view(-1, 2)
    d = torch.tensor([len(x) for x in neighbors])
    r = torch.tensor(smiles_encoder(atoms))

    unique_values = {}

    for rad in range(depth + 1):
        if rad != 0:
            # the message is in the direction (atom 1 -> atom 0) for each edge, so the message is the current atom label of atom1
            # the messages from each incoming atom are then split by receiving atom,
            # so the messages that are going into a particular atom are all grouped together
            messages = list(torch.split(r[a[:, 0]], d.tolist()))
            # the messages incoming to each atom are then added together to enforce permutation invariance
            messages = [messages[n].sum(0) for n in range(len(d))]
            messages = torch.stack(messages)

            # the message is then appended to the current state to remember order of messages
            r = torch.cat([r, messages], dim=1)
        if rad not in unique_values.keys():
            unique_values[rad], one_hot_mapping = r.unique(dim=0, return_inverse=True)
        index = index_of(r, unique_values[rad])
        r = torch.eye(len(unique_values[rad])).to(torch.long)[index]
    print("One-hot encoding has the shape", r.unique(dim=1).shape)

    atom_names = []
    for i in torch.nonzero(r):
        atom_names.append(atoms[i[0]] + str(i[1].cpu().numpy()))

    return r, atom_names, atoms, bonds_typed