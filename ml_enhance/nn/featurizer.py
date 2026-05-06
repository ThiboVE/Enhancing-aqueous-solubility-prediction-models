from chemprop.featurizers import MultiHotAtomFeaturizer, MultiHotBondFeaturizer, SimpleMoleculeMolGraphFeaturizer
from rdkit.Chem.rdchem import HybridizationType


def get_custom_atom_featurizer() -> MultiHotAtomFeaturizer:
    # These represent all atoms present in the processed AqSol dataset
    atomic_nums = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 35, 53]
    degrees = [0, 1, 2, 3, 4, 5]
    formal_charges = []
    chiral_tags = [0, 1, 2, 3]
    num_Hs = [0, 1, 2, 3, 4]
    hybridizations = [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ]

    return MultiHotAtomFeaturizer(atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations)


def get_featurizer(
    *,
    use_custom_atom_featurizer: bool = False,
    use_custom_bond_featurizer: bool = False,
    extra_atom_fdim: int = 0,
    extra_bond_fdim: int = 0,
) -> SimpleMoleculeMolGraphFeaturizer:
    atom_featurizer = get_custom_atom_featurizer() if use_custom_atom_featurizer else MultiHotAtomFeaturizer.v2()

    if use_custom_bond_featurizer:
        raise NotImplementedError
    bond_featurizer = MultiHotBondFeaturizer()

    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        extra_atom_fdim=extra_atom_fdim,
        extra_bond_fdim=extra_bond_fdim,
    )
