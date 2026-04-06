from collections.abc import Sequence
from typing import Self

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondStereo, BondType, ChiralType, HybridizationType

# dtypes
DenseArr = np.ndarray[tuple[int], np.dtype[np.int32]]
SparseArr = np.ndarray[tuple[int], np.dtype[np.float32]]

# atom type
DEFAULT_ATOM_TYPES: list[str] = [
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "At",
]
DEFAULT_CHARGES: list[int] = [-2, -1, 0, 1, 2]
DEFAULT_DEGREES: list[int] = [0, 1, 2, 3, 4, 5]
DEFAULT_CHIRAL_TYPES: list[ChiralType] = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
]
DEFAULT_NUM_H_RANGE: list[int] = [0, 1, 2, 3, 4]
DEFAULT_HYBRIDIZATION_TYPES: list[HybridizationType] = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
]

# bond type
DEFAULT_BOND_TYPES: list[BondType] = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]
DEFAULT_BOND_STEREO: list[BondStereo] = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
]


class Vocabulary[T]:
    def __init__(self, vocab: Sequence[T], allow_unk: bool = True):
        self.vocab: tuple[T, ...] = tuple(vocab)
        self.vocab_dict: dict[T, int] = {v: i for i, v in enumerate(vocab)}
        self.allow_unk: bool = allow_unk
        self.unk_id: int = len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab) + int(self.allow_unk)

    def encode(self, token: T) -> int:
        if self.allow_unk:
            return self.vocab_dict.get(token, self.unk_id)
        else:
            return self.vocab_dict[token]

    def decode(self, id: int) -> T:
        return self.vocab[id]


class AtomFeaturizer:
    def __init__(
        self,
        atom_types: Sequence[str],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[ChiralType],
        num_Hs: Sequence[int],
        hybridization_types: Sequence[HybridizationType],
        isotopes: Sequence[int],
    ):
        self.atom_type = Vocabulary[str](atom_types)
        self.degree = Vocabulary[int](degrees)
        self.charge = Vocabulary[int](formal_charges)
        self.chiral = Vocabulary[ChiralType](chiral_tags)
        self.numH = Vocabulary[int](num_Hs)
        self.hybridization = Vocabulary[str](hybridization_types)
        self.isotope = Vocabulary[int](isotopes)

        self._subfeat_size: list[int] = [
            len(self.atom_type),
            len(self.degree),
            len(self.charge),
            len(self.chiral),
            len(self.numH),
            len(self.hybridization),
            len(self.isotope),
            1,  # aromatic
            1,  # mass
            1,  # dummy (dummy node for empty graph - graph transformer)
        ]
        self.num_feats: int = len(self._subfeat_size)
        self.size: int = sum(self._subfeat_size)

    def __len__(self) -> int:
        return self.size

    def encode(self, atom: Chem.Atom) -> DenseArr:
        """Encode a value to its index in the vocabulary.
        inspired by ChemProp
        """
        feats = [
            self.atom_type.encode(atom.GetSymbol()),
            self.degree.encode(atom.GetTotalDegree()),
            self.charge.encode(atom.GetFormalCharge()),
            self.chiral.encode(atom.GetChiralTag()),
            self.numH.encode(atom.GetTotalNumHs()),
            self.hybridization.encode(atom.GetHybridization()),
            self.isotope.encode(atom.GetIsotope()),
            int(atom.GetIsAromatic()),
            int(atom.GetMass()),
            0,  # flag to indicate a dummy node (for empty graphs)
        ]
        return np.array(feats, dtype=np.int32)

    def dense_to_sparse(self, feats: DenseArr) -> SparseArr:
        """Convert dense features to sparse representation."""
        x = np.zeros(self.size, dtype=np.float32)
        # Encode each feature into the sparse vector
        offset: int = 0
        for v, num_feats in zip(feats[:-3], self._subfeat_size[:-3], strict=True):
            x[offset + int(v)] = 1.0
            offset += num_feats
        assert offset + 3 == self.size, "Offset mismatch in sparse encoding"
        # Insert the last three features
        x[-3] = float(feats[-3])  # aromatic
        x[-2] = float(feats[-2] / 100)  # mass
        x[-1] = float(feats[-1])  # flag for dummy node
        return x

    @classmethod
    def get_featurizer_explore(cls) -> Self:
        return cls(
            atom_types=DEFAULT_ATOM_TYPES,
            degrees=DEFAULT_DEGREES,
            formal_charges=DEFAULT_CHARGES,
            chiral_tags=DEFAULT_CHIRAL_TYPES,
            num_Hs=DEFAULT_NUM_H_RANGE,
            hybridization_types=DEFAULT_HYBRIDIZATION_TYPES,
            isotopes=DEFAULT_ISOTOPE,
        )


class BondFeaturizer:
    def __init__(
        self,
        bond_types: Sequence[BondType],
        stereo: Sequence[BondStereo],
    ):
        self.bond_type = Vocabulary[str](bond_types, allow_unk=False)
        self.stereo = Vocabulary[BondStereo](stereo)

        self._subfeat_size: list[int] = [
            len(self.bond_type),
            len(self.stereo),
            1,  # is_conjugated
            1,  # is_in_ring
        ]
        self.num_feats: int = len(self._subfeat_size)
        self.size: int = sum(self._subfeat_size)

    def __len__(self) -> int:
        return self.size

    def encode(self, bond: Chem.Bond) -> DenseArr:
        """Encode a value to its index in the vocabulary.
        inspired by ChemProp
        """

        feats = [
            self.bond_type.encode(bond.GetBondType()),
            self.stereo.encode(bond.GetStereo()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        return np.array(feats, dtype=np.int32)

    def dense_to_sparse(self, feats: DenseArr) -> SparseArr:
        x = np.zeros(self.size, dtype=np.float32)
        # Encode each feature into the sparse vector
        offset: int = 0
        for v, num_feats in zip(feats[:-2], self._subfeat_size[:-2], strict=True):
            x[offset + v] = 1.0
            offset += num_feats
        assert offset + 2 == self.size, "Offset mismatch in sparse encoding"
        # insert the last two features
        x[-2] = float(feats[-2])
        x[-1] = float(feats[-1])
        return x

    @classmethod
    def get_featurizer_explore(cls) -> Self:
        return cls(
            bond_types=DEFAULT_BOND_TYPES,
            stereo=DEFAULT_BOND_STEREO,
        )
