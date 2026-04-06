import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Crippen, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_PROPERTY_FUNC = {
    "heavyatom": rdMolDescriptors.CalcNumHeavyAtoms,
    "hba": rdMolDescriptors.CalcNumHBA,
    "hbd": rdMolDescriptors.CalcNumHBD,
    "ring": rdMolDescriptors.CalcNumRings,
    "arom": rdMolDescriptors.CalcNumAromaticRings,
    "rotb": rdMolDescriptors.CalcNumRotatableBonds,
    "mw": rdMolDescriptors.CalcExactMolWt,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "logp": Crippen.MolLogP,
}
_PROPERTY_SCALE = {
    "heavyatom": 10,
    "hba": 5,
    "hbd": 5,
    "ring": 10,
    "arom": 5,
    "rotb": 5,
    "mw": 500,
    "tpsa": 50,
    "logp": 5,
}

FP_RADIUS: int = 2
MORGAN_FP_DIM: int = 512
NUM_MACCS_KEYS: int = 166
BLOCK_FP_DIM = MORGAN_FP_DIM + NUM_MACCS_KEYS

PROPERTY_KEY: list[str] = [
    "heavyatom",
    "hba",
    "hbd",
    "ring",
    "arom",
    "rotb",
    "mw",
    "tpsa",
    "logp",
]
PROPERTY_SCALE: list[float] = [_PROPERTY_SCALE[key] for key in PROPERTY_KEY]
NUM_PROPERTIES: int = len(PROPERTY_KEY)

ADDED_MASS_FOR_LINKER: float = 29.0  # molecular weight of -CH2CH3


def _get_mol_properties(mol: str | Chem.Mol) -> NDArray[np.float32]:
    """Common RDKit Descriptors"""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return np.array([_PROPERTY_FUNC[key](mol) for key in PROPERTY_KEY], np.float32)


def get_state_properties(mol: str | Chem.Mol) -> NDArray[np.float32]:
    """Get properties of a state molecule."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    prop = _get_mol_properties(mol)
    # Remove At atom MW
    ats = [int(atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetSymbol() == "At"]
    mw_removing_group = sum(ats)
    prop[PROPERTY_KEY.index("mw")] -= mw_removing_group
    return prop


def get_block_properties(mol: str | Chem.Mol) -> NDArray[np.float32]:
    """Get properties of a building block molecule."""
    # WARN: this is hard-coded for synple/eXplore block library
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    prop = _get_mol_properties(mol)
    ats = [int(atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetSymbol() == "At"]
    assert len(ats) in (1, 2, 3)
    if len(ats) == 1:  # arm
        at = ats[0]
        mw_removing_group = at
        mw_adding_group = 0
    else:  # linker
        mw_removing_group = sum(ats)
        mw_adding_group = ADDED_MASS_FOR_LINKER
    prop[PROPERTY_KEY.index("mw")] += mw_adding_group - mw_removing_group
    return prop


def get_block_features(
    mol: str | Chem.Mol,
    fp_radius: int = FP_RADIUS,
    fp_dim: int = MORGAN_FP_DIM,
) -> tuple[NDArray, NDArray]:
    """Get features of a building block molecule.
    This includes MACCS keys, Morgan fingerprint, and common RDKit descriptors.
    """

    # Setup Building Block Datas
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    assert mol is not None, "Invalid SMILES or molecule object"

    # MACCS keys are 1-indexed.
    maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float16)[
        1 : NUM_MACCS_KEYS + 1
    ]

    # morgan Fingerprint
    mg = GetMorganGenerator(fp_radius, fpSize=fp_dim)
    morgan_fp = mg.GetCountFingerprintAsNumPy(mol).astype(np.float16)

    # Concatenate MACCS and Morgan fingerprints
    fp = np.concatenate([maccs_fp, morgan_fp])

    # Common RDKit Descriptors
    prop = get_block_properties(mol)
    return fp, prop
