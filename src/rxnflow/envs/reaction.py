from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


class Reaction:
    def __init__(self, template: str):
        self._rxn: ChemicalReaction = self.__init_reaction(template)
        self.num_reactants: int = self._rxn.GetNumReactantTemplates()
        self.pattern: str = template

    def __init_reaction(self, template: str) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def is_reactant(self, mol: RDMol, order: int) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        # return mol.HasSubstructMatch(self.reactant_pattern[order])
        return mol.HasSubstructMatch(self._rxn.GetReactantTemplate(order))

    def __call__(self, *reactants: RDMol) -> list[list[RDMol]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: RDMol
                reactants

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """
        return self.forward(*reactants)

    def forward(self, *reactants: RDMol, strict: bool = True) -> list[list[RDMol]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: reactants

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """

        # Run reaction
        assert len(reactants) == self.num_reactants
        ps: list[list[RDMol]] = self._rxn.RunReactants(tuple(reactants), 10)
        if strict and len(ps) == 0:
            # Logging for debugging.
            logger.error(
                "Reaction did not yield any products. Reactants: {}, SMARTS: {}",
                [Chem.MolToSmiles(mol) for mol in reactants],
                self.pattern,
            )
            raise ValueError("Reaction did not yield any products.")

        refine_ps: list[list[RDMol]] = []
        for p in ps:
            _p = []
            for mol in p:
                try:
                    mol = _refine_molecule(mol)
                except (
                    Chem.rdchem.KekulizeException,
                    Chem.rdchem.AtomKekulizeException,
                    Chem.rdchem.AtomValenceException,
                ):
                    continue
                _p.append(mol)
            if len(_p) == len(p):
                refine_ps.append(_p)
        refine_ps = _deduplicate_products(refine_ps)
        return refine_ps

    def forward_smi(
        self, *reactants: RDMol, strict: bool = True
    ) -> list[tuple[str, ...]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: reactants

        Returns:
            producs: list[list[str]]
                The smiles of products of the reaction.
        """

        # Run reaction
        assert len(reactants) == self.num_reactants
        ps: list[list[RDMol]] = self._rxn.RunReactants(tuple(reactants), 10)
        if strict and len(ps) == 0:
            # Logging for debugging.
            logger.error(
                "Reaction did not yield any products. Reactants: {}, SMARTS: {}",
                [Chem.MolToSmiles(mol) for mol in reactants],
                self.pattern,
            )
            raise ValueError("Reaction did not yield any products.")

        refine_ps: list[tuple[str, ...]] = []
        for p in ps:
            _p: list[str] = []
            for mol in p:
                try:
                    mol = Chem.RemoveHs(mol, updateExplicitCount=True)
                    smi = Chem.MolToSmiles(mol)
                except (
                    Chem.rdchem.KekulizeException,
                    Chem.rdchem.AtomKekulizeException,
                    Chem.rdchem.AtomValenceException,
                ):
                    break
                smi = smi.replace("[CH]", "C")
                _p.append(smi)
            if len(_p) == len(p):
                refine_ps.append(tuple(_p))
        return list(set(refine_ps))


def _refine_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)
    if "[CH]" in smi:
        smi = smi.replace("[CH]", "C")
    return Chem.MolFromSmiles(smi)


def _deduplicate_products(products: list[list[RDMol]]) -> list[list[RDMol]]:
    """Remove redundant cases from a `RunReactants` output.

    Compares each `list[Mol]` and leave only unique ones.
    """
    unique_cases: list[list[RDMol]] = []
    seen_cases: set[frozenset[str]] = set()
    for case in products:
        smi_in_case = frozenset(Chem.MolToSmiles(mol) for mol in case)
        if smi_in_case not in seen_cases:
            seen_cases.add(smi_in_case)
            unique_cases.append(case)
    return unique_cases
