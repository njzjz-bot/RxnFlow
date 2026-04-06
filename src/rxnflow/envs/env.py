from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, RDLogger

from ..gflownet.types import GFNEnvironment
from .action import RxnAction, RxnActionType
from .workflow import Protocol, Workflow

RDLogger.DisableLog("rdApp.*")


class MolGraph:
    mol: Chem.Mol
    smi: str

    def __init__(self, mol: str | Chem.Mol = ""):
        if isinstance(mol, Chem.Mol):
            self.mol = mol
            self.smi = Chem.MolToSmiles(mol)
        else:
            self.mol = Chem.MolFromSmiles(mol)
            self.smi = mol
        self.graph_cache: Any = None

    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumHeavyAtoms()

    def __repr__(self):
        return f"MolGraph({self.smi})"

    def __getstate__(self):
        """Get the state of the object for pickling."""
        state = self.__dict__.copy()
        # remove the RDKit Mol object and data cache for pickling
        state.pop("mol", None)
        state.pop("graph_cache", None)
        return state

    def __setstate__(self, state):
        """Set the state of the object after unpickling."""
        self.__dict__.update(state)
        # Recreate the RDKit Mol object from the SMILES string
        self.mol = Chem.MolFromSmiles(self.smi)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES string: {self.smi}")
        # Initialize data cache
        self.graph_cache = None


class SynthesisEnv(GFNEnvironment[MolGraph, RxnAction]):
    """Molecules and reaction templates environment. The new (initial) state are Empty Molecular Graph.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(self, env_dir: str | Path):
        """A reaction template and building block environment instance"""
        self.env_dir = env_dir = Path(env_dir)

        workflow_config_path = env_dir / "workflow_map.csv"
        protocol_config_path = env_dir / "protocol.yaml"
        workflow_config: pd.DataFrame = pd.read_csv(workflow_config_path, index_col=0)
        protocol_config: DictConfig = OmegaConf.load(protocol_config_path)
        # first set protocols
        all_protocols: list[Protocol] = []
        for type_str, cfg_dict in protocol_config.items():
            match str(type_str):
                case "FirstBlock":
                    action_type = RxnActionType.FirstBlock
                case "BiRxn":
                    action_type = RxnActionType.BiRxn
                case "UniRxn":
                    action_type = RxnActionType.UniRxn
                case _:
                    raise ValueError(type_str)
            for name, cfg in cfg_dict.items():
                all_protocols.append(Protocol(name, action_type, **cfg))
        protocol_dict: dict[str, Protocol] = {
            protocol.name: protocol for protocol in all_protocols
        }

        # then load workflows
        self.workflows: list[Workflow] = []
        for workflow_id, row in workflow_config.iterrows():
            workflow_id = str(workflow_id)
            workflow_name = str(row["workflow name"])
            protocols = [
                protocol_dict[str(p)]
                for p in [row["protocol 1"], row["protocol 2"], row["protocol 3"]]
                if pd.notna(p)
            ]
            self.workflows.append(Workflow(workflow_id, workflow_name, protocols))
        self.workflow_dict: dict[str, Workflow] = {
            workflow.id: workflow for workflow in self.workflows
        }
        self.num_workflows: int = len(self.workflow_dict)

    def new(self) -> MolGraph:
        """get initial graph"""
        return MolGraph("")

    def step(self, g: MolGraph, action: RxnAction) -> MolGraph:
        """Applies the action to the current state and returns the next state.

        Parameters
        ----------
        g: MolGraph
            The current state of the environment, which is a molecular graph.
        action: RxnAction
            The action to be applied to the current state.

        Returns
        -------
        MolGraph
            The next state of the environment after applying the action.
        """
        match action.action:
            case RxnActionType.SetWorkflow:
                assert g.smi == "", "SetWorkflow should be called on empty graph"
                assert g.workflow == "", "SetWorkflow should be called on empty graph"
                assert g.traj_idx == 0, "SetWorkflow should be called on empty graph"
                return MolGraph(workflow=action.workflow, traj_idx=1)

            case RxnActionType.FirstBlock:
                assert g.workflow == action.workflow, "Workflow should be same"
                assert g.smi == "", "FirstBlock should be called on empty graph"
                assert g.traj_idx == 1, (
                    "FirstBlock should be the second action in the trajectory"
                )
                return MolGraph(mol=action.block, workflow=action.workflow, traj_idx=2)

            case RxnActionType.BiRxn:
                assert g.workflow == action.workflow, "workflow should be same"
                assert g.smi != "", "BiRxn should not be called on empty graph"
                assert g.traj_idx > 1, (
                    "BiRxn should be called after SetWorkflow and FirstBlock"
                )
                workflow = self.workflow_dict[action.workflow]
                protocol = workflow[action.protocol_order]
                block = Chem.MolFromSmiles(action.block)
                ps = protocol.rxn_forward(g.mol, block)
                if len(ps) != 1:
                    logger.error(
                        "Multiple or no products from reactant: {} block: {} reaction: {}",
                        Chem.MolToSmiles(g.mol),
                        Chem.MolToSmiles(block),
                        protocol.forward,
                    )
                    raise RuntimeError("Multiple or no products from reactant")
                return MolGraph(
                    mol=ps[0][0], workflow=action.workflow, traj_idx=g.traj_idx + 1
                )

            case RxnActionType.UniRxn:
                assert g.workflow == action.workflow, "Workflow should be same"
                assert g.smi != "", "UniRxn should not be called on empty graph"
                assert g.traj_idx > 1, (
                    "UniRxn should be called after SetWorkflow and FirstBlock"
                )
                assert action.block == "", "Block should not be set for UniRxn"
                workflow = self.workflow_dict[action.workflow]
                protocol = workflow[action.protocol_order]
                ps = protocol.rxn_forward(g.mol)
                if len(ps) != 1:
                    logger.error(
                        "Multiple or no products from reactant: {} reaction: {}",
                        Chem.MolToSmiles(g.mol),
                        protocol.forward,
                    )
                    raise RuntimeError("Multiple or no products from reactant")
                return MolGraph(
                    mol=ps[0][0], workflow=action.workflow, traj_idx=g.traj_idx + 1
                )

            case _:
                raise ValueError(action.action)

    def reverse(self, ra: RxnAction) -> RxnAction:
        """Returns the reverse action of the given action."""
        raise RuntimeWarning("Reverse actions are not used in RxnFlow")
        match ra.action:
            case RxnActionType.SetWorkflow:
                return RxnAction(RxnActionType.BckFirstBlock, ra.workflow)
            case RxnActionType.BckSetWorkflow:
                return RxnAction(RxnActionType.FirstBlock, ra.workflow)
            case RxnActionType.FirstBlock:
                return RxnAction(
                    RxnActionType.BckFirstBlock,
                    ra.workflow,
                    0,
                    ra.block,
                    ra.block_cluster_idx,
                    ra.block_idx,
                )
            case RxnActionType.BckFirstBlock:
                return RxnAction(
                    RxnActionType.FirstBlock,
                    ra.workflow,
                    0,
                    ra.block,
                    ra.block_cluster_idx,
                    ra.block_idx,
                )
            case RxnActionType.BiRxn:
                return RxnAction(
                    RxnActionType.BckBiRxn,
                    ra.workflow,
                    ra.order,
                    ra.block,
                    ra.block_cluster_idx,
                    ra.block_idx,
                )
            case RxnActionType.BckBiRxn:
                return RxnAction(
                    RxnActionType.BiRxn,
                    ra.workflow,
                    ra.order,
                    ra.block,
                    ra.block_cluster_idx,
                    ra.block_idx,
                )
            case RxnActionType.UniRxn:
                return RxnAction(RxnActionType.BckUniRxn, ra.workflow)
            case RxnActionType.BckUniRxn:
                return RxnAction(RxnActionType.UniRxn, ra.workflow)
