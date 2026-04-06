import gc
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from ..config import ConstraintConfig
from ..gflownet.types import GFNEnvironmentContext
from ..utils.feature import (
    NUM_PROPERTIES,
    PROPERTY_KEY,
    PROPERTY_SCALE,
    get_state_properties,
)
from ..utils.vocab import AtomFeaturizer, BondFeaturizer
from .action import RxnAction, RxnActionType
from .env import MolGraph, SynthesisEnv
from .workflow import Workflow

# type aliases
BlockType = str
ClusterId = Any


@dataclass(frozen=True, slots=True)
class BlockTensor:
    type: int  # index of block type
    fp: torch.Tensor
    property: torch.Tensor
    tier: torch.Tensor

    def __len__(self) -> int:
        """Return the number of clusters in the block type."""
        return self.fp.shape[0]


class SynthesisEnvContext(
    GFNEnvironmentContext[RDMol, MolGraph, RxnAction, RxnActionType]
):
    """This context specifies how to create molecules by applying reaction templates."""

    action_type_order: list[RxnActionType] = [
        RxnActionType.SetWorkflow,
        RxnActionType.FirstBlock,
        RxnActionType.BiRxn,
        RxnActionType.UniRxn,
    ]
    bck_action_type_order: list[RxnActionType] = [
        RxnActionType.BckSetWorkflow,
        RxnActionType.BckFirstBlock,
        RxnActionType.BckBiRxn,
        RxnActionType.BckUniRxn,
    ]

    def __init__(
        self,
        env: SynthesisEnv,
        device: torch.device,
        property_constraint: ConstraintConfig,
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            property_constraint: dict[str, float]
        """
        # === load chemical reaction from Environment === #
        self.env: SynthesisEnv = env
        self.workflows: list[Workflow] = env.workflows
        self.workflow_dict: dict[str, Workflow] = env.workflow_dict
        self.num_workflows: int = len(self.workflows)
        self.workflow_to_idx: dict[str, int] = {
            workflow.id: i for i, workflow in enumerate(self.workflows)
        }

        # === Setup vocabulary and dimensions === #
        self.atom_vocab: AtomFeaturizer = AtomFeaturizer.get_featurizer_explore()
        self.bond_vocab: BondFeaturizer = BondFeaturizer.get_featurizer_explore()
        self.num_node_dim: int = len(self.atom_vocab)
        self.num_edge_dim: int = len(self.bond_vocab)
        self.num_graph_dim: int = (
            NUM_PROPERTIES  # mw, tpsa, numhbd, numhba, logp, rotbonds, numrings
        )

        # === Setup property standardization === #
        self._property_scale: np.ndarray = np.array(PROPERTY_SCALE, dtype=np.float32)

        # === Setup property constraint for action masking === #
        constraint_key: list[str] = [
            key for key in PROPERTY_KEY if property_constraint.get(key, None) is not None
        ]
        constraint_scale = [property_constraint[key] for key in constraint_key]
        self._constraint_scale: np.ndarray = np.array(constraint_scale, dtype=np.float32)
        self._constraint_idx: list[int] = [
            PROPERTY_KEY.index(key) for key in constraint_key
        ]

        # === Setup Building Block Datas === #
        env_dir: Path = Path(env.env_dir)
        self.setup_building_block_library(
            env_dir / "smiles",
            env_dir / "block_props.npz",
            env_dir / "block_fps.npz",
            env_dir / "block_clusters.npz",
        )
        self.block_type_to_idx: dict[str, int] = {
            block_type: i for i, block_type in enumerate(self.block_types)
        }
        self.num_block_clusters: dict[BlockType, int] = {
            block_type: len(self.block_clusters[block_type])
            for block_type in self.block_types
        }
        self.num_blocks: dict[BlockType, list[int]] = {
            block_type: [len(smis) for smis in cluster_smis]
            for block_type, cluster_smis in self.block_smis.items()
        }

        # Deliver layer dimensions.
        self.num_block_types: int = len(self.block_types)
        self.num_block_tiers: int = 5  # HACK: hardcoded. should it be in config?
        self.block_prop_dim: int = next(iter(self.block_props.values()))[0].shape[1]
        self.block_fp_dim: int = next(iter(self.block_fps.values()))[0].shape[1]

        # Cache the block features to avoid transferring them to GPU every time
        self.cache_block_features(device)

        # Estimate the chemical space for each workflow
        self.estimate_chemical_space(device)

    def setup_building_block_library(
        self,
        block_smiles_dir: Path,
        block_prop_path: Path,
        block_fp_path: Path,
        block_cluster_path: Path,
    ):
        """load building blocks and their features"""

        block_smis: dict[BlockType, list[str]] = {}
        block_codes: dict[BlockType, list[str]] = {}
        block_tiers: dict[BlockType, list[int]] = {}

        # load building blocks
        for file in Path(block_smiles_dir).iterdir():
            key = file.stem
            with file.open() as f:
                lines = f.readlines()
            # line format: {smi} \t Tier {tier} \t {code}
            block_smis[key] = [ln.split()[0] for ln in lines]
            block_codes[key] = [ln.strip().split()[3] for ln in lines]
            block_tiers[key] = [int(ln.split()[2]) for ln in lines]

        # load pre-computed block features
        block_props = np.load(block_prop_path)
        block_fps = np.load(block_fp_path)
        block_clusters = np.load(block_cluster_path)

        # get block info
        self.block_types: list[BlockType] = sorted(list(block_smis.keys()))

        # split into clusters
        self.block_clusters: dict[BlockType, list[ClusterId]] = {}
        self.block_smis: dict[BlockType, list[list[str]]] = {}
        self.block_codes: dict[BlockType, list[list[str]]] = {}
        self.block_tiers: dict[BlockType, list[list[int]]] = {}  # [Nblocks,]
        self.block_props: dict[BlockType, list[np.ndarray]] = {}  # [Nblocks, Nprops]
        self.block_fps: dict[BlockType, list[np.ndarray]] = {}  # [Nblocks, Nfps]

        for block_type in self.block_types:
            # cluster id should be starting from 0
            cluster_ids = block_clusters[block_type]
            uniq_cluster_ids = sorted(np.unique(cluster_ids).tolist())
            self.block_clusters[block_type] = uniq_cluster_ids

            # initialize empty list for this block type
            self.block_smis[block_type] = []
            self.block_codes[block_type] = []
            self.block_tiers[block_type] = []
            self.block_props[block_type] = []
            self.block_fps[block_type] = []

            props = block_props[block_type]
            fps = block_fps[block_type]

            for id in uniq_cluster_ids:
                # get indices of blocks in this cluster
                idxs = np.where(cluster_ids == id)[0]

                # get blocks smiles, codes, and tiers for this cluster
                _smis = [block_smis[block_type][i] for i in idxs]
                _codes = [block_codes[block_type][i] for i in idxs]
                _tiers = [block_tiers[block_type][i] for i in idxs]

                # get pre-computed properties, and fingerprints
                _props = props[idxs]
                _fps = fps[idxs]

                # append to lists
                self.block_smis[block_type].append(_smis)
                self.block_codes[block_type].append(_codes)
                self.block_tiers[block_type].append(_tiers)
                self.block_props[block_type].append(_props)
                self.block_fps[block_type].append(_fps)

        # close NPZFile
        block_props.close()
        block_fps.close()
        block_clusters.close()

    def get_workflow(self, workflow_idx: int) -> Workflow:
        assert workflow_idx >= 0, "Workflow ID must be non-negative"
        return self.workflows[workflow_idx]

    def graph_to_Data(self, g: MolGraph, do_cache: bool = True) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance

        Strategy:
        - Cache the data to avoid recomputing (`g.graph_cache`)
            - Store the dense representation of node and edge attributes to reduce memory usage
            - Store the numpy array instead of torch tensor for compatibility under various environments
        - Convert dense attributes to sparse representation for PyTorch Geometric

        """
        if g.graph_cache is not None:
            dense_feats: dict[str, Any] = g.graph_cache
        else:
            if g.num_atoms == 0:
                node_attr = np.zeros((1, self.atom_vocab.num_feats), dtype=np.int32)
                node_attr[0, -1] = 1  # set a dummy node for empty graph
                edge_attr = np.zeros((0, self.bond_vocab.num_feats), dtype=np.int32)
                edge_index = np.zeros((2, 0), dtype=np.int64)
                mol_properties = np.zeros((self.num_graph_dim,), dtype=np.float32)
            else:
                mol: Chem.Mol = self.graph_to_obj(g)

                # node attributes
                atoms: list[Chem.Atom] = mol.GetAtoms()
                node_attr = np.stack(
                    [self.atom_vocab.encode(atom) for atom in atoms], axis=0
                )  # [V, Fnode]

                # bi-directional edges
                bonds: list[Chem.Bond] = mol.GetBonds()
                edge_index_list = []
                edge_attr_list = []
                for bond in bonds:
                    # Add both directions for undirected graph
                    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_index_list.extend([(begin, end), (end, begin)])
                    edge_attr_list.extend([self.bond_vocab.encode(bond)] * 2)
                edge_attr = np.stack(edge_attr_list, axis=0)  # [E, Fedge]
                edge_index = np.array(edge_index_list, dtype=np.int64).T

                # molecular properties
                mol_properties = get_state_properties(mol)

            # molecular properties (global features)
            graph_attr = (mol_properties / self._property_scale).reshape(
                1, -1
            )  # [1, Fgraph]

            # for action-budget masking
            budget = self.compute_property_budget(mol_properties).reshape(
                1, -1
            )  # [1, Nmetric]

            # NOTE: Get action type - there is only one allowable action type for each state
            traj_idx = g.traj_idx
            protocol_order = traj_idx - 1  # initial action is SetWorkflow.
            if traj_idx == 0:
                # the first step of each generation is always workflow sampling
                action_type_idx = self.action_type_order.index(RxnActionType.SetWorkflow)
                workflow_idx = -1
            else:
                workflow = self.workflow_dict[g.workflow]
                workflow_idx = self.workflow_to_idx[workflow.id]
                protocol = workflow[protocol_order]
                action_type_idx = self.action_type_order.index(protocol.type)

            dense_feats = dict(
                node_attr=node_attr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                graph_attr=graph_attr,
                budget=budget,
                action_type=action_type_idx,
                workflow=workflow_idx,
                protocol_order=protocol_order,
            )
            if do_cache:
                g.graph_cache = dense_feats  # caching

        # dense to sparse conversion
        if dense_feats["node_attr"].shape[0] > 0:
            node_attr = np.stack(
                [self.atom_vocab.dense_to_sparse(v) for v in dense_feats["node_attr"]],
                axis=0,
            )
        else:
            node_attr = np.zeros((1, self.atom_vocab.size), dtype=np.float32)
        if dense_feats["edge_attr"].shape[0] > 0:
            edge_attr = np.stack(
                [self.bond_vocab.dense_to_sparse(v) for v in dense_feats["edge_attr"]],
                axis=0,
            )
        else:
            edge_attr = np.zeros((0, self.bond_vocab.size), dtype=np.float32)

        # return a torch geometric Data object
        return gd.Data(
            x=torch.from_numpy(node_attr),
            edge_attr=torch.from_numpy(edge_attr),
            edge_index=torch.from_numpy(dense_feats["edge_index"]),
            graph_attr=torch.from_numpy(dense_feats["graph_attr"]),
            budget=torch.from_numpy(dense_feats["budget"]),
            action_type=dense_feats["action_type"],
            workflow=dense_feats["workflow"],
            protocol_order=dense_feats["protocol_order"],
        )

    def compute_property_budget(self, mol_features: np.ndarray) -> np.ndarray:
        assert mol_features.ndim in (1, 2), "mol_features should be 1D or 2D tensor"
        if len(self._constraint_idx) > 0:
            former_dims = (1,) * (mol_features.ndim - 1)
            metrics = mol_features[..., self._constraint_idx]  # [..., Nmetric]
            budget = metrics / self._constraint_scale.reshape(*former_dims, -1)
        else:
            # if no constraint is given, return zero budget with shape of [..., 1]
            budget = np.zeros(mol_features.shape[:-1] + (1,), dtype=np.float32)
        assert mol_features.shape[:-1] == budget.shape[:-1]
        return budget

    def get_action(
        self,
        action_type: RxnActionType,
        workflow_idx: int = -1,
        protocol_order: int = -1,
        cluster_idx: int = -1,
        action_idx: int = -1,
    ) -> RxnAction:
        # the first step of each generation is workflow sampling
        if action_type is RxnActionType.SetWorkflow:
            workflow = self.get_workflow(action_idx)
            return RxnAction(action_type, workflow.id)

        workflow = self.get_workflow(workflow_idx)
        protocol = workflow[protocol_order]
        assert action_type is protocol.type, (
            "Action type does not match the protocol type"
        )
        match action_type:
            case RxnActionType.FirstBlock:
                block = self.block_smis[protocol.block_type][cluster_idx][action_idx]
                return RxnAction(
                    action_type, workflow.id, 0, block, cluster_idx, action_idx
                )
            case RxnActionType.BiRxn:
                block = self.block_smis[protocol.block_type][cluster_idx][action_idx]
                return RxnAction(
                    action_type,
                    workflow.id,
                    protocol_order,
                    block,
                    cluster_idx,
                    action_idx,
                )
            case RxnActionType.UniRxn:
                return RxnAction(action_type, workflow.id, protocol_order)
            case _:
                raise ValueError(protocol)

    def collate(self, graphs: list[gd.Data]) -> gd.Batch:
        return gd.Batch.from_data_list(graphs, follow_batch=["x"])

    def check_property_constraint(self, g: MolGraph, eps: float = 1e-3) -> bool:
        """Check if the molecule satisfies the property constraints."""
        if g.smi == "":
            # Initial state
            return True
        if len(self._constraint_idx) == 0:
            # No constraints
            return True
        mol_properties = get_state_properties(g.mol)
        metrics = mol_properties[self._constraint_idx]
        return np.all(metrics <= (self._constraint_scale * (1 + eps))).item()

    def obj_to_graph(self, obj: RDMol) -> MolGraph:
        """Convert an RDMol to a Graph"""
        g = MolGraph(obj)
        return g

    def graph_to_obj(self, g: MolGraph) -> RDMol:
        """Convert a Graph to an RDKit Mol"""
        return g.mol

    def object_to_log_repr(self, g: MolGraph) -> str:
        """Convert a Graph to a string representation"""
        return g.smi

    def traj_to_workflow(self, traj: list[tuple[MolGraph | RDMol, RxnAction]]) -> str:
        """Get trajectory's workflow"""
        return traj[0][1].workflow

    def traj_to_log_repr(self, traj: list[tuple[MolGraph | RDMol, RxnAction]]) -> str:
        """Convert a trajectory of (Graph, Action) to a trajectory of json representation"""
        repr_obj = []
        traj_logs = self.read_traj(traj)
        for i, (smiles, action_repr) in enumerate(traj_logs):
            repr_obj.append(
                OrderedDict([("step", i), ("smiles", smiles), ("action", action_repr)])
            )
        return json.dumps(repr_obj, sort_keys=False)

    def read_traj(
        self, traj: list[tuple[MolGraph, RxnAction]]
    ) -> list[tuple[str, tuple[str, ...]]]:
        """Convert a trajectory of (Graph, Action) to a trajectory of tuple representation"""
        traj_repr = []
        for g, action in traj:
            match action.action:
                case RxnActionType.SetWorkflow:
                    continue
                case RxnActionType.FirstBlock:
                    protocol = self.workflow_dict[action.workflow][0]
                    block_code = self.block_codes[protocol.block_type][
                        action.block_cluster_idx
                    ][action.block_idx]
                    action_repr = (protocol.name, action.block, block_code)
                case RxnActionType.BiRxn:
                    protocol = self.workflow_dict[action.workflow][action.protocol_order]
                    block_code = self.block_codes[protocol.block_type][
                        action.block_cluster_idx
                    ][action.block_idx]
                    action_repr = (protocol.name, action.block, block_code)
                case RxnActionType.UniRxn:
                    protocol = self.workflow_dict[action.workflow][action.protocol_order]
                    action_repr = (protocol.name,)
                case _:
                    raise ValueError(action.action)
            obj_repr = self.object_to_log_repr(g)
            traj_repr.append((obj_repr, action_repr))
        return traj_repr

    # === Block Features === #
    def cache_block_features(self, device: torch.device) -> None:
        """Cache the block features to avoid transferring them to GPU every time.

        This method should be called before the environment is used, to ensure that all block features are cached.
        It will cache the cluster fingerprints, block budgets, and block features for each block type and cluster index.
        The cached features will be used in `get_cluster_fps`, `get_block_budgets`, and `get_block_features` methods.

        """
        self._cache_cluster_fps: dict[str, Tensor] = {}
        self._cache_block_features: dict[tuple[str, int], BlockTensor] = {}
        self._cache_block_budgets: dict[tuple[str, int], Tensor] = {}

        for block_type in self.block_types:
            self.get_cluster_fps(block_type, device)
            for cluster_idx in range(self.num_block_clusters[block_type]):
                self.get_block_budgets(block_type, cluster_idx, device)
                self.get_block_features(block_type, cluster_idx, device)

        # free memory
        del self.block_fps
        del self.block_props
        gc.collect()

    def get_cluster_fps(
        self,
        block_type: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Get the cluster fingerprints of a block type."""
        # Cache the cluster fingerprints to avoid transferring them to GPU every time
        if block_type not in self._cache_cluster_fps:
            # Get representative features for each cluster
            center_fps = np.stack(
                [fps.mean(axis=0) for fps in self.block_fps[block_type]], axis=0
            )  # [Ncluster, Ffp]
            center_fps = torch.from_numpy(center_fps)
            self._cache_cluster_fps[block_type] = center_fps.to(
                dtype=torch.float32, device=device
            )
        return self._cache_cluster_fps[block_type]

    def get_block_budgets(
        self,
        block_type: str,
        cluster_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get the budget of a block type and cluster index."""
        key = (block_type, cluster_index)
        if key not in self._cache_block_budgets:
            prop = self.block_props[block_type][cluster_index]
            budget = self.compute_property_budget(prop)
            self._cache_block_budgets[key] = torch.from_numpy(budget).to(
                dtype=torch.float32, device=device
            )
        return self._cache_block_budgets[key]

    def get_block_features(
        self,
        block_type: str,
        cluster_index: int,
        device: torch.device,
    ) -> BlockTensor:
        # Cache the block features to avoid transferring them to GPU every time
        key = (block_type, cluster_index)
        if key not in self._cache_block_features:
            """Get the features of a block type and cluster index."""
            type_idx = self.block_type_to_idx[block_type]  # integer
            prop = torch.from_numpy(
                self.block_props[block_type][cluster_index]
            )  # [Nblock, Fproperty]
            fp = torch.from_numpy(
                self.block_fps[block_type][cluster_index]
            )  # [Nblock, Ffp]

            # standardize property
            prop = prop / self._property_scale.reshape(1, -1)

            # get tier
            tier = torch.tensor(
                self.block_tiers[block_type][cluster_index], dtype=torch.int32
            )  # [Nblock,]
            tier = tier - 1  # [1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]

            self._cache_block_features[key] = BlockTensor(
                type=type_idx,
                fp=fp.to(dtype=torch.float32, device=device),
                property=prop.to(dtype=torch.float32, device=device),
                tier=tier.to(device),
            )
        return self._cache_block_features[key]

    def estimate_chemical_space(self, device: torch.device) -> None:
        """Estimate the chemical space for each workflow."""
        MAX_SAMPLES = 1000
        THRESHOLD = 1 + 1e-2

        self.workflow_chemical_space: dict[str, int] = {}
        for workflow in self.workflows:
            protocols = workflow.protocols
            # Start with initial state budget
            state_budgets = torch.zeros(
                (1, 1), dtype=torch.float32, device=device
            )  # [1, Nprop]
            # To decrease the computation cost, we sample a limited number of states and actions.
            # To estimate the entire chemical space from the sampled state/actions, we record the ratio of sampled states.
            sampled_ratio = 1.0
            # Accumulate the block budgets
            for protocol in protocols:
                if protocol.type is RxnActionType.UniRxn:
                    # there are no blocks for UniRxn action
                    continue

                # Sample states
                if state_budgets.shape[0] > MAX_SAMPLES:
                    sampled_ratio *= MAX_SAMPLES / state_budgets.shape[0]
                    state_budgets = state_budgets[
                        torch.randperm(state_budgets.shape[0])[:MAX_SAMPLES]
                    ]

                # Get the block budgets for the current protocol
                block_type = protocol.block_type
                block_budgets = torch.cat(
                    [
                        self.get_block_budgets(block_type, cluster_idx, device)
                        for cluster_idx in range(self.num_block_clusters[block_type])
                    ],
                    dim=0,
                )  # [Nblock, Nprop]

                # Filter out blocks
                is_valid = (block_budgets < THRESHOLD).all(-1)
                block_budgets = block_budgets[is_valid]

                # If no valid blocks, return 0
                if block_budgets.shape[0] == 0:
                    state_budgets = torch.zeros((0, 1))
                    break

                # Sample blocks
                if block_budgets.shape[0] > MAX_SAMPLES:
                    sampled_ratio *= MAX_SAMPLES / block_budgets.shape[0]
                    block_budgets = block_budgets[
                        torch.randperm(block_budgets.shape[0])[:MAX_SAMPLES]
                    ]

                # Accumulate the budgets
                all_pair_budgets = state_budgets.unsqueeze(1) + block_budgets.unsqueeze(0)
                # Filter out with budgets
                is_valid = (all_pair_budgets < THRESHOLD).all(-1)
                state_budgets = all_pair_budgets[is_valid]  # [Nstate, Nprop]

            # the chemical space is the number of valid budgets
            self.workflow_chemical_space[workflow.id] = int(
                state_budgets.shape[0] / sampled_ratio
            )
