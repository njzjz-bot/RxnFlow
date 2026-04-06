import copy
import math
from collections import deque
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
from torch import Tensor

from ..config import Config
from ..envs import SynthesisEnvContext
from ..envs.action import RxnAction, RxnActionType
from ..envs.workflow import Protocol
from ..gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from ..utils.misc import get_global_rng
from .graph_transformer import GraphTransformer
from .layers import BlockEmbedding, SimilarityMDP
from .nn import MLP, Embedding, Linear

r"""
RxnFlow: Generative Flow Network for Synthesis Pathway Design

# Architecture Overview

The RxnFlow model extends TrajectoryBalanceModel with a graph transformer backbone and
hierarchical action space for chemical synthesis planning.

## Core Components:
- **Backbone**: GraphTransformer for state embedding
- **Hierarchical MDP**: Two-stage sampling (cluster → block) for scalable action spaces
- **Similarity-based scoring**: Uses molecular fingerprints and learned embeddings
- **Action masking**: Satisfies property constraints through action masking, i.e., budget constraints

# Trajectory Structure

Each synthesis trajectory follows a structured sequence:
1. **SetWorkflow**: Selects synthesis recipe (workflow) - action name: "init"
2. **FirstBlock**: Selects initial reactant building block
3. **Reaction steps**: Alternating UniRxn (unimolecular) and BiRxn (bimolecular) reactions

Action naming convention: `"{workflow_idx}-o{protocol_order}"` (except SetWorkflow)

## Action Types:
- **SetWorkflow**: Workflow selection using MLP classifier
- **FirstBlock**: Initial reactant selection (hierarchical: cluster → block)
- **BiRxn**: Bimolecular reaction with reactant selection (hierarchical: cluster → block)
- **UniRxn**: Unimolecular reaction (deterministic, no block selection)

# Hierarchical Action Sampling

For FirstBlock and BiRxn actions with large building block spaces:

## 1. Cluster Selection:
```python
cluster_embs = emb_cluster(cluster_fps)                    # [Ncluster, d_emb]
logits = similarity_mdp(state_emb, cluster_embs)           # [Ncluster]
logits += log(n_valid_blocks_per_cluster)                  # cluster size bonus
cluster_idx ~ softmax(logits / temperature)
```

## 2. Block Selection:
```python
block_features = get_block_features(block_type, cluster_idx)
block_embs = emb_block(block_features)                     # [Nblock, d_emb]
logits = similarity_mdp(state_emb, block_embs)             # [Nblock]
logits.masked_fill_(~action_mask, -inf)                    # action masking
block_idx ~ softmax(logits / temperature)
```

# Exploration Strategies

1. **Temperature-based sampling**: Controls exploration-exploitation tradeoff
2. **Epsilon-greedy**: Random action sampling with probability `random_action_prob`
3. **Novelty bonus**: Encourages exploration of less-visited actions using frequency-based bonuses
   - Tracks action history in sliding window memory
   - Applies configurable bonus functions: sqrt, log, linear, inverse

# Model Architecture Details

- **State embedding**: Embedding from GraphTransformer backbone
- **Cluster embedding**: Average fingerprint embeddings for clusters
- **Block embedding**: Multi-modal features (type, tier, fingerprint, properties)
- **Action embedding**: Embedding for similarity-based scoring
- **Learnable temperatures**: Per-action similarity MDP modules
- **Action masking**: Satisfies property constraints through action masking
"""


class RxnFlow(TrajectoryBalanceModel):
    """GraphTransformer class which outputs an RxnActionCategorical."""

    epsilon = 1e-10
    log_epsilon = math.log(epsilon)

    _deepcopy_exclude = {
        "ctx",
        "rng",
        "rng_np",
    }  # attributes to exclude from deepcopy (do shallow copy)

    def __init__(self, env_ctx: SynthesisEnvContext, cfg: Config) -> None:
        super().__init__()
        # Global context
        self.ctx: SynthesisEnvContext = env_ctx
        self.rng_np: np.random.RandomState = get_global_rng()

        # Parameters
        self.d_emb: int = cfg.model.emb_dim
        self.dropout: float = cfg.model.dropout
        self.act: str = cfg.model.activation

        # === Action names === #
        # SetWorkflow: 'init'
        # Other actions: '{workflow_idx}-o{protocol_order}'
        setworkflow_actions = ["init"]
        firstblock_actions = [
            self._get_action_name(widx, porder)
            for widx, workflow in enumerate(env_ctx.workflows)
            for porder in range(len(workflow.protocols))
            if workflow.protocols[porder].type is RxnActionType.FirstBlock
        ]
        birxn_actions = [
            self._get_action_name(widx, porder)
            for widx, workflow in enumerate(env_ctx.workflows)
            for porder in range(len(workflow.protocols))
            if workflow.protocols[porder].type is RxnActionType.BiRxn
        ]
        unirxn_actions = [
            self._get_action_name(widx, porder)
            for widx, workflow in enumerate(env_ctx.workflows)
            for porder in range(len(workflow.protocols))
            if workflow.protocols[porder].type is RxnActionType.UniRxn
        ]
        all_actions = (
            setworkflow_actions + firstblock_actions + birxn_actions + unirxn_actions
        )
        self.all_actions: list[str] = all_actions
        self.action_name_to_idx: dict[str, int] = {
            v: i for i, v in enumerate(all_actions)
        }

        # === State embedding === #
        # Action-name embedding
        self.action_name_emb = Embedding(len(all_actions), self.d_emb)

        # Backbone model
        self.backbone = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_graph_dim + self.d_emb,
            num_emb=self.d_emb,
            hidden_dim=cfg.model.graph_transformer.hidden_dim,
            num_layers=cfg.model.graph_transformer.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        # === SetWorkflow === #
        # simple classification MLP
        num_workflows = len(env_ctx.workflows)
        self.mdp_setworkflow = Linear(self.d_emb, num_workflows)

        # === FirstBlock & BiRxn === #
        # cluster embedding (no dropout)
        self.emb_cluster = MLP(env_ctx.block_fp_dim, self.d_emb, self.d_emb, 2, norm=True)
        # block embedding (no dropout)
        self.emb_block = BlockEmbedding(
            env_ctx.num_block_types,
            env_ctx.num_block_tiers,
            env_ctx.block_fp_dim,
            env_ctx.block_prop_dim,
            n_hid=self.d_emb,
            n_out=self.d_emb,
            n_layer=2,
        )
        # Select MDP class based on exploration configuration
        mdp_params = {
            "similarity_type": cfg.model.similarity_type,
            "init_temp": cfg.model.init_temperature,
            "min_temp": 0.01,
            "max_temp": 10.0,
        }
        self.mdp_cluster: nn.ModuleDict = nn.ModuleDict(
            {
                key: SimilarityMDP(**mdp_params)
                for key in firstblock_actions + birxn_actions
            }
        )
        self.mdp_block: nn.ModuleDict = nn.ModuleDict(
            {
                key: SimilarityMDP(**mdp_params)
                for key in firstblock_actions + birxn_actions
            }
        )

        # === UniRxn === #
        # unirxn is a dummy action, no model

        # === Trajectory Balance === #
        self._logZ = nn.Parameter(torch.tensor([cfg.model.init_logZ]))  # =Linear(1, 1)

        # === Exploration === #
        exploration_cfg = cfg.model.exploration
        self.use_novelty_bonus: bool = exploration_cfg.use_novelty_bonus
        if exploration_cfg.use_novelty_bonus:
            self.novelty_weight: float = exploration_cfg.novelty_weight
            self.novelty_memory_size: int = exploration_cfg.novelty_memory_size
            self.novelty_bonus_type: str = exploration_cfg.novelty_bonus_type
            assert self.novelty_bonus_type in ["sqrt", "log", "linear", "inverse"], (
                f"Invalid novelty bonus type: {self.novelty_bonus_type}"
            )
            # Create memory buffers for each action
            self.novelty_memories: dict[str, deque[int]] = {}
            self.novelty_counts: dict[str, np.ndarray] = {}

        # === Cache === #
        self._cache_workflow_weight = None

    def __deepcopy__(self, memo: dict) -> Self:
        """Deep copy the model, excluding the global context"""
        # Create new instance
        new_obj = type(self).__new__(self.__class__)

        # Deep copy the attributes excluding pre-defined ones
        for key, value in self.__dict__.items():
            if key in self._deepcopy_exclude:
                # If key is in the _deepcopy_exclude list, do a shallow copy
                setattr(new_obj, key, value)
            else:
                # Otherwise, do a deep copy
                setattr(new_obj, key, copy.deepcopy(value, memo))

        return new_obj

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return next(self.parameters()).device

    # === Hook methods === #
    def logZ(self) -> Tensor:
        return self._logZ.squeeze(-1)  # [Nstate,]

    def hook_workflow(
        self,
        state_emb: Tensor,
        random: bool = False,
    ) -> Tensor:
        """Calculate the logits for workflow selection.

        Parameters
        ----------
        state_emb : Tensor
            State embedding tensor.
            shape: [Fstate,]
        random : bool, optional
            If True, return uniform distribution logits instead of model-based
            logits. Default is False.

        Returns
        -------
        Tensor
            Logits for each workflow.
            shape: [Nworkflow,]
        """
        if self._cache_workflow_weight is None:
            with torch.no_grad():
                # get the workflow size
                workflow_size = torch.tensor(
                    [
                        self.ctx.workflow_chemical_space[workflow.id]
                        for workflow in self.ctx.workflows
                    ],
                    device=self.device,
                )
                # scale the workflow size to a log scale
                # TODO: what is better? sqrt, log, ... should I configurate this?
                scaled_workflow_size = (workflow_size + 1).log().clamp_(min=self.epsilon)
                # cache the workflow weight logits
                self._cache_workflow_weight = scaled_workflow_size.log()
        workflow_weight = self._cache_workflow_weight

        if random:
            # sample a random workflow
            assert not torch.is_grad_enabled(), (
                "Random sampling is not supported in training mode."
            )
            logits = workflow_weight  # [workflow,]
        else:
            # model forward hook
            logits = self.mdp_setworkflow(state_emb)  # [workflow,]
            # adjust the logits by workflow sizes
            logits = logits + workflow_weight  # [workflow,]
        return logits

    def hook_cluster(
        self,
        state_emb: Tensor,
        state_budget: Tensor | None,
        action_name: str,
        block_type: str,
        random: bool = False,
    ) -> Tensor:
        """Calculate the logits for cluster selection.

        This method computes logits for selecting clusters based on similarity
        between state embedding and cluster center embeddings.

        The logits are computed as:
        F(s,a) = similarity(s, a) + log(n_valid_actions)

        where s is the state embedding and a is the cluster center embedding.

        Parameters
        ----------
        state_emb : Tensor
            State embedding tensor.
            shape: [Fstate,]
        state_budget : Tensor
            Current budget state tensor used to determine valid actions.
            For the initial state, this can be None.
            shape: [Nbudget,]
        action_name : str
            Key identifying the protocol to use for the MDP cluster model.
        block_type : str
            Type of block being processed, used to retrieve cluster information
            and configurations.
        random : bool, optional
            If True, return uniform distribution logits instead of model-based
            logits. Default is False.

        Returns
        -------
        Tensor
            Logits for each cluster.
            shape: [Ncluster,]
        """
        # Compute the valid actions for each cluster
        n_clusters = self.ctx.num_block_clusters[block_type]
        action_budgets_list = [
            self.ctx.get_block_budgets(block_type, i, state_emb.device)
            for i in range(n_clusters)
        ]
        action_mask_list = [
            self._get_action_mask(state_budget, action_budgets)
            for action_budgets in action_budgets_list
        ]
        n_valid_actions = torch.stack(
            [action_mask.sum() for action_mask in action_mask_list]
        )  # [Ncluster,]
        cluster_mask = n_valid_actions > 0  # [Ncluster,]

        if random:
            # Sample a random cluster without considering the number of valid actions for exploration
            assert not torch.is_grad_enabled(), (
                "Random sampling is not supported in training mode."
            )
            logits = torch.where(cluster_mask, 0, -float("inf"))
        elif not cluster_mask.any():
            # If there is no allowed cluster, return a distribution proportional to plain cluster sizes
            n_actions = torch.tensor(
                [m.shape[0] for m in action_mask_list], device=state_emb.device
            )  # [Ncluster,]
            logits = n_actions.log()  # [Ncluster,]
        else:
            # Model forward hook
            clusters = self.ctx.get_cluster_fps(block_type, state_emb.device)
            cluster_embs = self.emb_cluster(clusters)  # [Ncluster, Faction]
            logits = self.mdp_cluster[action_name](state_emb, cluster_embs)  # [Ncluster,]

            # Weight the logits by the number of valid actions in each cluster
            # NOTE: This returns -inf where n_valid_actions is 0, but this is a intended design.
            logits = logits + n_valid_actions.log()  # [Ncluster,]
        return logits

    def hook_block(
        self,
        state_emb: Tensor,
        state_budget: Tensor | None,
        action_name: str,
        block_type: str,
        cluster_idx: int,
        random: bool = False,
    ) -> Tensor:
        """Calculate the logits for block selection.

        Parameters
        ----------
        state_emb : Tensor
            State embedding tensor.
            shape: [Fstate,]
        state_budget : Tensor
            Current budget state tensor used to determine valid actions.
            For the initial state, this can be None.
            shape: [Nbudget,]
        action_name : str
            Key identifying the protocol to use for the MDP cluster model.
        block_type : str
            Type of block being processed, used to retrieve cluster information
            and configurations.
        cluster_idx : int
            Index of the cluster from which to select a block.
        random : bool, optional
            If True, return uniform distribution logits instead of model-based
            logits. Default is False.

        Returns
        -------
        Tensor
            Logits for each building block.
            shape: [Nblock,]
        """
        # compute the action mask for the selected cluster
        action_budgets = self.ctx.get_block_budgets(
            block_type, cluster_idx, state_emb.device
        )  # [Naction, Nbudget]
        action_mask = self._get_action_mask(state_budget, action_budgets)  # [Naction,]

        if random:
            # sample a random block
            assert not torch.is_grad_enabled(), (
                "Random sampling is not supported in training mode."
            )
            logits = torch.where(action_mask, 0, -float("inf"))
        elif not action_mask.any():
            # if there is no allowed action, return a uniform distribution over all blocks
            num_blocks = self.ctx.num_blocks[block_type][cluster_idx]
            logits = torch.zeros((num_blocks,), device=state_emb.device)
        else:
            # model forward hook
            blocks = self.ctx.get_block_features(
                block_type, cluster_idx, state_emb.device
            )
            block_embs = self.emb_block(blocks)  # [Nblock, Faction]
            logits = self.mdp_block[action_name](state_emb, block_embs)  # [Naction,]
            # mask unallowed actions
            logits.masked_fill_(~action_mask, -float("inf"))
        return logits

    # === Sampling methods === #
    @torch.no_grad()
    def sample(
        self,
        g: gd.Batch,
        random_action_prob: float = 0.0,
        temperature: float = 1.0,
    ) -> list[RxnAction]:
        """Sample actions from the model given a batch of states.
        Parameters
        ----------
        g : gd.Batch
            A batch of states, containing graph attributes, action types, workflow indices, and protocol orders.
        random_action_prob : float, optional
            Probability of sampling a random action instead of a model-based action. (epsilon-greedy sampling)
            Default is 0.0 (no random sampling).
        temperature : float, optional
            Temperature for sampling from the logits.
            Higher values lead to more exploration, while lower values lead to more exploitation.
        Returns
        -------
        list[RxnAction]
            A list of sampled actions, each represented as a RxnAction object.
        """

        # get information from the batch
        num_graphs: int = g.num_graphs
        state_budgets: list[Tensor] = g.budget
        action_types: list[RxnActionType] = [
            self.ctx.action_type_order[t] for t in g.action_type
        ]
        workflow_idxs: list[int] = g.workflow.tolist()
        protocol_orders: list[int] = g.protocol_order.tolist()

        # === Get state embeddings === #
        # get the names of MDP actions
        action_names = [
            self._get_action_name(widx, order)
            for widx, order in zip(workflow_idxs, protocol_orders, strict=True)
        ]
        action_name_idx = torch.tensor(
            [self.action_name_to_idx[name] for name in action_names],
            device=self.device,
            dtype=torch.int32,
        )  # [Nstate,]
        action_name_emb = self.action_name_emb(action_name_idx)  # [Nstate, Femb]
        global_feats = torch.cat(
            [g.graph_attr, action_name_emb], dim=-1
        )  # [Nstate, Fgraph + Femb]
        # state embedding
        state_embs = self.backbone(g, global_feats)

        # === Sampling === #
        actions: list[RxnAction] = []
        for i in range(num_graphs):
            action_type = action_types[i]
            workflow_idx = workflow_idxs[i]
            protocol_order = protocol_orders[i]
            match action_type:
                case RxnActionType.SetWorkflow:
                    assert workflow_idx == -1, (
                        "workflow_idx should be -1 for SetWorkflow action"
                    )
                    assert protocol_order == -1, (
                        "protocol_order should be -1 for SetWorkflow action"
                    )
                    cluster_idx = -1  # SetWorkflow does not have cluster_idx
                    action_idx = self._sample_setworkflow(
                        state_emb=state_embs[i],
                        action_name=action_names[i],
                        temperature=temperature,
                        random_prob=random_action_prob,
                    )
                case RxnActionType.FirstBlock:
                    assert workflow_idx >= 0, (
                        "workflow_idx should be non-negative for FirstBlock action"
                    )
                    assert protocol_order == 0, (
                        "protocol_order should be 0 for FirstBlock action"
                    )
                    cluster_idx, action_idx = self._sample_firstblock(
                        state_emb=state_embs[i],
                        action_name=action_names[i],
                        temperature=temperature,
                        random_prob=random_action_prob,
                    )
                case RxnActionType.BiRxn:
                    assert workflow_idx >= 0, (
                        "workflow_idx should be non-negative for BiRxn action"
                    )
                    assert protocol_order >= 0, (
                        "protocol_order should be greater than 0 for BiRxn action"
                    )
                    cluster_idx, action_idx = self._sample_birxn(
                        state_emb=state_embs[i],
                        state_budget=state_budgets[i],
                        action_name=action_names[i],
                        temperature=temperature,
                        random_prob=random_action_prob,
                    )
                case RxnActionType.UniRxn:
                    assert workflow_idx >= 0, (
                        "workflow_idx should be non-negative for UniRxn action"
                    )
                    assert protocol_order >= 0, (
                        "protocol_order should be greater than 0 for UniRxn action"
                    )
                    cluster_idx = (
                        action_idx
                    ) = -1  # UniRxn is dummy action (only a single action available)
                case _:
                    raise ValueError(f"Invalid action type: {action_type}")
            action = self.ctx.get_action(
                action_type, workflow_idx, protocol_order, cluster_idx, action_idx
            )
            actions.append(action)
        return actions

    def _sample_setworkflow(
        self,
        state_emb: Tensor,
        action_name: str,
        temperature: float = 1.0,
        random_prob: float = 0.0,
    ) -> int:
        """The sampling function to be called for workflow selection."""
        assert action_name == "init", f"action_name must be 'init', got {action_name}"
        is_random_workflow = self._should_sample_random(random_prob)
        # calculate the logits for workflow selection
        logits = self.hook_workflow(state_emb, random=is_random_workflow)
        # sample a workflow from the logits
        return self.sample_from_logits(logits, temperature, action_name)

    def _sample_firstblock(
        self,
        state_emb: Tensor,
        action_name: str,
        temperature: float = 1.0,
        random_prob: float = 0.0,
    ) -> tuple[int, int]:
        """The sampling function to be called for firstblock selection."""
        # Get block type
        protocol = self._get_protocol(action_name)
        assert protocol.type is RxnActionType.FirstBlock, (
            f"action_type must be FirstBlock, got {protocol.type}"
        )
        block_type = protocol.block_type

        # Initial state (empty) has zero-budget
        state_budget = None

        # Scale random probability for hierarchical sampling
        # Ensures overall random probability matches the parameter
        # For 2 levels: p' = 1 - sqrt(1 - p) gives P(at least one random) = p
        random_prob = 1 - math.sqrt(1 - random_prob)

        # Sample cluster
        is_random_cluster = self._should_sample_random(random_prob)
        logits_cluster = self.hook_cluster(
            state_emb, state_budget, action_name, block_type, random=is_random_cluster
        )
        cluster_idx = self.sample_from_logits(
            logits_cluster, temperature, action_name + "-cluster"
        )

        # Sample block
        is_random_block = self._should_sample_random(random_prob)
        logits_block = self.hook_block(
            state_emb,
            state_budget,
            action_name,
            block_type,
            cluster_idx,
            random=is_random_block,
        )
        block_idx = self.sample_from_logits(
            logits_block, temperature, action_name + f"-block{cluster_idx}"
        )

        return cluster_idx, block_idx

    def _sample_birxn(
        self,
        state_emb: Tensor,
        state_budget: Tensor,
        action_name: str,
        temperature: float = 1.0,
        random_prob: float = 0.0,
    ) -> tuple[int, int]:
        """The sampling function to be called for birxn selection."""
        # Get block type
        protocol = self._get_protocol(action_name)
        assert protocol.type is RxnActionType.BiRxn, (
            f"action_type must be BiRxn, got {protocol.type}"
        )
        block_type = protocol.block_type

        # Scale random probability for hierarchical sampling
        # Ensures overall random probability matches the parameter
        # For 2 levels: p' = 1 - sqrt(1 - p) gives P(at least one random) = p
        random_prob = 1 - math.sqrt(1 - random_prob)

        # Sample cluster
        do_random_cluster = self._should_sample_random(random_prob)
        logits_cluster = self.hook_cluster(
            state_emb, state_budget, action_name, block_type, random=do_random_cluster
        )
        cluster_idx = self.sample_from_logits(
            logits_cluster, temperature, action_name + "-cluster"
        )

        # Sample block
        do_random_block = self._should_sample_random(random_prob)
        logits_block = self.hook_block(
            state_emb,
            state_budget,
            action_name,
            block_type,
            cluster_idx,
            random=do_random_block,
        )
        block_idx = self.sample_from_logits(
            logits_block, temperature, action_name + f"-block{cluster_idx}"
        )

        return cluster_idx, block_idx

    # === LogP Calculation === #
    def compute_log_prob(self, g: gd.Batch, actions: list[RxnAction]) -> Tensor:
        # get information from the batch
        num_graphs: int = g.num_graphs
        state_budgets: list[Tensor] = g.budget
        action_types: list[RxnActionType] = [
            self.ctx.action_type_order[t] for t in g.action_type
        ]
        workflow_idxs: list[int] = g.workflow.tolist()
        protocol_orders: list[int] = g.protocol_order.tolist()

        # === Get state embeddings === #
        # get the names of MDP actions
        action_names = [
            self._get_action_name(widx, order)
            for widx, order in zip(workflow_idxs, protocol_orders, strict=True)
        ]
        action_name_idx = torch.tensor(
            [self.action_name_to_idx[name] for name in action_names],
            device=self.device,
            dtype=torch.int32,
        )  # [Nstate,]
        action_name_emb = self.action_name_emb(action_name_idx)  # [Nstate, Femb]
        global_feats = torch.cat(
            [g.graph_attr, action_name_emb], dim=-1
        )  # [Nstate, Fgraph + Femb]
        # state embedding
        state_embs = self.backbone(g, global_feats)

        # === LogP calculation === #
        action_logprobs: list[Tensor] = []
        for i in range(num_graphs):
            action = actions[i]
            action_type = action_types[i]

            match action_type:
                case RxnActionType.SetWorkflow:
                    logp = self._compute_log_prob_setworkflow(
                        state_emb=state_embs[i],
                        action=action,
                    )
                case RxnActionType.FirstBlock:
                    logp = self._compute_log_prob_firstblock(
                        state_emb=state_embs[i],
                        action=action,
                    )
                case RxnActionType.BiRxn:
                    logp = self._compute_log_prob_birxn(
                        state_emb=state_embs[i],
                        state_budget=state_budgets[i],
                        action=action,
                    )
                case RxnActionType.UniRxn:
                    logp = self._compute_log_prob_unirxn(
                        state_emb=state_embs[i],
                        state_budget=state_budgets[i],
                        action=action,
                    )
                case _:
                    raise ValueError(f"Invalid action type: {action_types[i]}")
            action_logprobs.append(logp)
        logprobs = torch.stack(action_logprobs, dim=0)  # [Nstate,]
        return logprobs

    def _compute_log_prob_setworkflow(
        self,
        state_emb: Tensor,
        action: RxnAction,
    ) -> Tensor:
        """The log-probability of the action for SetWorkflow selection."""
        workflow_idx = self.ctx.workflow_to_idx[action.workflow]
        logits = self.hook_workflow(state_emb)
        return self.log_softmax(logits)[workflow_idx]

    def _compute_log_prob_firstblock(
        self,
        state_emb: Tensor,
        action: RxnAction,
    ) -> Tensor:
        """The log-probability of the action for FirstBlock selection."""
        workflow_idx = self.ctx.workflow_to_idx[action.workflow]
        action_name = self._get_action_name(workflow_idx, action.protocol_order)
        protocol = self._get_protocol(action_name)
        assert protocol.type is RxnActionType.FirstBlock, (
            f"action_type must be FirstBlock, got {protocol.type}"
        )
        block_type = protocol.block_type

        state_budget = None  # initial state does not have budget

        # calculate the log probability of the selected cluster
        logits_cluster = self.hook_cluster(
            state_emb, state_budget, action_name, block_type
        )
        logp_cluster = self.log_softmax(logits_cluster)[action.block_cluster_idx]

        # calculate the log probability of the selected block
        logits_block = self.hook_block(
            state_emb, state_budget, action_name, block_type, action.block_cluster_idx
        )
        logp_block = self.log_softmax(logits_block)[action.block_idx]

        return logp_cluster + logp_block

    def _compute_log_prob_birxn(
        self,
        state_emb: Tensor,
        state_budget: Tensor,
        action: RxnAction,
    ) -> Tensor:
        """The log-probability of the action for BiRxn selection."""
        workflow_idx = self.ctx.workflow_to_idx[action.workflow]
        action_name = self._get_action_name(workflow_idx, action.protocol_order)
        protocol = self._get_protocol(action_name)
        assert protocol.type is RxnActionType.BiRxn, (
            f"action_type must be BiRxn, got {protocol.type}"
        )
        block_type = protocol.block_type

        # calculate the log probability of the selected cluster
        logits_cluster = self.hook_cluster(
            state_emb, state_budget, action_name, block_type
        )
        logp_cluster = self.log_softmax(logits_cluster)[action.block_cluster_idx]

        # calculate the log probability of the selected block
        logits_block = self.hook_block(
            state_emb, state_budget, action_name, block_type, action.block_cluster_idx
        )
        logp_block = self.log_softmax(logits_block)[action.block_idx]

        return logp_cluster + logp_block

    def _compute_log_prob_unirxn(
        self,
        state_emb: Tensor,
        state_budget: Tensor,
        action: RxnAction,
    ) -> Tensor:
        """The log-probability of the action for UniRxn selection."""
        return torch.tensor(0.0, device=state_emb.device)  # only one action.

    # === Exploration Strategies === #
    def _get_novelty_bonus(self, logits: Tensor, action_name: str) -> Tensor:
        """Calculate novelty bonus based on action frequency to encourage exploration.

        Parameters
        ----------
        logits : Tensor
            Action logits tensor.
            shape: [Naction,]
        action_name : str
            Action name representing the current action context for tracking.

        Returns
        -------
        Tensor
            Novelty bonus values per action.
            shape: [Naction,]
        """
        # If action name is not in novelty memory, initialize it
        if action_name not in self.novelty_memories:
            n_actions = logits.shape[0]
            self.novelty_memories[action_name] = deque(maxlen=self.novelty_memory_size)
            self.novelty_counts[action_name] = np.zeros(n_actions, dtype=np.int32)

        # Calculate novelty bonus for each possible action
        frequencies = torch.as_tensor(
            self.novelty_counts[action_name], dtype=logits.dtype, device=logits.device
        )

        # Apply different novelty bonus functions based on configuration
        match self.novelty_bonus_type:
            case "sqrt":
                novelty_bonuses = torch.sqrt(1 / (1 + frequencies))
            case "log":
                novelty_bonuses = torch.log(1 + 1 / (1 + frequencies))
            case "linear":
                novelty_bonuses = 1 / (1 + frequencies)
            case "inverse":
                novelty_bonuses = 1 / (1 + frequencies**2)
            case _:
                raise ValueError(f"Invalid novelty bonus type: {self.novelty_bonus_type}")

        return novelty_bonuses

    def _update_novelty_memory(self, action_name: str, selected_action: int) -> None:
        """Update novelty memory with the selected action.

        Parameters
        ----------
        action_name : str
            Action name representing the current action context.
        selected_action : int
            Index of the selected action.
        """
        memory = self.novelty_memories[action_name]
        cnt = self.novelty_counts[action_name]

        # If the memory is full, remove the oldest action and updates its count
        # deque automatically removes the oldest element, so only update the count
        if len(memory) == self.novelty_memory_size:
            old_action = memory[0]  # Will be removed when we append
            cnt[old_action] -= 1
            assert cnt[old_action] >= 0, "Action count cannot be negative"

        # Add new action
        memory.append(selected_action)
        cnt[selected_action] += 1

    # === Helper methods === #
    def sample_from_logits(
        self, logits: Tensor, temperature: float = 1.0, action_name: str = ""
    ) -> int:
        """Sample an action index from the logits with exploration strategies.

        Parameters
        ----------
        logits : Tensor
            Action logits tensor.
            shape: [Naction,]
        temperature : float
            Sampling temperature.
        action_name : str
            Action name for novelty tracking.

        Returns
        -------
        int
            Selected action index.
        """
        # assert action_name in self.all_actions, f"Invalid action name: {action_name}"

        # if there is only one action, return index=0
        if logits.numel() == 1:
            return 0

        # if all logits are -inf (no valid actions), randomly sample without frequency tracking
        if logits.isinf().all():
            selected_idx = self.rng_np.randint(logits.shape[0])
            return selected_idx

        # Apply novelty bonus (adds per-action bonuses)
        if self.training and self.use_novelty_bonus:
            novelty_bonuses = self._get_novelty_bonus(logits, action_name)
            logits = logits + novelty_bonuses * self.novelty_weight

        # Sample from modified logits
        probs = F.softmax(logits / temperature, dim=-1)
        selected_idx = int(torch.multinomial(probs, 1, generator=self.rng).item())

        # Update novelty memory after selection
        if self.training and self.use_novelty_bonus:
            self._update_novelty_memory(action_name, selected_idx)

        return selected_idx

    def log_softmax(self, logits: Tensor) -> Tensor:
        """'Compute the log-softmax of the logits."""
        return logits.log_softmax(dim=-1)

    @staticmethod
    def _get_action_name(workflow_idx: int, protocol_order: int) -> str:
        if workflow_idx == -1:
            assert protocol_order == -1, (
                "protocol_order must be -1 for SetWorkflow action"
            )
            return "init"
        else:
            assert protocol_order >= 0, (
                "protocol_order must be non-negative for FirstBlock, BiRxn, and UniRxn actions"
            )
            return f"{workflow_idx}-o{protocol_order}"

    def _get_protocol(self, action_name: str) -> Protocol:
        """Get the protocol from the action name."""
        assert action_name != "init", "Action name 'init' is not a valid protocol action"
        workflow_idx, protocol_order = map(int, action_name.split("-o"))
        workflow = self.ctx.get_workflow(workflow_idx)
        return workflow[protocol_order]

    @torch.no_grad()
    def _should_sample_random(self, random_prob: float) -> bool:
        """epsilon-greedy sampling: decide whether to sample a random action."""
        return self.rng_np.random() < random_prob

    @staticmethod
    @torch.no_grad()
    def _get_action_mask(
        state_budget: Tensor | None,
        action_budgets: Tensor,
        eps: float = 1e-2,
    ) -> Tensor:
        """Mask of the action for single state
        if budget(state) + budget(action) > 1, the action is masked.
        Use small epsilon to avoid floating point numerical issues.

        Parameters
        ----------
        state_budget : Tensor | None [Nprop]
            budget of state
        action_budgets : Tensor [Naction, Nprop]
            budget of actions

        Returns
        -------
        action_mask: Tensor [Naction]
            mask of the action (action)
        """
        if state_budget is None:
            # if state_budget is None, we assume that the state budget is zero
            next_state_budget = action_budgets
        else:
            # estimate the next state budget
            next_state_budget = state_budget.unsqueeze(0) + action_budgets
        return (next_state_budget < (1 + eps)).all(-1)  # avoid numerical issues
