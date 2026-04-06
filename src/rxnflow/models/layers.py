import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..envs.env_context import BlockTensor
from .nn import MLP, Embedding, LayerNorm, Linear


class BlockEmbedding(nn.Module):
    """Embeds a heterogeneous block tensor into a single fixed-size vector.

    This module processes four types of features associated with a block:
    1. A single categorical type ID.
    2. A batch of categorical tier IDs.
    3. A batch of continuous fingerprint vectors.
    4. A batch of continuous property vectors.

    Each feature is projected into a hidden dimension, concatenated, and then
    processed by an MLP to produce the final embedding.

    Parameters
    ----------
    n_type : int
        Number of unique block types (vocabulary size for type embedding).
    n_tier : int
        Number of unique block tiers (vocabulary size for tier embedding).
    fp_dim : int
        Dimension of the fingerprint vector.
    prop_dim : int
        Dimension of the property vector.
    n_hid : int
        The hidden dimension for internal projections and the MLP.
    n_out : int
        The dimension of the final output embedding.
    n_layer : int
        Number of hidden layers in the final MLP.
    """

    def __init__(
        self,
        n_type: int,
        n_tier: int,
        fp_dim: int,
        prop_dim: int,
        n_hid: int,
        n_out: int,
        n_layer: int,
    ):
        super().__init__()
        self.emb_type = Embedding(n_type, n_hid)
        self.emb_tier = Embedding(n_tier, n_hid)
        self.lin_fp = nn.Sequential(Linear(fp_dim, n_hid), LayerNorm(n_hid))
        self.lin_prop = nn.Sequential(Linear(prop_dim, n_hid), LayerNorm(n_hid))
        self.mlp = MLP(4 * n_hid, n_hid, n_out, n_layer, "relu", norm=True)

    def forward(self, block: BlockTensor) -> Tensor:
        """Embeds a block tensor into a fixed-size embedding.

        Parameters
        ----------
        block : BlockTensor
            A data structure containing the block features. Expected attributes are:
            - type: A single integer representing the block type.
            - tier: A 1D tensor of tier IDs for each block in the batch of shape (n_blocks,)
            - fp: A 2D tensor of fingerprint vectors of shape (n_blocks, fp_dim)
            - property: A 2D tensor of property vectors of shape (n_blocks, prop_dim)

        Returns
        -------
        Tensor
            The final block embedding tensor of shape (n_blocks, n_out).
        """
        # Project continuous features
        x_fp = self.lin_fp(block.fp)
        x_prop = self.lin_prop(block.property)
        # Embed categorical features
        x_tier = self.emb_tier(block.tier)
        # Handle the single shared type embedding and expand it to the batch size.
        x_type = self.emb_type.weight[block.type].view(1, -1).expand_as(x_fp)
        # Concatenate all features
        x = torch.cat([x_fp, x_prop, x_tier, x_type], dim=-1)
        return self.mlp(x)


class SimilarityMDP(nn.Module):
    """Base MDP layer that supports both cosine and dot product similarity."""

    def __init__(
        self,
        init_temp: float = 1.0,
        min_temp: float = 0.01,
        max_temp: float = 10.0,
        similarity_type: str = "cosine",
    ):
        super().__init__()
        assert similarity_type in ["cosine", "dot"], (
            f"similarity_type must be 'cosine' or 'dot', got {similarity_type}"
        )
        assert min_temp <= init_temp <= max_temp, (
            f"init_temp must be between [{min_temp}, {max_temp}], got {init_temp}, {min_temp}, {max_temp}"
        )

        self.min_temp: float = min_temp
        self.max_temp: float = max_temp
        self.similarity_type: str = similarity_type

        # convert to logit temp
        self.temp_range = max_temp - min_temp
        normalized = (init_temp - min_temp) / self.temp_range
        self._logit_temp = nn.Parameter(torch.logit(torch.tensor(normalized)))

    @property
    def temperature(self) -> torch.Tensor:
        normalized = torch.sigmoid(self._logit_temp)
        return self.min_temp + normalized * self.temp_range

    def compute_similarity(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Computes similarity between a single state embedding and action embeddings.
        Parameters
        ----------
        state : torch.Tensor
            State embedding tensor.
            shape: (embed_dim,)
        action : torch.Tensor
            Action embedding tensor.
            shape: (num_actions, embed_dim)
        Returns
        -------
        torch.Tensor
            Similarity scores for each action.
            shape: (num_actions,)

        Notes
        -----
        Here, we always perform normalization for action embeddings.
        This prevents that some of actions have very small embeddings, which
        would lead to zero-probability regardless of the state embedding.
        """

        # Compute similarity based on type
        if self.similarity_type == "cosine":
            state = F.normalize(state, dim=-1)
            actions = F.normalize(actions, dim=-1)
            similarity = (
                actions @ state
            )  # (num_actions, embed_dim) @ (embed_dim,) -> (num_actions,)
        elif self.similarity_type == "dot":  # dot product
            actions = F.normalize(actions, dim=-1)
            similarity = (
                actions @ state
            )  # (num_actions, embed_dim) @ (embed_dim,) -> (num_actions,)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        return similarity

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Computes similarity between state and action embeddings.

        Parameters
        ----------
        state : torch.Tensor
            State embedding tensor.
            shape: (embed_dim,)
        action : torch.Tensor
            Action embedding tensor.
            shape: (num_actions, embed_dim)

        Returns
        -------
        torch.Tensor
            Logits for action selection.
        """
        similarity = self.compute_similarity(state, action)
        return similarity / self.temperature
