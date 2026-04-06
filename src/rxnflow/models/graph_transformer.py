import inspect
from typing import Any

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from loguru import logger
from torch_geometric.utils import add_self_loops

from .nn import MLP, LayerNorm, Linear


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes

    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.

    The per node outputs are the final (post graph-convolution) node embeddings.

    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    node embeddings, and of the final virtual node embeddings.
    """

    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        g_dim: int,
        num_emb: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
        ln_type: str = "pre",
    ):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features
        num_emb: int
            The number of embedding dimensions, i.e. embedding size. Default 64.
        hidden_dim: int
            The number of hidden dimensions. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        ln_type: str
            The location of Layer Norm in the transformer, either 'pre' or 'post', default 'pre'.
            (apparently, before is better than after, see https://arxiv.org/pdf/2002.04745.pdf)
        """
        super().__init__()
        self.num_layers: int = num_layers
        assert ln_type in ["pre", "post"]
        self.ln_type: str = ln_type

        self.x2h = MLP(x_dim, hidden_dim, hidden_dim, 2)
        self.e2h = MLP(e_dim, hidden_dim, hidden_dim, 2)
        self.c2h = MLP(max(1, g_dim), hidden_dim, hidden_dim, 2)
        n_att = hidden_dim * num_heads
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(
                            hidden_dim,
                            hidden_dim,
                            **self._gen_conv_kwargs(num_layers=1, aggr="add", norm=None),
                        ),
                        gnn.TransformerConv(
                            hidden_dim * 2,
                            n_att // num_heads,
                            edge_dim=hidden_dim,
                            heads=num_heads,
                        ),
                        gnn.Linear(n_att, hidden_dim),
                        gnn.LayerNorm(hidden_dim, affine=False),
                        MLP(hidden_dim, hidden_dim * 4, hidden_dim, 1),
                        gnn.LayerNorm(hidden_dim, affine=False),
                        gnn.Linear(hidden_dim, hidden_dim * 2),
                    ]
                    for _ in range(self.num_layers)
                ],
                [],
            )
        )
        self.proj = Linear(hidden_dim * 2, num_emb)
        self.norm = LayerNorm(num_emb)

    def forward(self, g: gd.Batch, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        g: gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        emb: torch.Tensor
            Graph embeddings. Shape: (g.num_graphs, self.num_emb).
        """
        o = self.x2h(g.x)  # [V, h]
        e = self.e2h(g.edge_attr)  # [E, h]
        c = self.c2h(cond)  # [G, h]

        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        num_total_nodes = g.x.shape[0]
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat(
            [g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1
        )
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)  # [V + G, h]

        # Run the graph transformer forward
        for i in range(self.num_layers):
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[
                i * 7 : (i + 1) * 7
            ]
            cs = cscale(c[aug_batch])
            if self.ln_type == "post":
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        o_final = o[: -c.shape[0]]  # [V, h]
        c_final = o[-c.shape[0] :]  # [G, h]

        # Global mean pooling of the final node embeddings
        g_final = gnn.global_mean_pool(o_final, g.batch)  # [G, h]
        glob = self.proj(torch.cat([g_final, c_final], 1))  # [G, num_emb]
        glob_norm = self.norm(glob)  # [G, num_emb]
        return glob_norm

    @staticmethod
    def _gen_conv_kwargs(**kwargs) -> dict[str, Any]:
        """Handle `torch_geometric` backward compatibility.

        torch_geometric >= 2.2: kwarg `bias` was added and defaulted to False,
        but it was implicitly True in older versions.
        """
        bias_kwarg = "bias"
        gen_conv_sig = inspect.signature(gnn.GENConv.__init__)
        gen_conv_has_bias = bias_kwarg in gen_conv_sig.parameters
        gen_conv_kwargs = dict(**kwargs)
        if gen_conv_has_bias and bias_kwarg not in kwargs:
            logger.debug("Setting bias=True for GENConv (backward compatibility)")
            gen_conv_kwargs[bias_kwarg] = True
        return gen_conv_kwargs
