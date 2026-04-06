from typing import Any

import torch
import torch.nn as nn


# ===============================
# Weight Initialization Functions
# ===============================
def init_linear_weight(m: nn.Linear, activation: nn.Module | None = None) -> None:
    """Initializes the weights of a linear layer based on the activation function.

    This function applies Kaiming (He) uniform initialization for layers followed
    by ReLU-like activations and Xavier (Glorot) uniform initialization for others.

    Parameters
    ----------
    m : nn.Linear
        The linear layer to initialize.
    activation : nn.Module, optional
        The activation function instance that follows the linear layer. The
        initialization strategy is chosen based on this activation. If None,
        Xavier initialization is used.
    """
    if not isinstance(m, nn.Linear):
        raise TypeError(f"Expected nn.Linear, but got {type(m).__name__}")

    if isinstance(activation, nn.ReLU | nn.GELU | nn.SiLU | nn.ELU):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    elif isinstance(activation, nn.LeakyReLU):
        # Use the actual negative_slope from the activation function instance
        nn.init.kaiming_uniform_(
            m.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
            a=activation.negative_slope,
        )
    else:  # For Tanh, Sigmoid, or no activation
        nn.init.xavier_uniform_(m.weight)

    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_embedding_weight(m: nn.Embedding) -> None:
    """Initializes the weights of an embedding layer.

    Weights are initialized from a uniform distribution U(-0.1, 0.1). If a
    `padding_idx` is specified, the corresponding embedding vector is
    initialized to all zeros.

    Parameters
    ----------
    m : nn.Embedding
        The embedding layer to initialize.
    """
    if not isinstance(m, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding, but got {type(m).__name__}")

    nn.init.uniform_(m.weight, -0.1, 0.1)
    if m.padding_idx is not None:
        with torch.no_grad():
            m.weight[m.padding_idx].fill_(0)


def init_layernorm_weight(m: nn.LayerNorm) -> None:
    """Initializes the weights of a LayerNorm layer.

    The affine transformation weights (`gamma`) are initialized to ones, and biases
    (`beta`) are initialized to zeros.

    Parameters
    ----------
    m : nn.LayerNorm
        The LayerNorm layer to initialize.
    """
    if not isinstance(m, nn.LayerNorm):
        raise TypeError(f"Expected nn.LayerNorm, but got {type(m).__name__}")

    if m.elementwise_affine:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ===============================
# Custom Layer Classes
# ===============================


class Linear(nn.Linear):
    """A linear layer with built-in weight initialization.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If set to ``False``, the layer will not learn an additive bias.
    activation : nn.Module, optional
        The activation function instance that will follow this layer. This is
        used to determine the best weight initialization strategy. If `None`,
        a general-purpose initialization (Xavier) is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: nn.Module | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        init_linear_weight(self, activation=activation)


class LayerNorm(nn.LayerNorm):
    """A LayerNorm layer with built-in weight initialization.

    Parameters
    ----------
    normalized_shape : int or list or torch.Size
        Input shape from an expected input of size.
    eps : float, default=1e-5
        A value added to the denominator for numerical stability.
    elementwise_affine : bool, default=True
        A boolean value that when set to ``True``, this module has learnable
        affine parameters.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )
        init_layernorm_weight(self)


class Embedding(nn.Embedding):
    """An embedding layer with built-in weight initialization.

    Parameters
    ----------
    num_embeddings : int
        Size of the dictionary of embeddings.
    embedding_dim : int
        The size of each embedding vector.
    padding_idx : int, optional
        If specified, the entries at `padding_idx` do not contribute to the
        gradient; therefore, the embedding vector at `padding_idx` is not
        updated during training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        init_embedding_weight(self)


# ===============================
# Helper to get Activation Class
# ===============================


def get_activation_cls(activation: str) -> type[nn.Module]:
    """Returns the activation function class based on the string name."""
    act_map = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    act_lower = activation.lower()
    return act_map[act_lower]


# ===============================
# Composite Module: MLP
# ===============================


class MLP(nn.Sequential):
    """A multi-layer perceptron (MLP) with flexible configuration.

    This module creates a sequence of layers:
    Linear -> Activation -> (LayerNorm) -> (Dropout)

    The final layer is a linear transformation without a subsequent activation
    function.

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_hid : int
        Number of features in the hidden layers.
    n_out : int
        Number of output features.
    n_layer : int, default=0
        Number of hidden layers. If 0, the MLP is just a single linear layer.
    activation : str, default="leakyrelu"
        The name of the activation function to use in hidden layers.
        Supported: "relu", "leakyrelu", "elu", "gelu", "silu", "tanh", "sigmoid".
    act_kwargs : dict, optional
        Keyword arguments to pass to the activation function's constructor.
        Example: `{'negative_slope': 0.2}` for LeakyReLU.
    norm : bool, default=False
        If ``True``, adds a `LayerNorm` layer after the activation in each
        hidden block.
    dropout : float, default=0.0
        If non-zero, adds a `Dropout` layer after the activation/norm in each
        hidden block.
    """

    def __init__(
        self,
        n_in: int,
        n_hid: int,
        n_out: int,
        n_layer: int = 0,
        activation: str = "leakyrelu",
        act_kwargs: dict[str, Any] | None = None,
        norm: bool = False,
        dropout: float = 0.0,
    ):
        if n_layer < 0:
            raise ValueError("Number of hidden layers (n_layer) must be non-negative.")

        act_cls = get_activation_cls(activation)
        act_fn = act_cls(**(act_kwargs or {}))

        layers = []
        d_in = n_in
        for _ in range(n_layer):
            layers.append(Linear(d_in, n_hid, activation=act_fn))
            layers.append(act_fn)
            if norm:
                layers.append(LayerNorm(n_hid))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = n_hid

        # The final layer has no activation, so we pass init_act=None
        # to use the default Xavier initialization.
        layers.append(Linear(d_in, n_out))

        super().__init__(*layers)
