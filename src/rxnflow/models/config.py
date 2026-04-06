from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphTransformerConfig:
    hidden_dim: int = 128
    num_heads: int = 2
    ln_type: str = "pre"
    num_layers: int = 4


@dataclass(slots=True)
class ExplorationConfig:
    """Configuration for exploration strategies

    Attributes
    ----------
    use_novelty_bonus : bool
        Whether to use a novelty bonus during on-policy training
    novelty_weight : float
        Weight for novelty bonus term
    novelty_memory_size : int
        Size of memory buffer for tracking action frequencies
        Large value does not guarantee better performance, since
        long-term novelty is not always beneficial.
    novelty_bonus_type : str
        Type of novelty bonus function: 'sqrt', 'log', 'linear', 'inverse'

    Notes
    -----
    If we use batch size of 64 with random workflow (90 workflows), then
    novelty-memory-size of 100 is corresponds to 100 * 90 / 64 = 140 steps

    """

    use_novelty_bonus: bool = True
    novelty_weight: float = 1.0
    novelty_memory_size: int = 1000
    novelty_bonus_type: str = "linear"


@dataclass(slots=True)
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    emb_dim : int
        The number of dimensions of the model
    dropout : float
        The dropout rate to use in the model (increase stochasticity)
    activation : str
        The activation function to use in the model (e.g., "silu", "relu", etc.)
    similarity_type : str
        The type of similarity function to use for action embeddings (e.g., "dot", "cosine")
    init_temperature : float
        The initial temperature for the similarity function
    exploration : ExplorationConfig
        Configuration for exploration strategies
    """

    emb_dim: int = 128
    dropout: float = 0.0
    activation: str = "silu"
    similarity_type: str = "dot"
    init_temperature: float = 0.2
    init_logZ: float = 0.0
    graph_transformer: GraphTransformerConfig = field(
        default_factory=GraphTransformerConfig
    )
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
