from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Self

from omegaconf import MISSING

from .gflownet.algo.config import AlgoConfig
from .gflownet.data.config import ReplayConfig
from .models.config import ModelConfig


@dataclass(slots=True)
class OptimizerConfig:
    """Generic configuration for optimizers

    Attributes
    ----------
    opt : str
        The optimizer to use (either "adam" or "adamw")
    learning_rate : float
        The learning rate
    Z_learning_rate : float
        The learning rate for the logZ parameter
    lr_decay : float
        The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))
    weight_decay : float
        The L2 weight decay
    betas : tuple[float, float]
        The betas for the Adam optimizer (default: (0.9, 0.999))
    eps : float
        The epsilon parameter for Adam
    gradient_clip_val : float
        The maximum gradient norm for gradient clipping
    """

    opt: str = "adamw"
    learning_rate: float = 1e-4
    Z_learning_rate: float = 1e-1
    lr_decay: float = 20_000
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    eps: float = 1e-8
    gradient_clip_val: float = 100.0


@dataclass
class ConstraintConfig:
    """Configuration for molecular size constraint

    Attributes
    ----------
    heavyatom : int | None
    hba : int | None
    hbd : int | None
    arom (aromatic ring) : int | None
    ring : int | None
    rotb (rotatble bond) : int | None
    mw : float | None
    tpsa : float | None
    logp: float | None

    Notes
    -----
    LogP can be decreased by adding more polar groups, while MW can be increased by adding more heavy atoms.
    Therefore, you have to set LogP constraints carefully. (I recommend setting LogP with margin of 1.0 or more)
    """

    heavyatom: int | None = None
    hba: int | None = None
    hbd: int | None = None
    arom: int | None = None
    ring: int | None = None
    rotb: int | None = None
    mw: float | None = None
    tpsa: float | None = None
    logp: float | None = None

    def __setitem__(self, key: str, value: int | float | None):
        self.__dict__[key] = value

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def to_dict(self) -> dict[str, float]:
        dic = asdict(self)
        return {k: v for k, v in dic.items() if v is not None}


@dataclass(slots=True)
class Config:
    """Base configuration for training

    Attributes
    ----------
    desc : str
        A description of the experiment
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda")
    seed : int
        The random seed
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    store_all_checkpoints : bool
        Whether to store all checkpoints or only the last one
    print_every : int
        The number of training steps after which to print the training loss
    num_training_steps : int
        The number of training steps
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    desc: str = "noDesc"
    env_dir: str = MISSING
    log_dir: str = MISSING
    device: str = "cuda"
    restart: bool = False
    seed: int = 1
    checkpoint_every: int = 1000
    store_all_checkpoints: bool = False
    print_every: int = 100
    num_training_steps: int = 10_000
    overwrite_existing_exp: bool = False
    pretrained_model_path: str | None = None
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)

    @classmethod
    def get_empty(cls) -> Self:
        """Get an empty instance of the Config class with all fields set to MISSING."""
        return init_empty(cls())


def init_empty[T](cfg: T) -> T:
    """
    Initialize a dataclass instance with all fields set to MISSING,
    including nested dataclasses.

    This is meant to be used on the user side (tasks) to provide
    some configuration using the Config class while overwritting
    only the fields that have been set by the user.
    """
    for f in fields(cfg):
        if is_dataclass(f.type):
            setattr(cfg, f.name, init_empty(f.type()))
        else:
            setattr(cfg, f.name, MISSING)

    return cfg
