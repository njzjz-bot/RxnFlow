import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, NewType

import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

# === Type aliases === #
# This type represents a trajectory
Traj = NewType("Traj", dict[str, Any])

# Thes types represent a graph action, which is an action that can be applied to a graph
Action = object
ActionType = enum.Enum

# This type represents a sampler that yields trajectories and batch information
BatchInfo = dict[str, Any]
Sampler = Iterable[tuple[list[Traj], BatchInfo]]

# These types represent callbacks and hooks
SamplingHook = Callable[[list[Traj]], BatchInfo]
Callback = Callable[[int], None]


# === GFlowNet instances === #
class GFNTask[Tobj](ABC):
    """Sets up a common structure of task"""

    # === TODO implement === #
    @abstractmethod
    def calc_rewards(self, objs: list[Tobj]) -> list[float]:
        """TODO Implement: Calculate the rewards for objects

        Parameters
        ----------
        objs : list[Any]
            A list of valid objects.

        Returns
        -------
        rewards: list[float]
            A list of rewards for the objects.
        """

    def check_is_valid(self, objs: list[Tobj]) -> list[bool]:
        """TODO Implement if needed: Check if the objects are valid

        Parameters
        ----------
        objs : list[Any]
            A list of objects.

        Returns
        -------
        is_valids: bool
            Whether the objects are valid or not.
            If not implemented, all objects are considered valid.
        """
        return [True for _ in objs]

    def get_task_info(self) -> dict[str, float]:
        """Hook to get task information after the reward computation of samples."""
        return {}

    # ==== Inner functions ==== #
    def compute_rewards(self, objs: list[Tobj]) -> tuple[list[float], list[bool]]:
        is_valid = self.check_is_valid(objs)
        rewards = [0.0] * len(objs)

        if any(is_valid):
            valid_objs = [obj for flag, obj in zip(is_valid, objs, strict=True) if flag]
            valid_rewards = self.calc_rewards(valid_objs)
            assert len(valid_rewards) == len(valid_objs)
            rewards = [0.0] * len(objs)
            j = 0
            for i, flag in enumerate(is_valid):
                if flag:
                    rewards[i] = valid_rewards[j]
                    j += 1
        return rewards, is_valid

    def get_sampling_hook(self) -> Callable[[list[Traj]], BatchInfo]:
        """Returns a hook that can be used to get task information after sampling trajectories."""

        class _SamplingHook:
            def __init__(self, task: GFNTask):
                self.task = task

            def __call__(self, _: list[Traj]) -> BatchInfo:
                """Hook to get task information after sampling trajectories."""
                return self.task.get_task_info()

        return _SamplingHook(self)


class GFNEnvironment[Tgraph, Taction](ABC):
    """Molecules and reaction templates environment. The new (initial) state are Empty Molecular Graph.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    @abstractmethod
    def new(self) -> Tgraph:
        """Returns a new empty molecular graph."""

    @abstractmethod
    def step(self, g: Tgraph, action: Taction) -> Tgraph:
        """Applies the action to the current state and returns the next state.
        Parameters
        ----------
        g : Tgraph
            The current state of the environment.
        action : Taction
            The action to be applied to the current state.
        Returns
        -------
        g_next : Tgraph
            The next state of the environment after applying the action.
        """

    @abstractmethod
    def reverse(self, ra: Taction) -> Taction:
        """Returns the reverse action of the given action."""


class GFNEnvironmentContext[Tobj, Tgraph, Taction: Action, Tactiontype: ActionType](ABC):
    """This context specifies how to create molecules by applying reaction templates."""

    @abstractmethod
    def get_action(self, action_type: Tactiontype, *args: int, **kwargs: int) -> Taction:
        """Create an action of the given type with the given arguments."""

    @abstractmethod
    def graph_to_Data(self, g: Tgraph) -> gd.Data:
        """Convert a Graph to a torch geometric Data instance"""

    def collate(self, graphs: list[gd.Data]) -> gd.Batch:
        return gd.Batch.from_data_list(graphs, follow_batch=["x"])

    @abstractmethod
    def obj_to_graph(self, obj: Tobj) -> Tgraph:
        """Convert an RDMol to a Graph"""

    @abstractmethod
    def graph_to_obj(self, g: Tgraph) -> Tobj:
        """Convert a Graph to an object"""

    @abstractmethod
    def object_to_log_repr(self, g: Tgraph) -> str:
        """Convert a Graph to a string representation"""

    @abstractmethod
    def traj_to_log_repr(self, traj: list) -> str:
        """Convert a Trajectory to a string representation"""


class GFNAlgorithm[Tbatch](ABC):
    """A GFlowNet algorithm that defines how to sample trajectories and compute losses"""

    @abstractmethod
    def get_random_action_prob(self, it: int) -> float:
        """epsilon-greedy exploration probability"""

    @abstractmethod
    def sample_training_data_from_model(
        self, model: nn.Module, num_samples: int, random_action_prob: float
    ) -> list[Traj]:
        """Creates training data from the model"""

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, trajs: list[Traj]
    ) -> tuple[Tensor, BatchInfo]:
        """Computes the loss over trajectories

        Parameters
        ----------
        model: nn.Module
            The model being trained
        trajs: list[Traj]
            A list of trajectories, where each trajectory is a dict with keys:
            - trajs: list[tuple[Tgraph, Taction]]
                the list of states and actions
            - bck_logprobs: list[float]
                list of [log P_B]
            - is_valid: bool
                is the generated graph valid according to the env & ctx
            - reward: float
                the reward for the trajectory

        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, float]
            Logged information about model predictions.
        """


class GFNSampler(ABC):
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(self, env: GFNEnvironment, ctx: GFNEnvironmentContext):
        """
        Parameters
        ----------
        env: SynthesisEnv
            A synthesis-oriented environment.
        ctx: SynthesisEnvContext
            A context.
        """
        self.env: GFNEnvironment = env
        self.ctx: GFNEnvironmentContext = ctx

    @abstractmethod
    def sample_from_model(
        self,
        model: nn.Module,
        n: int,
        temperature: float = 1.0,
        random_action_prob: float = 0.0,
    ) -> list[Traj]:
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model to sample from.
        n: int
            Number of graphs to sample
        temperature: float
            Softmax temperature used when sampling
        random_action_prob: float
            The probability of taking a random action

        Returns
        -------
        data: list[Traj]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[tuple[MolGraph, RxnAction]]
                the list of states and actions
           - bck_logprobs: list[float]
                list of [log P_B]
           - is_valid: bool
                is the generated graph valid according to the env & ctx
        """

    @abstractmethod
    def sample_inference(
        self,
        model: nn.Module,
        n: int,
        temperature: float = 1.0,
        random_action_prob: float = 0.0,
    ) -> list[Traj]:
        """Model Sampling (Inference - Non Retrosynthetic Analysis)

        Parameters
        ----------
        model: nn.Module
            Model to sample from.
        n: int
            Number of samples
        temperature: float
            Softmax temperature used when sampling
        random_action_prob: float
            The probability of taking a random action

        Returns
        -------
        data: list[Traj]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Chem.Mol, RxnAction]], the list of states and actions
           - is_valid: is the generated graph valid according to the env & ctx
        """


class GFNTrainer[Tconfig](ABC):
    model: nn.Module
    cfg: Tconfig
    env: GFNEnvironment
    ctx: GFNEnvironmentContext
    task: GFNTask
    algo: GFNAlgorithm

    def __init__(self, config: Tconfig):
        """A gflownet trainer"""
        self.cfg = config
        self.setup()

    @abstractmethod
    def run(self, task: GFNTask):
        """Run the GFlowNet training loop for the given task."""
        raise NotImplementedError(
            "GFlowNet training loop should be implemented in the subclass."
        )

    @abstractmethod
    def get_default_cfg(self) -> Tconfig:
        """Get the default configuration dataclass for the GFlowNet trainer."""

    def set_default_hps(self, base: Tconfig):  # noqa
        """Set the default hyperparameters for the GFlowNet trainer.

        This method should be called to set the default hyperparameters for the GFlowNet trainer.
        It will be called automatically when the trainer is initialized.
        """
        pass

    def setup(self):
        self.setup_env()
        self.setup_env_context()
        self.setup_model()
        self.setup_algo()
        self.setup_opt()

    @abstractmethod
    def setup_env(self):
        """Set up the environment, which is used to convert graphs to Data objects"""

    @abstractmethod
    def setup_env_context(self):
        """Set up the environment context, which is used to convert graphs to Data objects"""

    @abstractmethod
    def setup_algo(self):
        """Set up the gflownet algorithm"""

    @abstractmethod
    def setup_model(self):
        """Set up the model and sampling model"""

    @abstractmethod
    def setup_opt(self):
        """Set up the optimizer for the model"""

    @abstractmethod
    def step(self, loss: Tensor) -> dict[str, Any]:
        """Step the optimizer and returns additional information"""

    @abstractmethod
    def build_training_data_sampler(self) -> Sampler:
        """Build the training data sampler that samples trajectories from the model and replay buffer."""
