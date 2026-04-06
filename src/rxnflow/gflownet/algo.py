import dataclasses
import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch_geometric.data as gd
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ...config import Config
from ..types import (
    Action,
    GFNAlgorithm,
    GFNEnvironment,
    GFNEnvironmentContext,
    GFNSampler,
    Traj,
)


@dataclasses.dataclass
class AlgoConfig:
    """Configuration for trajectory balance algorithms

    Attributes
    ----------
    num_from_policy : int
        The number of on-policy samples for a training batch.
    max_len : int
        The maximum length of a trajectory. In this project, it is fixed value
    reward_exponent: float
        The exponent for the reward function, R(x)^β
    loss_fn: str
        The loss function to use for training
    temperature : float
        Softmax temperature used when sampling
    random_action_prob : float
        The probability of taking a random action during training
        Equivalent to the epsilon in epsilon-greedy exploration
    ema_decay : float
        The EMA factor(τ) for the sampling model (θ_ema = τ * θ_ema + (1-τ) * θ)
    random_workflow: bool
        Take random workflow for sampling
    """

    num_from_policy: int = 64
    max_len: int = 4  # NOTE: FIXED VALUE!!
    reward_exponent: float = 32.0  # beta in R(x)^{beta}
    reward_clip_min: float = 0.01
    loss_fn: str = "mse"  # 'mse', 'mae', 'huber'
    temperature: float = 1.0
    random_action_prob: float = 0.2
    ema_decay: float = 0.0
    random_workflow: bool = False


class TrajectoryBalanceModel(nn.Module, ABC):
    """A model that implements the trajectory balance algorithm for RxnFlow training.

    This model is used to train a GNN to predict the logZ of a trajectory
    and the log probabilities of actions in the trajectory.
    """

    @abstractmethod
    def logZ(self) -> Tensor:
        """Return the trainable logZ parameter."""

    @abstractmethod
    def compute_log_prob(self, batch: Any, actions: list[Action]) -> Tensor:
        """Predicts the log probabilities of actions in a trajectory.

        Parameters
        ----------
        batch: Any
            A batch of graphs, as constructed by `GFNAlgorithm.construct_batch`.
        actions: list[Action]
            List of action for each graph in the batch

        Returns
        -------
        log_prob: Tensor
            The predicted log probabilities of actions, shape (num_states,)
        """

    def logZ_parameters(self) -> list[torch.Tensor]:
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    def non_logZ_parameters(self) -> list[torch.Tensor]:
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" not in k]


class TrajectoryBalance[TBModel: TrajectoryBalanceModel](GFNAlgorithm, ABC):
    """A GFlowNet algorithm that implements the trajectory balance algorithm."""

    def __init__(
        self,
        env: GFNEnvironment,
        ctx: GFNEnvironmentContext,
        cfg: Config,
    ) -> None:
        """Instanciate a TB algorithm.

        Parameters
        ----------
        env: SynthesisEnv
            A graph environment.
        ctx: SynthesisEnvContext
            A context.
        cfg: Config
            Hyperparameters
        """
        self.ctx = ctx
        self.env = env
        self.cfg = cfg

        self.reward_exponent: float = self.cfg.algo.reward_exponent
        self.log_reward_clip_min: float = math.log(self.cfg.algo.reward_clip_min)
        self.tb_loss: str = self.cfg.algo.loss_fn
        self.temperature: float = self.cfg.algo.temperature
        self.sampler: GFNSampler = self.build_sampler()

    @abstractmethod
    def build_sampler(self) -> GFNSampler:
        """Builds a sampler for the trajectory balance algorithm."""

    def sample_training_data_from_model(
        self, model: TBModel, n: int, random_action_prob: float = 0.0
    ) -> list[Traj]:
        data = self.sampler.sample_from_model(
            model, n, self.temperature, random_action_prob
        )
        return data

    def construct_batch(self, trajs: list[Traj]) -> gd.Batch:
        """Construct a batch from a list of trajectories"""
        torch_graphs: list[gd.Data] = [
            self.ctx.graph_to_Data(g) for tj in trajs for g, _ in tj["traj"]
        ]
        actions: list[Action] = [action for tj in trajs for _, action in tj["traj"]]

        batch = self.ctx.collate(torch_graphs)

        # trajectory-level info
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs], dtype=torch.int32)
        batch.rewards = torch.tensor([i["reward"] for i in trajs], dtype=torch.float32)
        batch.num_trajs = len(trajs)

        # state-level info
        batch.log_p_B = torch.tensor(
            sum([i["bck_logprobs"] for i in trajs], []), dtype=torch.float32
        )
        batch.actions = actions  # list[Action]

        return batch

    def compute_loss(
        self, model: TBModel, trajs: list[Traj]
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the losses over trajectories

        Parameters
        ----------
        model: TrajectoryBalanceModel
            A model that implements the trajectory balance algorithm.
        trajs: list[Traj]
            A list of trajectories, each containing a sequence of graphs and actions.
        """
        dev = model.device

        # Construct a batch from the trajectories
        batch = self.construct_batch(trajs).to(dev)

        # A single trajectory is comprised of multiple graphs
        num_trajs: int = batch.num_trajs
        num_states: int = batch.num_graphs

        # === Compute the log reward (logR(x)^{beta}) === #
        rewards = batch.rewards  # [num_trajs,]
        log_R = rewards.log().clamp_(self.log_reward_clip_min)  # log(R(x))
        scaled_log_R = log_R * self.cfg.algo.reward_exponent  # log(R(x)^{beta})

        # === Compute the log prob of each action === #
        log_p_F = model.compute_log_prob(batch, batch.actions)
        log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape == (num_states,), (
            "Log probabilities should have the same shape as the number of states(graphs) in the batch"
        )
        # This is the log probability of each trajectory [num_trajs,]
        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,2,2,2,...].
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        traj_log_p_F = scatter_sum(log_p_F, batch_idx, dim=0, dim_size=num_trajs)
        traj_log_p_B = scatter_sum(log_p_B, batch_idx, dim=0, dim_size=num_trajs)

        # === Compute the log partition function logZ === #
        log_Z = model.logZ()  # [num_trajs,] logZ for each trajectory

        # === Compute trajectory loss: log(Z*Pf(τ) / R(x)*Pb(τ))^2 === #
        traj_losses = self._loss(log_Z + traj_log_p_F - scaled_log_R - traj_log_p_B)
        loss = traj_losses.mean()

        info = {
            "loss": loss.item(),
            "logZ": log_Z.mean().item(),
            "batch_entropy": -traj_log_p_F.mean().item(),
        }
        return loss, info

    def _loss(self, x: Tensor) -> Tensor:
        match self.tb_loss:
            case "mse":
                return x**2
            case "mae":
                return torch.abs(x)
            case "huber":
                ax = torch.abs(x)
                return torch.where(ax < 1, 0.5 * x**2, ax - 0.5)
            case _:
                raise ValueError(f"Unknown loss function: {self.tb_loss}")

    def get_random_action_prob(self, it: int) -> float:
        return self.cfg.algo.random_action_prob

