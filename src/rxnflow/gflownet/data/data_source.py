import warnings
from collections.abc import Callable, Generator, Sequence
from typing import Any

from torch import nn
from torch.utils.data import IterableDataset

from ...utils.misc import get_global_rng
from ..types import GFNAlgorithm, GFNEnvironmentContext, GFNTask, Traj
from .replay_buffer import ReplayBuffer

BatchInfo = dict[str, Any]
SamplingHook = Callable[[list[Traj]], BatchInfo]
Source = Generator[tuple[list[Traj], BatchInfo], None, None]


class DataSource(IterableDataset):
    def __init__(
        self,
        ctx: GFNEnvironmentContext,
        algo: GFNAlgorithm,
        task: GFNTask,
        start_at_step: int = 0,
    ):
        """A DataSource mixes multiple iterators into one. These are created with do_* methods."""
        self.ctx: GFNEnvironmentContext = ctx
        self.algo: GFNAlgorithm = algo
        self.task: GFNTask = task
        self.current_iter: int = start_at_step

        self.sources: list[Source] = []
        self.sampling_hooks: list[SamplingHook] = []
        self.replay_hooks: list[SamplingHook] = []
        self.data_hooks: list[SamplingHook] = []

    def add_sampling_hook(self, hook: SamplingHook):
        """Add a hook that is called when sampling new trajectories.

        The hook should take a list of trajectories as input.
        The hook will not be called on trajectories that are sampled from the replay buffer or dataset.
        """
        self.sampling_hooks.append(hook)

    def __iter__(self) -> Source:
        self.rng = get_global_rng()
        while True:
            self.current_iter += 1
            samples = [next(i, None) for i in self.sources]
            if any(i is None for i in samples):
                if not all(i is None for i in samples):
                    warnings.warn(
                        "Some iterators are done, but not all. You may be mixing incompatible iterators.",
                        stacklevel=2,
                    )
                    samples = [i for i in samples if i is not None]
                else:
                    break
            traj_lists, batch_infos = zip(*samples, strict=False)
            trajs: list[Traj] = sum(traj_lists, [])
            # Merge all the dicts into one
            batch_info: BatchInfo = {}
            for d in batch_infos:
                batch_info.update(d)
            yield trajs, batch_info

    def connect_model(self, model: nn.Module, num_samples: int):
        """Sample trajectories from the model."""

        def iterator() -> Source:
            while True:
                it = self.current_iter
                random_prob = self.algo.get_random_action_prob(it)
                trajs = self.algo.sample_training_data_from_model(
                    model, num_samples, random_prob
                )
                self.compute_rewards(trajs)

                # call the sampling hooks to compute additional batch info
                batch_info: BatchInfo = {}
                for hook in self.sampling_hooks:
                    batch_info.update(hook(trajs))

                yield trajs, batch_info

        self.sources.append(iterator())

    def connect_replay(self, replay_buffer: ReplayBuffer, num_samples: int):
        """Sample trajectories from the replay buffer."""

        # add a hook to push the sampled trajectories to the replay buffer
        self.sampling_hooks.append(replay_buffer.get_connecting_hook())

        def iterator() -> Source:
            while True:
                trajs = replay_buffer.sample(num_samples)

                # call the sampling hooks to compute additional batch info
                batch_info: BatchInfo = {}
                for hook in self.replay_hooks:
                    batch_info.update(hook(trajs))

                yield trajs, batch_info

        self.sources.append(iterator())

    def connect_dataset(self, data: Sequence[Traj], num_samples: int):
        """Sample trajectories from a dataset."""

        def iterator() -> Source:
            while True:
                idcs = self.rng.choice(len(data), num_samples, replace=False)
                trajs = [data[i] for i in idcs]
                for tj in trajs:
                    assert "reward" not in tj, (
                        "Traj should not have 'reward' set when sampling from dataset"
                    )
                    assert "is_valid" not in tj, (
                        "Traj should not have 'is_valid' set when sampling from dataset"
                    )

                # call the sampling hooks to compute additional batch info
                batch_info: BatchInfo = {}
                for hook in self.data_hooks:
                    batch_info.update(hook(trajs))

                yield trajs, batch_info

        self.sources.append(iterator())

    def compute_rewards(self, trajs: list[Traj]):
        """Sets trajs' reward and is_valid keys by querying the task."""
        # fetch the valid trajectories endpoints
        valid_idcs = [i for i, tj in enumerate(trajs) if tj["is_valid"]]
        objs = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]
        # ask the task to compute their reward
        rewards, m_is_valid = self.task.compute_rewards(objs)

        # insert the rewards and is_valid flags into the trajs
        j = 0
        for i, tj in enumerate(trajs):
            if i in valid_idcs:
                tj["reward"] = rewards[j]
                tj["is_valid"] = m_is_valid[j]
                j += 1
            else:
                tj["reward"] = 0.0
                tj["is_valid"] = False
