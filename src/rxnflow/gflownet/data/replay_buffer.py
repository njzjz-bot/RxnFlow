from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...config import Config
from ...utils.misc import get_global_rng
from ..types import Traj


@dataclass(slots=True)
class ReplayConfig:
    """Replay buffer configuration

    Attributes
    ----------
    use : bool
        Whether to use a replay buffer
    capacity : int
        The capacity of the replay buffer
    warmup : int
        The number of samples to collect before starting to sample from the replay buffer
    num_from_replay : Optional[int]
        The number of replayed samples for a training batch (defaults to cfg.algo.num_from_policy, i.e. a 50/50 split)
    """

    use: bool = False
    capacity: int | None = None
    warmup: int | None = None
    num_from_replay: int | None = None


class ReplayBuffer:
    def __init__(self, cfg: Config):
        """
        Replay buffer for storing and sampling arbitrary data (e.g. transitions or trajectories)
        In self.push(), the buffer detaches any torch tensor and sends it to the CPU.
        """
        replay_cfg = cfg.replay
        assert replay_cfg.capacity is not None, (
            "ReplayBuffer capacity must be set in the config"
        )
        assert replay_cfg.warmup is not None, (
            "ReplayBuffer warmup must be set in the config"
        )

        self.capacity: int = replay_cfg.capacity
        self.warmup: int = replay_cfg.warmup
        assert self.warmup <= self.capacity, (
            "ReplayBuffer warmup must be smaller than capacity"
        )

        self.buffer: deque[Traj] = deque(maxlen=self.capacity)

    def push(self, traj: Traj):
        self.buffer.append(traj)

    def sample(self, batch_size: int) -> list[Traj]:
        if len(self.buffer) == 0:
            return []
        batch_size = min(batch_size, len(self.buffer))
        idxs = get_global_rng().choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in idxs]

    def __len__(self):
        return len(self.buffer)

    def get_connecting_hook(self) -> Callable[[list[Traj]], dict[str, Any]]:
        """Returns a hook that connects the replay buffer to the data source."""

        class BufferPushHook:
            def __init__(self, buffer: ReplayBuffer):
                self.buffer = buffer

            def __call__(self, trajs: list[Traj]) -> dict[str, Any]:
                """Pushes the trajectories to the replay buffer."""
                for traj in trajs:
                    self.buffer.push(traj)
                return {"replay_buffer_size": len(self.buffer)}

        return BufferPushHook(self)
