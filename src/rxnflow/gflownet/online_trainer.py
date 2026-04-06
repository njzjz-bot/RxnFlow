import copy
import gc
import logging
import math
import shutil
import time
from abc import ABC
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

import torch
import wandb
from omegaconf import OmegaConf
from rdkit import RDLogger
from torch import Tensor, nn

from ..config import Config
from ..utils.misc import create_logger, set_global_device, set_global_rng_seed
from .algo.trajectory_balance import TrajectoryBalanceModel
from .data.data_source import DataSource
from .data.replay_buffer import ReplayBuffer
from .types import Callback, GFNTask, GFNTrainer, Sampler, SamplingHook, Traj
from .utils.sqlite_log import SQLiteLogHook


def cycle[T](it: Iterable[T]) -> Generator[T, None, None]:
    while True:
        yield from it


@torch.no_grad()
def model_grad_norm(model) -> float:
    total_norm_sq = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.norm().item() ** 2
    return math.sqrt(total_norm_sq)


class SamplingMetricHook:
    def __call__(self, trajs: list[Traj]) -> dict[str, float]:
        assert len(trajs) > 0, "No new trajectories sampled"
        is_valid: list[bool] = [t["is_valid"] for t in trajs]
        traj_lens: list[int] = [len(t["traj"]) for t in trajs]
        rewards: list[float] = [t["reward"] for t in trajs if t["is_valid"]]
        return {
            "invalid_ratio": 1 - sum(is_valid) / len(trajs),
            "traj_len": sum(traj_lens) / len(traj_lens) if traj_lens else float("nan"),
            "avg_reward": sum(rewards) / len(rewards) if rewards else float("nan"),
        }


class BaseGFNTrainer(GFNTrainer[Config], ABC):
    """Base class for GFlowNet trainers with trajectory balance objective."""

    model: TrajectoryBalanceModel
    loggers: list[logging.Logger]
    callbacks: list[Callback]
    global_step: int

    def __init__(
        self,
        config: Config,
        logger: logging.Logger | None = None,
        callbacks: list[Callable] | None = None,
    ):
        """A GFlowNet trainer with Trajectory balance objective

        Notes
        -----
        There are three sources of config values
          - The default values specified in individual config classes
          - The default values specified in the `default_hps` method, typically what is defined by a task
          - The values passed in the constructor, typically what is called by the user
        The final config is obtained by merging the three sources with the following precedence:
          config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        """
        # setup the config and export it
        default_cfg: Config = self.get_default_cfg()
        self.set_default_hps(default_cfg)
        self.cfg: Config = OmegaConf.merge(default_cfg, config)

        self.global_step = self.cfg.start_at_step
        self.print_every = self.cfg.print_every
        self.checkpoint_every = self.cfg.checkpoint_every

        # GFlowNet setup
        self.setup()

        # Log hook
        self.logger = logger or create_logger(
            "gflownet", logfile=Path(self.cfg.log_dir) / "train.log"
        )

        # Callback hook
        self.callbacks = [self.checkpoint_hook, self.garbage_collection_hook] + (
            callbacks or []
        )

    # === Configuration === #
    @property
    def config_path(self) -> Path:
        return Path(self.cfg.log_dir) / "config.yaml"

    def get_default_cfg(self) -> Config:
        return Config()

    def setup(self):
        RDLogger.DisableLog("rdApp.*")
        self.setup_device()
        self.setup_seed()
        self.setup_log_dir()

        # setup the model, algo, task, env, and ctx
        super().setup()

        # export the config to a file
        self.export_config()

        # print model summary
        self.print_model_summary()

    def setup_device(self):
        self.device = torch.device(self.cfg.device)
        self.check_device()
        set_global_device(self.device)

    def check_device(self):
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Using CPU instead.")
                self.device = torch.device("cpu")

    def setup_seed(self):
        set_global_rng_seed(self.cfg.seed)
        # to ensure the model initialization is deterministic
        torch.manual_seed(self.cfg.seed)

    def setup_log_dir(self):
        if Path(self.cfg.log_dir).exists():
            if self.cfg.overwrite_existing_exp:
                logger.warning("Deleting existing log dir: {}", self.cfg.log_dir)
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def setup_opt(self):
        """Construct the optimizer and lr scheduler for trajectory balance model"""
        if self.cfg.opt.opt == "adam":
            opt_cls = torch.optim.Adam
        elif self.cfg.opt.opt == "adamw":
            opt_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.opt.opt}")

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = self.model.logZ_parameters()
        non_Z_params = self.model.non_logZ_parameters()
        self.optimizer = opt_cls(
            non_Z_params,
            lr=self.cfg.opt.learning_rate,
            betas=self.cfg.opt.betas,
            weight_decay=self.cfg.opt.weight_decay,
            eps=self.cfg.opt.eps,
        )
        self.optimizer.add_param_group(
            {"params": Z_params, "lr": self.cfg.opt.Z_learning_rate}
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay)
        )

    def export_config(self):
        yaml_cfg = OmegaConf.to_yaml(self.cfg)
        print("Hyperparameters (agent):")
        print()
        print(yaml_cfg.strip())
        print()
        with open(self.config_path, "w", encoding="utf8") as f:
            f.write(yaml_cfg)

    def print_model_summary(self):
        total_params = 0
        print("Model Summary")
        print(f"{'Layer':<20}{'Type':<20}{'Parameters':>12}")
        print("-" * 52)

        # Print the submodules
        for name, module in self.model.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if module_params > 0 or len(list(module.children())) > 0:
                print(f"{name:<20}{module.__class__.__name__:<20}{module_params:>12,}")
            total_params += module_params

        # Print parameters
        for name, param in self.model.named_parameters(recurse=False):
            if param.requires_grad:
                param_count = param.numel()
                print(f"{name:<20}{'nn.Parameter':<20}{param_count:>12,}")
                total_params += param_count

        print("-" * 52)
        print(f"{'Total':<20}{'':<20}{total_params:>12,}")
        print()

    # === Model training === #
    def run(self, task: GFNTask):
        """Run the training loop for the given task"""
        self.task = task

        # move to the device
        self.model.to(self.device)

        # create the data loader
        data_sampler = self.build_training_data_sampler()

        # Train the model
        self._fit(data_sampler)

    def _fit(self, data_sampler: Sampler):
        """Fit the model using the provided data loader"""
        self.logger.info("Starting training")
        start_time = time.time()
        for trajs, sample_info in cycle(data_sampler):
            self.global_step += 1

            # Train the model on the current batch
            step_info = self.training_step(trajs)
            step_info["time_spent"] = time.time() - start_time
            start_time = time.time()

            # Callback
            info = step_info | sample_info
            self.log_hook(info, self.global_step)
            for hook in self.callbacks:
                hook(self.global_step)

            # Break if the training is done
            if self.global_step >= self.cfg.num_training_steps:
                break
        # Save the final model state
        self._save_state()

    def training_step(self, trajs: list[Traj]) -> dict[str, float]:
        """Train the model on a single batch and return the training information"""
        tick = time.time()
        self.model.train()
        loss, info = self.algo.compute_loss(self.model, trajs)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        step_info = self.step(loss)
        info.update(step_info)
        info["train_time"] = time.time() - tick
        return info

    def step(self, loss: Tensor) -> dict[str, float]:
        """Step the optimizer and returns additional information"""
        self.optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Compute the gradient norm before clipping
        grad_norm = model_grad_norm(self.model)

        if (clip_val := self.cfg.opt.gradient_clip_val) > 0:
            # Clip gradients except for the logZ parameters
            non_Z_params = self.model.non_logZ_parameters()
            torch.nn.utils.clip_grad_norm_(non_Z_params, clip_val)

        # Step the optimizers and lr schedulers
        self.optimizer.step()
        self.lr_scheduler.step()

        return {"grad_norm": grad_norm}

    # === Logging === #
    def log_hook(self, info: dict[str, Any], it: int):
        """Log the training loss and other information every `self.print_every` iterations"""
        if wandb.run is not None:
            wandb.log(
                {f"gflownet/{k}": v for k, v in info.items()}, step=self.global_step
            )
        if it % self.print_every == 0:
            self.logger.info(
                f"Iteration {it}: " + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
            )

    # ==== Callbacks ==== #
    def garbage_collection_hook(self, it: int):
        """A simple callback that collects garbage and empties the CUDA cache."""
        if it % 1024 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def checkpoint_hook(self, it: int):
        """A simple callback that saves the model state every `self.checkpoint_every` iterations."""
        if self.checkpoint_every > 0 and it % self.checkpoint_every == 0:
            self._save_state()

    # === Internal methods === #
    def _save_state(self, it: int | None = None):
        it = it or self.global_step
        state = {
            "model_state_dict": self.model.state_dict(),
            "cfg": self.cfg,
            "step": it,
        }
        fn = Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(state, fd)
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, Path(self.cfg.log_dir) / f"model_state_{it}.pt")


class OnlineTrainer(BaseGFNTrainer):
    """Compatible with the standard GFlowNet training and Double-GFlowNet training"""

    sampling_model: nn.Module
    sampling_hooks: list[SamplingHook]
    replay_buffer: ReplayBuffer | None

    def setup(self):
        super().setup()
        self.setup_replay_buffer()
        self.setup_sampling_model()

        # This is called after sampling and reward computation.
        self.sampling_hooks = []
        self.sampling_hooks.append(
            SamplingMetricHook()
        )  # default sampling hook to log metrics
        if self.cfg.use_sqlite_log:
            self.sampling_hooks.append(
                SQLiteLogHook(str(Path(self.cfg.log_dir) / "sqlite_log.db"), self.ctx)
            )

    def setup_sampling_model(self):
        """Set up the sampling model with EMA"""
        self.ema_decay = self.cfg.algo.ema_decay
        if self.ema_decay > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

    def step(self, loss: Tensor) -> dict[str, float]:
        """Perform a training step and update the sampling model with EMA if applicable"""
        step_info = super().step(loss)
        # EMA update
        if self.ema_decay > 0:
            with torch.no_grad():
                for p, ema_p in zip(
                    self.model.parameters(), self.sampling_model.parameters(), strict=True
                ):
                    ema_p.lerp_(p, 1 - self.ema_decay)
        return step_info

    def setup_replay_buffer(self):
        """Set up the replay buffer, which is used for off-policy sampling"""
        if self.cfg.replay.use:
            self.replay_buffer = ReplayBuffer(self.cfg)
        else:
            self.replay_buffer = None

    def run(self, task: GFNTask):
        """Run the training loop for the given task"""
        # Set up the task
        self.task = task

        # Add the sampling hook of the task
        self.sampling_hooks.append(self.task.get_sampling_hook())

        # Move to the device
        self.model.to(self.device)
        self.sampling_model.to(self.device)

        # Create the data loader
        data_sampler = self.build_training_data_sampler()

        # Warm up the replay buffer if needed
        if (
            self.replay_buffer is not None
            and len(self.replay_buffer) < self.replay_buffer.warmup
        ):
            self._warmup(data_sampler)

        # Train the model
        self._fit(data_sampler)

    def _warmup(self, sampler: Sampler):
        """Warm up the replay buffer without training"""
        assert self.replay_buffer is not None, "Replay buffer is not set up"
        assert len(self.replay_buffer) < self.replay_buffer.warmup, (
            "Replay buffer warmup is already done"
        )
        self.logger.info("Warm up replay buffer")
        for _ in cycle(sampler):
            self.global_step += 1
            self.logger.info(
                f"Iteration {self.global_step}: {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
            )
            if len(self.replay_buffer) >= self.replay_buffer.warmup:
                self.logger.info("Replay buffer warmup done")
                break

    def build_training_data_sampler(self) -> Sampler:
        """Build the training data sampler that samples trajectories from the model and replay buffer."""
        replay_buffer = self.replay_buffer
        model = self.sampling_model

        num_from_policy = self.cfg.algo.num_from_policy
        num_from_replay = (
            self.cfg.replay.num_from_replay or num_from_policy
            if replay_buffer is not None
            else 0
        )

        data_source = DataSource(self.ctx, self.algo, self.task)
        # add replay first to avoid that the samples from replay are just sampled from the model
        if replay_buffer is not None:
            data_source.connect_replay(replay_buffer, num_from_replay)
        # then sample from the model
        data_source.connect_model(model, num_from_policy)

        # add sampling hooks
        for hook in self.sampling_hooks:
            data_source.add_sampling_hook(hook)

        return data_source

    def _save_state(self, it: int | None = None):
        it = it or self.global_step
        state = {
            "model_state_dict": self.model.state_dict(),
            "cfg": self.cfg,
            "step": it,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = self.sampling_model.state_dict()

        fn = Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(state, fd)
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, Path(self.cfg.log_dir) / f"model_state_{it}.pt")
