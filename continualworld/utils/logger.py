import sys
from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import wandb


class Logger(ABC):
    def __init__(self) -> None:
        self._timestep: int = 0
        self._supports_video: bool = False

    @property
    def supports_video(self) -> bool:
        return self._supports_video


    def increase_timestep(self) -> None:
        self._timestep += 1

    def set_timestep(self, timestep: int) -> None:
        self._timestep = timestep

    @abstractmethod
    def log(self, metric: str, value: float) -> None:
        ...

    def log_video(self, name: str, frames: np.ndarray) -> None:
        return

    def flush(self) -> None:
        sys.stdout.flush()

    def close(self) -> None:
        self.flush()


class TerminalLogger(Logger):
    def log(self, metric: str, value: float) -> None:
        print(f'[{self._timestep}] {metric}: {value:.2f}')


class WandbLogger(Logger):
    """Buffer metrics and send them to Weights & Biases as one timestep."""

    def __init__(
            self,
            project: str,
            entity: str | None = None,
            name: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.entity = entity
        self.name = name
        self.config = dict(config) if config is not None else None
        self._metrics: dict[str, float] = {}
        self._run: wandb.Run | None = None
        self._supports_video: bool = True

    def log(self, metric: str, value: float) -> None:
        self._metrics[metric] = value

    def log_video(self, name: str, frames: np.ndarray) -> None:
        self._init_run()
        self._run.log({
            'video': wandb.Video(frames, fps=30, format='mp4'),
        })

    def flush(self) -> None:
        if not self._metrics:
            return

        self._init_run()

        self._run.log(self._metrics, step=self._timestep)
        self._metrics.clear()

    def close(self) -> None:
        self.flush()
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _init_run(self) -> None:
        if self._run is None:
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
            )