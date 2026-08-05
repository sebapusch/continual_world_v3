"""Gymnasium environment for stepping through Continual World v3 tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces
from metaworld.types import Task

from continualworld.tasks import resolve_sequence

META_WORLD_TIME_HORIZON = 200


class _SequenceOneHotWrapper(gym.ObservationWrapper):
    """Append a sequence identifier while preserving Meta-World's dtype."""

    def __init__(self, env: gym.Env, task_idx: int, num_tasks: int) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Meta-World must expose a Box observation space")
        dtype = env.observation_space.dtype
        self._one_hot = np.zeros(num_tasks, dtype=dtype)
        self._one_hot[task_idx] = 1
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [env.observation_space.low, np.zeros(num_tasks, dtype=dtype)]
            ),
            high=np.concatenate(
                [env.observation_space.high, np.ones(num_tasks, dtype=dtype)]
            ),
            dtype=dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, self._one_hot])


class _TaskSamplerWrapper(gym.Wrapper):
    """Select a Meta-World goal before each evaluation episode."""

    def __init__(
        self, env: gym.Env, tasks: Sequence[Task], rng: np.random.Generator
    ) -> None:
        super().__init__(env)
        self._tasks = tasks
        self._rng = rng

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        task_index = int(self._rng.integers(len(self._tasks)))
        self.env.unwrapped.set_task(self._tasks[task_index])
        return self.env.reset(seed=seed, options=options)


class ContinualWorldEnv(gym.Env[np.ndarray, np.ndarray]):
    """Run a fixed sequence of Meta-World v3 tasks.

    The active task changes after exactly ``steps_per_task`` interactions. A
    one-hot vector identifying the task's *position in the sequence* is
    appended to Meta-World's observation. At a sequence boundary ``step``
    returns ``truncated=True`` and the following ``reset`` starts the next
    task. The environment is exhausted after the final task budget.

    ``test_envs`` contains independently resettable evaluation environments
    in sequence order. Indexing it selects a particular sequence position;
    ``current_test_env`` (or ``test_env``) selects the active position. Their
    observations use the same sequence-position one-hot encoding as training.

    This class intentionally contains no learning algorithm. It is a small
    benchmark/environment layer that can be used with any Gymnasium agent.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        tasks: str | Sequence[str] = "CW20",
        *,
        steps_per_task: int = 1_000_000,
        episode_horizon: int = META_WORLD_TIME_HORIZON,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if steps_per_task <= 0:
            raise ValueError("steps_per_task must be positive")
        if episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")

        self.tasks = resolve_sequence(tasks if isinstance(tasks, str) else list(tasks))
        self.steps_per_task = int(steps_per_task)
        self.episode_horizon = int(episode_horizon)
        self.render_mode = render_mode
        self.num_tasks = len(self.tasks)
        self.total_step_limit = self.num_tasks * self.steps_per_task

        invalid = [name for name in self.tasks if name not in metaworld.MT1.ENV_NAMES]
        if invalid:
            raise ValueError(
                f"Unknown Meta-World v3 task name(s): {', '.join(invalid)}"
            )

        # Build benchmarks once per task type. CW20 therefore reuses the same
        # 50 Meta-World goals when it revisits a task in its second half.
        seed_sequence = np.random.SeedSequence(seed)
        unique_names = tuple(dict.fromkeys(self.tasks))
        child_seeds = seed_sequence.spawn(len(unique_names) + 2 * self.num_tasks)
        self._benchmarks: dict[str, metaworld.MT1] = {}
        for index, name in enumerate(unique_names):
            benchmark_seed = int(child_seeds[index].generate_state(1)[0])
            self._benchmarks[name] = metaworld.MT1(name, seed=benchmark_seed)

        training_rng_seeds = child_seeds[
            len(unique_names) : len(unique_names) + self.num_tasks
        ]
        self._task_rngs = [
            np.random.default_rng(child) for child in training_rng_seeds
        ]
        self._test_task_rngs = [
            np.random.default_rng(child)
            for child in child_seeds[len(unique_names) + self.num_tasks :]
        ]
        self._active_env: gym.Env | None = None
        self._active_raw_env: gym.Env | None = None
        self._needs_reset = True
        self._sequence_index = 0
        self._steps_in_task = 0
        self._total_steps = 0
        self._episode_success = False

        # Every Meta-World v3 environment has the same 39-dimensional state
        # layout and four-dimensional action layout. Obtain their dtypes and
        # shapes from the first class without retaining its MuJoCo model.
        first_benchmark = self._benchmarks[self.tasks[0]]
        probe = first_benchmark.train_classes[self.tasks[0]]()
        base_space = probe.observation_space
        self.action_space = probe.action_space
        probe.close()
        assert isinstance(base_space, spaces.Box)
        base_size = int(np.prod(base_space.shape))
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.full(base_size, -np.inf, dtype=base_space.dtype),
                    np.zeros(self.num_tasks, dtype=base_space.dtype),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(base_size, np.inf, dtype=base_space.dtype),
                    np.ones(self.num_tasks, dtype=base_space.dtype),
                ]
            ),
            dtype=base_space.dtype,
        )
        self.test_envs: list[gym.Env] = [
            self._build_test_env(index) for index in range(self.num_tasks)
        ]

    @property
    def current_task_index(self) -> int | None:
        """Current sequence index, or ``None`` after all budgets are consumed."""
        return self._sequence_index if self._sequence_index < self.num_tasks else None

    @property
    def current_task_name(self) -> str | None:
        """Current Meta-World environment name, or ``None`` when exhausted."""
        index = self.current_task_index
        return self.tasks[index] if index is not None else None

    @property
    def current_test_env(self) -> gym.Env | None:
        """Evaluation environment matching the current sequence position."""
        index = self.current_task_index
        return self.test_envs[index] if index is not None else None

    @property
    def test_env(self) -> gym.Env | None:
        """Short alias for :attr:`current_test_env`."""
        return self.current_test_env

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def exhausted(self) -> bool:
        return self._sequence_index >= self.num_tasks

    def _build_active_env(self) -> gym.Env:
        name = self.current_task_name
        index = self.current_task_index
        if name is None or index is None:
            raise RuntimeError("The Continual World sequence is exhausted")

        benchmark = self._benchmarks[name]
        raw_env = benchmark.train_classes[name](render_mode=self.render_mode)
        wrapped: gym.Env = _SequenceOneHotWrapper(
            raw_env, task_idx=index, num_tasks=self.num_tasks
        )
        wrapped = gym.wrappers.TimeLimit(
            wrapped, max_episode_steps=self.episode_horizon
        )
        self._active_raw_env = raw_env
        self._active_env = wrapped
        return wrapped

    def _build_test_env(self, index: int) -> gym.Env:
        """Build an independently resettable environment for one sequence entry."""
        name = self.tasks[index]
        benchmark = self._benchmarks[name]
        raw_env = benchmark.train_classes[name](render_mode=self.render_mode)
        wrapped: gym.Env = _TaskSamplerWrapper(
            raw_env, benchmark.train_tasks, self._test_task_rngs[index]
        )
        wrapped = _SequenceOneHotWrapper(
            wrapped, task_idx=index, num_tasks=self.num_tasks
        )
        return gym.wrappers.TimeLimit(
            wrapped, max_episode_steps=self.episode_horizon
        )

    def _select_goal(self) -> Task:
        index = self.current_task_index
        name = self.current_task_name
        if index is None or name is None:
            raise RuntimeError("The Continual World sequence is exhausted")
        goals = self._benchmarks[name].train_tasks
        goal_index = int(self._task_rngs[index].integers(len(goals)))
        return goals[goal_index]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.exhausted:
            raise RuntimeError(
                "Cannot reset: the Continual World sequence is exhausted"
            )
        if seed is not None:
            assert self.current_task_index is not None
            self._task_rngs[self.current_task_index] = np.random.default_rng(seed)

        env = self._active_env or self._build_active_env()
        assert self._active_raw_env is not None
        self._active_raw_env.unwrapped.set_task(self._select_goal())
        observation, info = env.reset(seed=seed, options=options)
        self._needs_reset = False
        self._episode_success = False
        info.update(self._task_info())
        return observation, info

    def _task_info(self) -> dict[str, Any]:
        return {
            "task_name": self.current_task_name,
            "sequence_index": self.current_task_index,
            "steps_in_task": self._steps_in_task,
            "total_steps": self._total_steps,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.exhausted:
            raise RuntimeError("Cannot step: the Continual World sequence is exhausted")
        if self._needs_reset or self._active_env is None:
            raise RuntimeError("Call reset() before step() or after an episode ends")

        task_name = self.current_task_name
        task_index = self.current_task_index
        observation, reward, terminated, truncated, info = self._active_env.step(action)
        self._steps_in_task += 1
        self._total_steps += 1
        self._episode_success = self._episode_success or bool(
            info.get("success", False)
        )

        task_boundary = self._steps_in_task >= self.steps_per_task
        if task_boundary:
            truncated = True
        self._needs_reset = bool(terminated or truncated)

        # Report the task which produced this transition, even though the
        # public current-task properties advance immediately at the boundary.
        info.update(
            {
                "task_name": task_name,
                "sequence_index": task_index,
                "steps_in_task": self._steps_in_task,
                "total_steps": self._total_steps,
                "task_boundary": task_boundary,
                "episode_success": self._episode_success,
            }
        )

        if task_boundary:
            self._active_env.close()
            self._active_env = None
            self._active_raw_env = None
            self._sequence_index += 1
            self._steps_in_task = 0

        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        if self._active_env is None:
            raise RuntimeError("Call reset() before render()")
        return self._active_env.render()

    def close(self) -> None:
        if self._active_env is not None:
            self._active_env.close()
        for env in self.test_envs:
            env.close()
        self._active_env = None
        self._active_raw_env = None


def get_cl_env(
    tasks: str | Sequence[str] = "CW20",
    steps_per_task: int = 1_000_000,
    *,
    episode_horizon: int = META_WORLD_TIME_HORIZON,
    seed: int | None = None,
    render_mode: str | None = None,
) -> ContinualWorldEnv:
    """Compatibility-friendly constructor for a continual-learning environment."""
    return ContinualWorldEnv(
        tasks,
        steps_per_task=steps_per_task,
        episode_horizon=episode_horizon,
        seed=seed,
        render_mode=render_mode,
    )
