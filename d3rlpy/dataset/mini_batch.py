import dataclasses
from typing import Sequence, Union

import numpy as np

from .components import PartialTrajectory, Transition
from .types import Shape
from .utils import (
    cast_recursively,
    check_dtype,
    check_non_1d_array,
    get_shape_from_observation_sequence,
    stack_observations,
)

__all__ = ["TransitionMiniBatch", "TrajectoryMiniBatch"]


@dataclasses.dataclass(frozen=True)
class TransitionMiniBatch:
    r"""Mini-batch of transitions.

    Args:
        observations: Batched observations.
        actions: Batched actions.
        rewards: Batched rewards.
        next_observations: Batched next observations.
        terminals: Batched environment terminal flags.
        intervals: Batched timesteps between observations and next
            observations.
    """
    observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, ...)
    actions: np.ndarray  # (B, ...)
    rewards: np.ndarray  # (B, 1)
    next_observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, ...)
    terminals: np.ndarray  # (B, 1)
    intervals: np.ndarray  # (B, 1)
    behavior_policy: np.ndarray  # (B, A)
    next_behavior_policy: np.ndarray  # (B, A)

    def __post_init__(self) -> None:
        assert check_non_1d_array(self.observations)
        assert check_dtype(self.observations, np.float32)
        assert check_non_1d_array(self.actions)
        assert check_dtype(self.actions, np.float32)
        assert check_non_1d_array(self.rewards)
        assert check_dtype(self.rewards, np.float32)
        assert check_non_1d_array(self.next_observations)
        assert check_dtype(self.next_observations, np.float32)
        assert check_non_1d_array(self.terminals)
        assert check_dtype(self.terminals, np.float32)
        assert check_non_1d_array(self.intervals)
        assert check_dtype(self.intervals, np.float32)
        assert check_non_1d_array(self.behavior_policy)
        assert check_dtype(self.behavior_policy, np.float32)
        assert check_non_1d_array(self.next_behavior_policy)
        assert check_dtype(self.next_behavior_policy, np.float32)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "TransitionMiniBatch":
        r"""Constructs mini-batch from list of transitions.

        Args:
            transitions: List of transitions.

        Returns:
            Mini-batch.
        """
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack(
            [transition.action for transition in transitions], axis=0
        )
        rewards = np.stack(
            [transition.reward for transition in transitions], axis=0
        )
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        behavior_policy = np.stack(
            [transition.behavior_policy for transition in transitions], axis=0
        )
        next_behavior_policy = np.stack(
            [transition.next_behavior_policy for transition in transitions],
            axis=0,
        )
        # Reshape in the case of continuous action space, i.e. one probability per transition
        if len(behavior_policy.shape) == 1:
            behavior_policy = behavior_policy.reshape(-1, 1)
        if len(next_behavior_policy.shape) == 1:
            next_behavior_policy = next_behavior_policy.reshape(-1, 1)
        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            behavior_policy=cast_recursively(behavior_policy, np.float32),
            next_behavior_policy=cast_recursively(next_behavior_policy, np.float32),
        )

    @property
    def observation_shape(self) -> Shape:
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]  # type: ignore

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]  # type: ignore

    def __len__(self) -> int:
        return int(self.actions.shape[0])
    
    @classmethod
    def from_partial_trajectories(
        cls, trajectories: Sequence[PartialTrajectory]
    ) -> "TransitionMiniBatch":
        r"""Constructs mini-batch from list of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            Mini-batch of trajectories.
        """
        observations = np.concatenate([traj.observations for traj in trajectories], axis=0)
        actions = np.concatenate([traj.actions for traj in trajectories], axis=0)
        rewards = np.concatenate([traj.rewards for traj in trajectories], axis=0)
        terminals = np.concatenate([traj.terminals for traj in trajectories], axis=0)
        behavior_policy = np.concatenate([traj.behavior_policy for traj in trajectories], axis=0)
        intervals = np.concatenate([traj.terminals for traj in trajectories], axis=0)

        # Get next observations and next behavior policy
        is_terminal = (terminals == 1).squeeze()
        next_observations = np.roll(observations, -1, axis=0)
        next_observations[is_terminal] = 0

        next_behavior_policy = np.roll(behavior_policy, -1, axis=0)
        next_behavior_policy[is_terminal] = 0
        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            behavior_policy=cast_recursively(behavior_policy, np.float32),
            next_behavior_policy=cast_recursively(next_behavior_policy, np.float32),
        )

@dataclasses.dataclass(frozen=True)
class TrajectoryMiniBatch:
    r"""Mini-batch of trajectories.

    Args:
        observations: Batched sequence of observations.
        actions: Batched sequence of actions.
        rewards: Batched sequence of rewards.
        returns_to_go: Batched sequence of returns-to-go.
        terminals: Batched sequence of environment terminal flags.
        timesteps: Batched sequence of environment timesteps.
        masks: Batched masks that represent padding.
        length: Length of trajectories.
    """
    observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, L, ...)
    actions: np.ndarray  # (B, L, ...)
    rewards: np.ndarray  # (B, L, 1)
    returns_to_go: np.ndarray  # (B, L, 1)
    terminals: np.ndarray  # (B, L, 1)
    timesteps: np.ndarray  # (B, L)
    masks: np.ndarray  # (B, L)
    length: int
    behavior_policy: np.ndarray  # (B, L, A)

    def __post_init__(self) -> None:
        assert check_dtype(self.observations, np.float32)
        assert check_dtype(self.actions, np.float32)
        assert check_dtype(self.rewards, np.float32)
        assert check_dtype(self.returns_to_go, np.float32)
        assert check_dtype(self.terminals, np.float32)
        assert check_dtype(self.timesteps, np.float32)
        assert check_dtype(self.masks, np.float32)
        assert check_dtype(self.behavior_policy, np.float32)

    @classmethod
    def from_partial_trajectories(
        cls, trajectories: Sequence[PartialTrajectory]
    ) -> "TrajectoryMiniBatch":
        r"""Constructs mini-batch from list of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            Mini-batch of trajectories.
        """
        observations = stack_observations(
            [traj.observations for traj in trajectories]
        )
        actions = np.stack([traj.actions for traj in trajectories], axis=0)
        rewards = np.stack([traj.rewards for traj in trajectories], axis=0)
        returns_to_go = np.stack(
            [traj.returns_to_go for traj in trajectories], axis=0
        )
        terminals = np.stack([traj.terminals for traj in trajectories], axis=0)
        timesteps = np.stack([traj.timesteps for traj in trajectories], axis=0)
        masks = np.stack([traj.masks for traj in trajectories], axis=0)
        behavior_policy = np.stack([traj.behavior_policy for traj in trajectories], axis=0)
        return TrajectoryMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            returns_to_go=cast_recursively(returns_to_go, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            timesteps=cast_recursively(timesteps, np.float32),
            masks=cast_recursively(masks, np.float32),
            length=trajectories[0].length,
            behavior_policy=cast_recursively(behavior_policy, np.float32),
        )

    @property
    def observation_shape(self) -> Shape:
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]  # type: ignore

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]  # type: ignore

    def __len__(self) -> int:
        return int(self.actions.shape[0])
    