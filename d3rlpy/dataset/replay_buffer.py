from typing import Any, BinaryIO, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import torch

from .buffers import BufferProtocol, FIFOBuffer, InfiniteBuffer
from .components import (
    Episode,
    EpisodeBase,
    PartialTrajectory,
    Signature,
    Transition,
)
from .episode_generator import EpisodeGeneratorProtocol
from .io import dump, load
from .mini_batch import TrajectoryMiniBatch, TransitionMiniBatch
from .trajectory_slicers import BasicTrajectorySlicer, TrajectorySlicerProtocol, EntireTrajectorySlicer
from .transition_pickers import BasicTransitionPicker, TransitionPickerProtocol
from .types import Observation
from .writers import (
    BasicWriterPreprocess,
    ExperienceWriter,
    WriterPreprocessProtocol,
)

__all__ = [
    "ReplayBuffer",
    "create_fifo_replay_buffer",
    "create_infinite_replay_buffer",
]


class ReplayBuffer:
    r"""Replay buffer for experience replay.

    This replay buffer implementation is used for both online and offline
    training in d3rlpy. To determine shapes of observations, actions and
    rewards, one of ``episodes``, ``env`` and signatures must be provided.

    .. code-block::

        from d3rlpy.dataset import FIFOBuffer, ReplayBuffer, Signature

        buffer = FIFOBuffer(limit=1000000)

        # initialize with pre-collected episodes
        replay_buffer = ReplayBuffer(buffer=buffer, episodes=<episodes>)

        # initialize with Gym
        replay_buffer = ReplayBuffer(buffer=buffer, env=<env>)

        # initialize with manually specified signatures
        replay_buffer = ReplayBuffer(
            buffer=buffer,
            observation_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            action_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            reward_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
        )

    Args:
        buffer (d3rlpy.dataset.BufferProtocol): Buffer implementation.
        transition_picker (Optional[d3rlpy.dataset.TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[d3rlpy.dataset.TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor (Optional[d3rlpy.dataset.WriterPreprocessProtocol]):
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        episodes (Optional[Sequence[d3rlpy.dataset.EpisodeBase]]):
            List of episodes to initialize replay buffer.
        env (Optional[gym.Env]): Gym environment to extract shapes of
            observations and action.
        observation_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of observation.
        action_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of action.
        reward_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of reward.
        cache_size (int): Size of cache to record active episode history used
            for online training. ``cache_size`` needs to be greater than the
            maximum possible episode length.
    """
    _buffer: BufferProtocol
    _transition_picker: TransitionPickerProtocol
    _trajectory_slicer: TrajectorySlicerProtocol
    _writer: ExperienceWriter
    _episodes: List[EpisodeBase]

    def __init__(
        self,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
        episodes: Optional[Sequence[EpisodeBase]] = None,
        env: Optional[gym.Env[np.ndarray, Any]] = None,
        observation_signature: Optional[Signature] = None,
        action_signature: Optional[Signature] = None,
        reward_signature: Optional[Signature] = None,
        cache_size: int = 10000,
    ):
        transition_picker = transition_picker or BasicTransitionPicker()
        trajectory_slicer = trajectory_slicer or EntireTrajectorySlicer()
        writer_preprocessor = writer_preprocessor or BasicWriterPreprocess()

        if not (
            observation_signature and action_signature and reward_signature
        ):
            if episodes:
                observation_signature = episodes[0].observation_signature
                action_signature = episodes[0].action_signature
                reward_signature = episodes[0].reward_signature
            elif env:
                observation_signature = Signature(
                    dtype=[env.observation_space.dtype],
                    shape=[env.observation_space.shape],  # type: ignore
                )
                action_signature = Signature(
                    dtype=[env.action_space.dtype],
                    shape=[env.action_space.shape],  # type: ignore
                )
                reward_signature = Signature(
                    dtype=[np.dtype(np.float32)],
                    shape=[[1]],
                )
            else:
                raise ValueError(
                    "Either episodes or env must be provided for signatures"
                )

        self._buffer = buffer
        self._writer = ExperienceWriter(
            buffer,
            writer_preprocessor,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=cache_size,
        )
        self._transition_picker = transition_picker
        self._trajectory_slicer = trajectory_slicer

        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        r"""Appends observation, action and reward to buffer.

        Args:
            observation: Observation.
            action: Action.
            reward: Reward.
        """
        self._writer.write(observation, action, reward)

    def append_episode(self, episode: EpisodeBase) -> None:
        r"""Appends episode to buffer.

        Args:
            episode: Episode.
        """
        for i in range(episode.transition_count):
            self._buffer.append(episode, i)

    def clip_episode(self, terminated: bool) -> None:
        r"""Clips current episode.

        Args:
            terminated: Flag to represent environmental termination. This flag
                should be ``False`` if the episode is terminated by timeout.
        """
        self._writer.clip_episode(terminated)

    def sample_transition(self) -> Transition:
        r"""Samples a transition.

        Returns:
            Transition.
        """
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        r"""Samples a mini-batch of transitions.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Mini-batch.
        """
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        r"""Samples a partial trajectory.

        Args:
            length: Length of partial trajectory.

        Returns:
            Partial trajectory.
        """
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
        return self._trajectory_slicer(episode, transition_index, length)
    
    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        r"""Samples a mini-batch of partial trajectories.

        Args:
            batch_size: Mini-batch size.
            length: Length of partial trajectories.

        Returns:
            Mini-batch.
        """
        return TrajectoryMiniBatch.from_partial_trajectories(
            [self.sample_trajectory(length) for _ in range(batch_size)]
        )
    
    def sample_trajectory_transition_batch(
        self, batch_size: int, length: int
    ) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_partial_trajectories(
            [self.sample_trajectory(length) for _ in range(batch_size)]
        )

    def dump(self, f: BinaryIO) -> None:
        """Dumps buffer data.

        .. code-block:: python

            with open('dataset.h5', 'wb') as f:
                replay_buffer.dump(f)

        Args:
            f: IO object to write to.
        """
        dump(self._buffer.episodes, f)

    @classmethod
    def from_episode_generator(
        cls,
        episode_generator: EpisodeGeneratorProtocol,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        """Builds ReplayBuffer from episode generator.

        Args:
            episode_generator: Episode generator implementation.
            buffer: Buffer implementation.
            transition_picker: Transition picker implementation for
                Q-learning-based algorithms.
            trajectory_slicer: Trajectory slicer implementation for
                Transformer-based algorithms.
            writer_preprocessor: Writer preprocessor implementation.

        Returns:
            Replay buffer.
        """
        return cls(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
        )

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        """Builds ReplayBuffer from dumped data.

        This method reconstructs replay buffer dumped by ``dump`` method.

        .. code-block:: python

            with open('dataset.h5', 'rb') as f:
                replay_buffer = ReplayBuffer.load(f, buffer)

        Args:
            f: IO object to read from.
            buffer: Buffer implementation.
            episode_cls: Eisode class used to reconstruct data.
            transition_picker: Transition picker implementation for
                Q-learning-based algorithms.
            trajectory_slicer: Trajectory slicer implementation for
                Transformer-based algorithms.
            writer_preprocessor: Writer preprocessor implementation.

        Returns:
            Replay buffer.
        """
        return cls(
            buffer,
            episodes=load(episode_cls, f),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
        )

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        """Returns sequence of episodes.

        Returns:
            Sequence of episodes.
        """
        return self._buffer.episodes

    def size(self) -> int:
        """Returns number of episodes.

        Returns:
            Number of episodes.
        """
        return len(self._buffer.episodes)

    @property
    def buffer(self) -> BufferProtocol:
        """Returns buffer.

        Returns:
            Buffer.
        """
        return self._buffer

    @property
    def transition_count(self) -> int:
        """Returns number of transitions.

        Returns:
            Number of transitions.
        """
        return self._buffer.transition_count

    @property
    def transition_picker(self) -> TransitionPickerProtocol:
        """Returns transition picker.

        Returns:
            Transition picker.
        """
        return self._transition_picker

    @property
    def trajectory_slicer(self) -> TrajectorySlicerProtocol:
        """Returns trajectory slicer.

        Returns:
            Trajectory slicer.
        """
        return self._trajectory_slicer


def create_fifo_replay_buffer(
    limit: int,
    episodes: Optional[Sequence[EpisodeBase]] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    env: Optional[gym.Env[np.ndarray, Any]] = None,
) -> ReplayBuffer:
    """Builds FIFO replay buffer.

    This function is a shortcut alias to build replay buffer with
    ``FIFOBuffer``.

    Args:
        limit: Maximum capacity of FIFO buffer.
        episodes: List of episodes to initialize replay buffer.
        transition_picker:
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer:
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor:
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        env: Gym environment to extract shapes of observations and action.

    Returns:
        Replay buffer.
    """
    buffer = FIFOBuffer(limit)
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        env=env,
    )


def create_infinite_replay_buffer(
    episodes: Optional[Sequence[EpisodeBase]] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    env: Optional[gym.Env[np.ndarray, Any]] = None,
) -> ReplayBuffer:
    """Builds infinite replay buffer.

    This function is a shortcut alias to build replay buffer with
    ``InfiniteBuffer``.

    Args:
        episodes: List of episodes to initialize replay buffer.
        transition_picker:
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer:
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor:
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        env: Gym environment to extract shapes of observations and action.

    Returns:
        Replay buffer.
    """
    buffer = InfiniteBuffer()
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        env=env,
    )

class ReplayBufferGPU:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        behavior_policy: Optional[np.ndarray] = None,
    ):
        self.replay_buffer = replay_buffer

        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.reshape(-1, 1).astype(np.float32)
        self.terminals = terminals.astype(np.float32)
        # Transition picker does this when is_terminal == 1:
        # next_observation = create_zero_observation(observation)
        # next_behavior_policy = np.zeros_like(behavior_policy)
        self.behavior_policy = behavior_policy.astype(np.float32)

        self.next_observations = np.roll(observations, -1, axis=0)
        self.next_observations[self.terminals == 1] = 0
        self.next_observations  = self.next_observations.astype(np.float32)

        self.next_behavior_policy = np.roll(behavior_policy, -1, axis=0)
        self.next_behavior_policy[self.terminals == 1] = 0
        self.next_behavior_policy = self.next_behavior_policy.astype(np.float32)

        self.terminals = self.terminals.reshape(-1, 1)
        self.intervals = np.ones_like(self.terminals)
        self._device = "cuda:0"

        self.observations = torch.from_numpy(self.observations).to(self._device)
        self.actions = torch.from_numpy(self.actions).to(self._device)
        self.rewards = torch.from_numpy(self.rewards).to(self._device)
        self.terminals = torch.from_numpy(self.terminals).to(self._device)
        self.intervals = torch.from_numpy(self.intervals).to(self._device)
        self.behavior_policy = torch.from_numpy(self.behavior_policy).to(self._device)
        self.next_observations = torch.from_numpy(self.next_observations).to(self._device)
        self.next_behavior_policy = torch.from_numpy(self.next_behavior_policy).to(self._device)

        from d3rlpy.torch_utility import TorchMiniBatch
        self.TorchMiniBatch = TorchMiniBatch


    def sample_transition_batch(self, batch_size: int) -> "TorchMiniBatch":
        r"""Samples a torch mini-batch of transitions.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Mini-batch.
        """
        idx = np.random.randint(self.observations.shape[0], size=batch_size)
        batch = self.TorchMiniBatch(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            terminals=self.terminals[idx],
            intervals=self.intervals[idx],
            behavior_policy=self.behavior_policy[idx],
            next_behavior_policy=self.next_behavior_policy[idx],
            device=self._device,
        )
        return batch
    
    def sample_transition(self):
        return self.replay_buffer.sample_transition()
    
    @property
    def episodes(self):
        return self.replay_buffer.episodes
