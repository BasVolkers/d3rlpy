import dataclasses
import math
from typing import Dict

import torch

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_discrete_q_function,
    create_parameter,
    create_squashed_normal_policy,
)
from ...metrics.divergence import DivergenceMetricFactory
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.brac_impl import DiscreteBRACImpl

__all__ = ["DiscreteBRACConfig", "DiscreteBRAC"]


@dataclasses.dataclass()
class DiscreteBRACConfig(LearnableConfig):
    r"""Config of Behavior Regularized Actor Critic algorithm for discrete action-space.

    This discrete version of BRAC is built based on the discrete version of SAC.

    It changes how the critic target and the actor loss are computed.

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        policy_alpha (float): Strength of policy regularization.
        value_alpha (float): Strength of value regularization.
        divergence_metric (str): Name of divergence metric.
    """
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 64
    gamma: float = 0.99
    n_critics: int = 2
    initial_temperature: float = 1.0
    target_update_interval: int = 8000
    policy_alpha: float = 0.0
    value_alpha: float = 0.0
    divergence_metric: str = None
    divergence_threshold: float = 0.0

    def create(self, device: DeviceArg = False) -> "DiscreteBRAC":
        return DiscreteBRAC(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_brac"


class DiscreteBRAC(QLearningAlgoBase[DiscreteBRACImpl, DiscreteBRACConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        q_func = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        policy = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        if self._config.initial_temperature == 0 and self._config.temp_learning_rate == 0:
            log_temp = None
        else:
            log_temp = create_parameter(
                (1, 1),
                math.log(self._config.initial_temperature),
                device=self._device,
            )

        policy_alpha = create_parameter(
            (1, 1),
            math.log(self._config.policy_alpha),
            device=self._device,
        )
        value_alpha = create_parameter(
            (1, 1),
            self._config.value_alpha,
            device=self._device,
        )

        divergence_metric = None
        if self._config.divergence_metric:
            divergence_metric = DivergenceMetricFactory(self._config.divergence_metric).create()

        critic_optim = self._config.critic_optim_factory.create(
            q_func.parameters(), lr=self._config.critic_learning_rate
        )
        actor_optim = self._config.actor_optim_factory.create(
            policy.parameters(), lr=self._config.actor_learning_rate
        )

        if log_temp is not None:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.parameters(), lr=self._config.temp_learning_rate
            )
        else:
            temp_optim = None


        self._impl = DiscreteBRACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            policy=policy,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            gamma=self._config.gamma,
            device=self._device,
            policy_alpha=policy_alpha,
            value_alpha=value_alpha,
            divergence_metric=divergence_metric,
            divergence_threshold=self._config.divergence_threshold,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}
        
        # lagrangian parameter update for SAC temeprature
        if self._config.temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        # lagrangian parametr update for alpha
        # if self._config.alpha_learning_rate > 0:
        #     alpha_loss, alpha = self._impl.update_alpha(batch)
        #     metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        # Compute divergence for metrics
        with torch.no_grad():
            log_probs = self._impl._policy.log_probs(batch.observations)
            divergence = self._impl.divergence(batch.observations, log_probs).mean()
            divergence = float(divergence.cpu().detach().numpy())
            metrics.update({"div": divergence})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE

    ### TODO: Do we need this method?
    # def policy_predict(self, x) -> np.ndarray:
    #     assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
    #     # assert check_non_1d_array(x), "Input must have batch dimension."

    #     # TODO: support tuple inputs
    #     torch_x = cast(
    #         torch.Tensor, convert_to_torch_recursively(x, self._device)
    #     )

    #     with torch.no_grad():
    #         if self._config.observation_scaler:
    #             torch_x = self._config.observation_scaler.transform(torch_x)

    #         action = self._impl.policy(torch_x)

    #         if self._config.action_scaler:
    #             action = self._config.action_scaler.reverse_transform(action)

    #     return action.cpu().detach().numpy()


register_learnable(DiscreteBRACConfig)
