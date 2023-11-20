import copy
import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....metrics.divergence import DivergenceMetricMixin
from ....models.torch import (
    CategoricalPolicy,
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    EnsembleQFunction,
    Parameter,
    Policy,
)
from ....torch_utility import TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase
from .ddpg_impl import DDPGBaseImpl
from .utility import DiscreteQFunctionMixin

__all__ = ["DiscreteBRACImpl"]

class DiscreteBRACImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _policy: CategoricalPolicy
    _q_func: EnsembleDiscreteQFunction
    _targ_q_func: EnsembleDiscreteQFunction
    _log_temp: Optional[Parameter]
    _actor_optim: Optimizer
    _critic_optim: Optimizer
    _temp_optim: Optimizer
    _behavior_model: Optional[torch.nn.Module] = None
    _policy_alpha: Optional[Parameter]
    _value_alpha: Optional[Parameter]
    _divergence_metric: Optional[DivergenceMetricMixin]
    _divergence_threshold: float
    

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        policy: CategoricalPolicy,
        log_temp: Parameter,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        gamma: float,
        device: str,
        policy_alpha: Parameter,
        value_alpha: Parameter,
        divergence_metric: Optional[DivergenceMetricMixin],
        divergence_threshold: float,

    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._gamma = gamma
        self._q_func = q_func
        self._policy = policy
        self._log_temp = log_temp
        self._policy_alpha = policy_alpha
        self._value_alpha = value_alpha
        self._divergence_metric = divergence_metric
        self._divergence_threshold = divergence_threshold
        self._actor_optim = actor_optim
        self._critic_optim = critic_optim
        self._temp_optim = temp_optim
        self._targ_q_func = copy.deepcopy(q_func)

    @train_api
    def update_critic(self, batch: TorchMiniBatch) -> float:
        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return float(loss.cpu().detach().numpy())
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.next_observations)
            probs = log_probs.exp()
            entropy = self.compute_entropy(log_probs)
            target = self._targ_q_func.compute_target(batch.next_observations)
            # ! value_penalty uses next states !
            policy_divergence = self.divergence(batch.next_observations, log_probs)
            # if self._divergence_threshold > 0:
            #     # policy_divergence = 1 if divergence is above threshold, 0 otherwise
            #     policy_divergence = (policy_divergence > self._divergence_threshold).float()
            value_penalty =  - policy_divergence * self._value_alpha()
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            
            # Probs, target, entropy all have shape (batch_size, n_actions)
            # Target = Q(s, *) for a row in batch
            # Probs = pi(*|s) for a row in batch
            # Entropy = H(pi(*|s)) for a row in batch
            # The final target will have a shape (batch_size, 1)
            # This represents a weighted loss

            # Adding value_penalty is not exactly the same as described in the theoretical part in the paper
            # In the paper: compare (r + gamma * target - alpha * divergence) with q_pred with MSELoss
            # In the BRAC implementation: target += value_penalty where value_penalty < 0, value_penalty = - alpha * divergence
            # This comes down to compare: (r + gamma * (target - alpha * divergence)) with q_pred with MSELoss
            # I don't think it matters for optimization purpose of policy
            return  (probs * (target - entropy + value_penalty)).sum(dim=1, keepdim=keepdims) 

    def compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    @train_api
    def update_actor(self, batch: TorchMiniBatch) -> float:
        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)
        loss.backward()
        self._actor_optim.step()

        return float(loss.cpu().detach().numpy())
    
    def divergence(self, states, log_probs) -> float:
        # Compute the brac policy regularization
        # Uses alpha config parameter
        # Returns a positive value which we want to minimize (i.e. make zero)
        # target_probs = torch.Tensor([1.0, 0.0]).to(self._device)
        if self._behavior_model is None:
            return torch.zeros_like(log_probs)
        target_probs = torch.from_numpy(self._behavior_model.predict_proba(states.cpu().numpy())).to(self._device)

        # print(torch.concat([probs, target_probs], dim=1))
        divergence = self._divergence_metric(log_probs, target_probs).reshape(-1, 1)

        # divergence is an array of shape (batch_size, 1) or (batch_size, 2)
        # If it is (batch_size, 1) then it is the same penalty for all actions
        # If it is (batch_size, 2) then it is a different penalty for each action
        # They will get summed but are reweighted by the probs so there is a slight difference
        # Get behavior probs
        return divergence
    


    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func(batch.observations, reduction="min")
        log_probs = self._policy.log_probs(batch.observations)
        probs = log_probs.exp()
        policy_divergence = self.divergence(batch.observations, log_probs)
        # if self._divergence_threshold > 0:
        #     # How to threshold the divergence?

        #     # (1) policy_divergence = 1 if divergence is above threshold, 0 otherwise
        #     # policy_divergence = (policy_divergence > self._divergence_threshold).float()

        #     # (2) relu: What does is if policy_divergence < 0.1 then then penalty becomes 0
        #     # policy_divergence > 0.1 then then penalty becomes divergence -0.1
        #     policy_divergence = torch.nn.functional.relu(policy_divergence - self._divergence_threshold)

        #     # (3) MSE between target divergence and threshold
        #     # policy_divergence = (self._divergence_threshold - policy_divergence).pow(2)

        policy_regularization = policy_divergence * self._policy_alpha().exp()
        entropy = self.compute_entropy(log_probs)

        ### Question
        # We can add the polcy_regularization term at two places
        # 1. (probs * (entropy - q_t + policy_regularization)).sum(dim=1).mean()
        # 2. (probs * (entropy - q_t)) + policy_regularization).sum(dim=1).mean()
        # I think (1) aligns better with the implementation of the BRAC paper

        # Remember that a negative loss is better (lower loss is better)
        # So we want to minimize policy_regularization
        # IMPORTANT: policy_regularization is a positive value
        # value_penalty is a negative value
        return (probs * (- q_t + entropy  + policy_regularization)).sum(dim=1).mean()#, policy_divergence.mean()

    @train_api
    def update_temp(self, batch: TorchMiniBatch) -> Tuple[float, float]:
        self._temp_optim.zero_grad()

        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.observations)
            probs = log_probs.exp()
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return float(loss.cpu().detach().numpy()), float(cur_temp)
    
    @train_api
    def update_alpha(self, batch: TorchMiniBatch) -> Tuple[float, float]:
        """
        Update alpha parameter of BRAC.
        Currently broken
        """
        self._alpha_optim.zero_grad()

        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.observations)
            divergence = self.divergence(batch.observations, log_probs)
            divergence_diff = self._divergence_threshold - divergence

        loss = (self._policy_alpha().exp() * divergence_diff).mean()

        loss.backward()
        self._alpha_optim.step()

        # current alpha value
        cur_alpha = self._policy_alpha().exp().cpu().detach().numpy()[0][0]

        return float(loss.cpu().detach().numpy()), float(cur_alpha)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.best_action(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.sample(x)

    def update_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    
    def compute_entropy(self, log_probs):
        if self._log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = self._log_temp().exp()
        return temp * log_probs


    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._actor_optim

    @property
    def q_function(self) -> EnsembleQFunction:
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        return self._critic_optim

