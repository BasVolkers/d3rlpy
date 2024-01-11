import copy
import math
from typing import List, Optional, Tuple

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
    _log_alpha: Parameter
    _actor_optim: Optimizer
    _critic_optim: Optimizer
    _temp_optim: Optimizer
    _alpha_optim: Optimizer
    _policy_alpha: Optional[Parameter]
    _warmup_alpha: Optional[Parameter]
    _value_alpha: Optional[Parameter]
    _divergence_metric: Optional[DivergenceMetricMixin]
    _divergence_threshold: float
    _max_alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        policy: CategoricalPolicy,
        log_temp: Parameter,
        log_alpha: Parameter,
        actor_optim: Optimizer,
        actor_warmup_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        alpha_optim: Optimizer,
        gamma: float,
        device: str,
        policy_alpha: Parameter,
        value_alpha: Parameter,
        divergence_metric: Optional[DivergenceMetricMixin],
        divergence_threshold: float,
        max_alpha: float,
        warmup_alpha: Optional[Parameter],
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
        self._log_alpha = log_alpha
        self._policy_alpha = policy_alpha
        self._warmup_alpha = warmup_alpha
        self._value_alpha = value_alpha
        self._divergence_metric = divergence_metric
        self._divergence_threshold = divergence_threshold
        self._actor_optim = actor_optim
        self._actor_warmup_optim = actor_warmup_optim
        self._critic_optim = critic_optim
        self._temp_optim = temp_optim
        self._alpha_optim = alpha_optim
        self._targ_q_func = copy.deepcopy(q_func)
        self._max_alpha = math.log(max_alpha)

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
            entropy = self.entropy_regularization(log_probs)
            target = self._targ_q_func.compute_target(batch.next_observations)
            if self._value_alpha() != 0:
                # ! value_penalty uses next states !
                policy_divergence = self.divergence(batch, batch.next_behavior_policy, log_probs)
                value_penalty =  self._value_alpha() * -policy_divergence
                value_penalty[batch.terminals == 1] = 0
            else:
                value_penalty = torch.zeros_like(target)
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
    def warmup_actor(self, batch: TorchMiniBatch) -> float:
        self._actor_warmup_optim.zero_grad()

        # Compute only divergence loss
        log_probs = self._policy.log_probs(batch.observations)
        policy_divergence = self.divergence(batch, batch.behavior_policy, log_probs)
        if self._warmup_alpha is None:
            alpha = self._log_alpha().exp()
        else:
            alpha = self._warmup_alpha
        policy_regularization = alpha * policy_divergence
        
        entropy_regularization = self.entropy_regularization(log_probs)
        # Loss is just the pr part
        loss = policy_regularization.mean() + entropy_regularization.mean()
        loss.backward()
        self._actor_warmup_optim.step()

        return float(loss.cpu().detach().numpy())

    @train_api
    def update_actor(self, batch: TorchMiniBatch) -> float:
        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss, pr, q_loss = self.compute_actor_loss(batch)
        loss.backward()
        self._actor_optim.step()

        return float(loss.cpu().detach().numpy()), float(pr.cpu().detach().numpy()), float(q_loss.cpu().detach().numpy())
    
    def divergence(self, batch, target_probs, log_probs) -> float:
        # Compute the brac policy regularization
        # Uses alpha config parameter
        # Returns a positive value which we want to minimize (i.e. make zero)
        # target_probs = torch.Tensor([1.0, 0.0]).to(self._device)
        # if self._behavior_model is None:
        #     return torch.zeros_like(log_probs)
        # target_probs = torch.from_numpy(self._behavior_model.predict_proba(states.cpu().numpy())).to(self._device)
        # target_probs = batch.behavior_policy

        # print(torch.concat([probs, target_probs], dim=1))
        if self._divergence_metric.use_dataset_actions():
            # Select only probabilities for actions taken in the batch
            log_probs = log_probs.gather(1, batch.actions.long())#.squeeze()
            target_probs = target_probs.gather(1, batch.actions.long())#.squeeze()

        divergence = self._divergence_metric(batch, log_probs, target_probs).reshape(-1, 1)

        # # divergence is an array of shape (batch_size, 1) or (batch_size, 2)
        # # If it is (batch_size, 1) then it is the same penalty for all actions
        # # If it is (batch_size, 2) then it is a different penalty for each action
        # # They will get summed but are reweighted by the probs so there is a slight difference
        # # Get behavior probs
        return divergence
    
    def policy_regularization(self, batch, log_probs) -> float:
        if self._log_alpha is not None:
            policy_divergence = self.divergence(batch, batch.behavior_policy, log_probs)

            # if self._divergence_threshold > 0:
            #     # Penalty for divergence 0.1 should be same as penalty for divergence 0.0
            #     # Penalty for divergence 0.2 should be higher then penalty for divergence 0.1policy_divergence = policy_divergence - self._divergence_threshold
            #     policy_divergence = policy_divergence - self._divergence_threshold
            #     policy_divergence = policy_divergence.clamp(min=0)

            policy_regularization = self._log_alpha().exp() * policy_divergence 
        else:
            policy_regularization = torch.zeros_like(log_probs)
        return policy_regularization


    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func(batch.observations, reduction="min")
        log_probs = self._policy.log_probs(batch.observations)
        probs = log_probs.exp()

            # if self._divergence_threshold > 0:
            #     # How to threshold the divergence?

            #     # (1) policy_divergence = 1 if divergence is above threshold, 0 otherwise
            #     # policy_divergence = (policy_divergence > self._divergence_threshold).float()

            #     # (2) relu: What does is if policy_divergence < 0.1 then then penalty becomes 0
            #     # policy_divergence > 0.1 then then penalty becomes divergence -0.1
            #     policy_divergence = torch.nn.functional.relu(policy_divergence - self._divergence_threshold)

            #     # (3) MSE between target divergence and threshold
            #     # policy_divergence = (self._divergence_threshold - policy_divergence).pow(2)

        policy_regularization = self.policy_regularization(batch, log_probs)
        entropy = self.entropy_regularization(log_probs)

        ### Question
        # We can add the polcy_regularization term at two places
        # 1. (probs * (entropy - q_t + policy_regularization)).sum(dim=1).mean()
        # 2. (probs * (entropy - q_t)) + policy_regularization).sum(dim=1).mean()
        # I think (1) aligns better with the implementation of the BRAC paper

        # Remember that a negative loss is better (lower loss is better)
        # So we want to minimize policy_regularization
        # IMPORTANT: policy_regularization is a positive value
        # value_penalty is a negative value
        return (probs * (- q_t + entropy  + policy_regularization)).sum(dim=1).mean(), policy_regularization.mean(), (probs * -q_t).mean()
        # return (probs * (- q_t + entropy)).sum(dim=1).mean() + policy_regularization.mean(), policy_regularization.mean(), (probs * -q_t).mean()

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
            divergence = self.divergence(batch, batch.behavior_policy, log_probs)
            # Compute divergence difference as current - target
            divergence_diff = divergence - self._divergence_threshold

        # Only penalize if divergence is above threshold
        loss = self._log_alpha().exp() * -divergence_diff.mean()#.clamp(min=0)
        # When divergence is negative, flip the sign of the loss
        loss.backward()
        self._alpha_optim.step()

        # self._log_alpha.data.clamp_(-5, math.log(1000.0))
        self._log_alpha.data.clamp_(-5, self._max_alpha)

        # current alpha value
        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return float(loss.cpu().detach().numpy()), float(cur_alpha)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.best_action(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.sample(x)

    def update_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    def entropy(self, log_probs):
        return log_probs
    
    def entropy_regularization(self, log_probs):
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
    
class DiscreteBRACImplMultiDivergence(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _policy: CategoricalPolicy
    _q_func: EnsembleDiscreteQFunction
    _targ_q_func: EnsembleDiscreteQFunction
    _log_temp: Optional[Parameter]
    _log_alphas: List[Parameter]
    _actor_optim: Optimizer
    _critic_optim: Optimizer
    _temp_optim: Optimizer
    _alpha_optims: List[Optimizer]
    _divergence_metrics: List[DiscreteQFunctionMixin]
    _divergence_thresholds: List[float]
    _max_alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        policy: CategoricalPolicy,
        log_temp: Parameter,
        log_alphas: List[Parameter],
        actor_optim: Optimizer,
        actor_warmup_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        alpha_optims: List[Optimizer],
        gamma: float,
        device: str,
        divergence_metrics: List[DivergenceMetricMixin],
        divergence_thresholds: List[float],
        max_alpha: float,
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
        self._log_alphas = log_alphas
        self._divergence_metrics = divergence_metrics
        self._divergence_thresholds = divergence_thresholds
        self._actor_optim = actor_optim
        self._actor_warmup_optim = actor_warmup_optim
        self._critic_optim = critic_optim
        self._temp_optim = temp_optim
        self._alpha_optims = alpha_optims
        self._targ_q_func = copy.deepcopy(q_func)
        self._max_alpha = math.log(max_alpha)

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
            entropy = self.entropy_regularization(log_probs)
            target = self._targ_q_func.compute_target(batch.next_observations)
            value_penalty = torch.zeros_like(target)
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
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
    def warmup_actor(self, batch: TorchMiniBatch) -> float:
        self._actor_warmup_optim.zero_grad()

        # Compute only divergence loss
        log_probs = self._policy.log_probs(batch.observations)
        policy_regularization = self.policy_regularization(batch, log_probs)
        entropy_regularization = self.entropy_regularization(log_probs)
        # Loss is just the pr part
        loss = policy_regularization.mean() + entropy_regularization.mean()
        loss.backward()
        self._actor_warmup_optim.step()

        return float(loss.cpu().detach().numpy())

    @train_api
    def update_actor(self, batch: TorchMiniBatch) -> float:
        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss, pr, q_loss = self.compute_actor_loss(batch)
        loss.backward()
        self._actor_optim.step()

        return float(loss.cpu().detach().numpy()), float(pr.cpu().detach().numpy()), float(q_loss.cpu().detach().numpy())
    
    def divergence(self, idx, batch, target_probs, log_probs) -> float:
        divergence_metric = self._divergence_metrics[idx]
        if divergence_metric.use_dataset_actions():
            # Select only probabilities for actions taken in the batch
            log_probs = log_probs.gather(1, batch.actions.long())#.squeeze()
            target_probs = target_probs.gather(1, batch.actions.long())#.squeeze()

        divergence = divergence_metric(batch, log_probs, target_probs).reshape(-1, 1)

        return divergence
    
    def policy_regularization(self, batch, log_probs) -> float:
        policy_regularization = torch.zeros_like(log_probs)
        for idx in range(len(self._divergence_metrics)):
            policy_divergence = self.divergence(idx, batch, batch.behavior_policy, log_probs)
            policy_regularization += self._log_alphas[idx]().exp() * policy_divergence 
        return policy_regularization


    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func(batch.observations, reduction="min")
        log_probs = self._policy.log_probs(batch.observations)
        probs = log_probs.exp()
        policy_regularization = self.policy_regularization(batch, log_probs)
        entropy = self.entropy_regularization(log_probs)
        return (probs * (- q_t + entropy  + policy_regularization)).sum(dim=1).mean(), policy_regularization.mean(), (probs * -q_t).mean()

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
    def update_alphas(self, batch: TorchMiniBatch) -> Tuple[float, float]:
        """
        Update alpha parameter of BRAC.
        """

        with torch.no_grad():
            log_probs = self._policy.log_probs(batch.observations)
        
        cur_alphas = []
        losses = []
        for idx in range(len(self._divergence_metrics)):
            if self._divergence_thresholds[idx] is None:
                continue
            self._alpha_optims[idx].zero_grad()

            divergence = self.divergence(idx, batch, batch.behavior_policy, log_probs)
            # Compute divergence difference as current - target
            divergence_diff = divergence - self._divergence_thresholds[idx]

            # Only penalize if divergence is above threshold
            loss = self._log_alphas[idx]().exp() * -divergence_diff.mean()#.clamp(min=0)
            loss.backward()
            self._alpha_optims[idx].step()

            # self._log_alpha.data.clamp_(-5, math.log(1000.0))
            self._log_alphas[idx].data.clamp_(-5, self._max_alpha)

            # current alpha values
            cur_alpha = self._log_alphas[idx]().exp().cpu().detach().numpy()[0][0]

            losses.append(float(loss.cpu().detach().numpy()))
            cur_alphas.append(float(cur_alpha))

        return losses, cur_alphas

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.best_action(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy.sample(x)

    def update_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    def entropy(self, log_probs):
        return log_probs
    
    def entropy_regularization(self, log_probs):
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
    
class BRACImpl(DDPGBaseImpl):
    _log_temp: Parameter
    _temp_optim: Optimizer
    _log_alpha: Parameter
    _alpha_optim: Optimizer
    _policy_alpha: Parameter
    _value_alpha: Parameter
    _divergence_metric: Optional[DivergenceMetricMixin]
    _divergence_threshold: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: Policy,
        q_func: EnsembleContinuousQFunction,
        log_temp: Parameter,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        gamma: float,
        tau: float,
        device: str,
        log_alpha: Parameter,
        alpha_optim: Optimizer,
        policy_alpha: Parameter,
        value_alpha: Parameter,
        divergence_metric: Optional[DivergenceMetricMixin],
        divergence_threshold: float,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._log_temp = log_temp
        self._temp_optim = temp_optim
        self._log_alpha = log_alpha
        self._alpha_optim = alpha_optim
        self._policy_alpha = policy_alpha
        self._value_alpha = value_alpha
        self._divergence_metric = divergence_metric
        self._divergence_threshold = divergence_threshold

    def compute_entropy(self, log_probs):
        if self._log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = self._log_temp().exp()
        return temp * log_probs
    
    def policy_regularization(self, batch, log_probs) -> float:
        if self._log_alpha is not None:
            policy_divergence = self.divergence(batch, batch.behavior_policy, log_probs)
            policy_regularization = self._log_alpha().exp() * (policy_divergence - self._divergence_threshold)
            # Penalty for divergence 0.1 should be same as penalty for divergence 0.0
            # Penalty for divergence 0.2 should be higher then penalty for divergence 0.1
            # policy_regularization = torch.nn.functional.relu(policy_regularization)
        else:
            policy_regularization = torch.zeros_like(log_probs)
        return policy_regularization
    
    def divergence(self, batch, target_probs, log_probs) -> float:
        divergence = self._divergence_metric(batch, log_probs, target_probs).reshape(-1, 1)
        return divergence

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self.compute_entropy(log_prob)
        q_t = self._q_func(batch.observations, action, "min")
        actor_loss = (entropy - q_t).mean()

        div_loss = self.policy_regularization(batch, log_prob).mean()
        return actor_loss + div_loss
    
    @train_api
    def warmup_actor(self, batch: TorchMiniBatch) -> float:
        # Compute only divergence loss
        self._actor_optim.zero_grad()

        # log_probs = self._policy.log_probs(batch.observations)
        # dist = self._policy.dist(batch.observations)
        # log_probs = dist.log_prob(batch.actions)
        # _, log_probs = self._policy.sample_n_with_log_prob(batch.observations, 1)
        h = self._policy._encoder(batch.observations)
        mu = self._policy._mu(h)
        logstd = self._policy._logstd(h)
        dist = torch.distributions.Normal(mu, logstd.exp())
        log_probs = dist.log_prob(batch.actions)
        loss = -log_probs.mean()

        # loss = (mu - batch.actions).pow(2).mean()
        # clipped_logstd = self._policy._compute_logstd(h)
        # dist = torch.distributions.Normal(mu, clipped_logstd.exp())
        # log_probs = dist.log_prob(batch.actions)

        # policy_regularization = self.policy_regularization(batch, log_probs)
        # Loss is just the pr part
        # loss = policy_regularization.mean()
        loss.backward()
        self._actor_optim.step()

        return float(loss.cpu().detach().numpy())

    @train_api
    def update_temp(self, batch: TorchMiniBatch) -> Tuple[float, float]:
        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob = self._policy.sample_with_log_prob(batch.observations)
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return float(loss.cpu().detach().numpy()), float(cur_temp)
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self.compute_entropy(log_prob)
            target = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy