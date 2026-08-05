import functools
import warnings
from typing import Optional, Sequence, Tuple, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from rl.datasets.dataset import Batch
from rl.networks import critic_net, policies
from rl.networks.common import InfoDict, Model, Params, PRNGKey


def _tree_is_finite(tree) -> bool:
    """Return whether every numeric leaf in a JAX pytree is finite."""
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _optimizer(
        learning_rate: float,
        max_grad_norm: Optional[float]
) -> optax.GradientTransformation:
    transforms = []
    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.adam(learning_rate=learning_rate))
    return optax.chain(*transforms)


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update_temperature(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info


def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           backup_entropy: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SACLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 alpha: Literal['auto'] | float = 'auto',
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 max_grad_norm: Optional[float] = 10.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=_optimizer(actor_lr, max_grad_norm))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=_optimizer(critic_lr, max_grad_norm))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=_optimizer(temp_lr, max_grad_norm))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1
        self.skipped_updates = 0
        self.nonfinite_action_count = 0
        self._warned_about_nonfinite_action = False
        self._warned_about_skipped_update = False

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       deterministic: bool = False) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature,
                                               distribution=(
                                                   'det' if deterministic
                                                   else 'log_prob'
                                               ))
        self.rng = rng

        actions = np.asarray(actions)
        if not np.all(np.isfinite(actions)):
            self.nonfinite_action_count += 1
            if not self._warned_about_nonfinite_action:
                warnings.warn(
                    "SAC produced a non-finite action; replacing it with a "
                    "bounded fallback action. Check skipped_updates and the "
                    "training losses.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_about_nonfinite_action = True
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if not _tree_is_finite(batch):
            return self._skip_update("the replay batch contains non-finite values")

        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0)

        self.rng = new_rng
        candidate_state = (
            new_actor.params,
            new_actor.opt_state,
            new_critic.params,
            new_critic.opt_state,
            new_target_critic.params,
            new_temp.params,
            new_temp.opt_state,
            info,
        )
        if not _tree_is_finite(candidate_state):
            self.step -= 1
            return self._skip_update("the optimizer produced non-finite state")

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return {**info, 'update_skipped': 0.0}

    def _skip_update(self, reason: str) -> InfoDict:
        """Keep the last finite learner state when an update is unsafe."""
        self.skipped_updates += 1
        if not self._warned_about_skipped_update:
            warnings.warn(
                f"Skipping SAC update because {reason}. The last finite "
                "learner state will be retained.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_about_skipped_update = True
        return {'update_skipped': 1.0}
