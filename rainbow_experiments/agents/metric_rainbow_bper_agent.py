# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rainbow Agent with the MICo loss."""

import collections
import functools
from absl import logging

from dopamine.jax import losses
from dopamine.jax.agents.rainbow import rainbow_agent
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers

# from mico.atari import metric_utils

import agents.metric_utils as metric_utils
import models.networks as networks
import agents.pretrained_agent as pretrained_agent
import replay_memory.custom_replay_buffer as custom_replay_buffer
import utils.eval_utils as eval_utils


@functools.partial(jax.jit, static_argnums=(0, 3, 12, 13, 14, 15))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, loss_weights,
          support, cumulative_gamma, mico_weight, distance_fn, bper_weight = 0):
  """Run a training step."""
  def loss_fn(params, bellman_target, loss_multipliers, target_r,
              target_next_r):
    def q_online(state):
      return network_def.apply(params, state, support)

    model_output = jax.vmap(q_online)(states)
    logits = model_output.logits
    logits = jnp.squeeze(logits)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    c51_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        bellman_target,
        chosen_action_logits)
    c51_loss *= loss_multipliers
    online_dist, norm_average, angular_distance = (
        metric_utils.representation_distances(
            representations, target_r, distance_fn,
            return_distance_components=True))
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * c51_loss +
            mico_weight * metric_loss)
    
    if bper_weight > 0:
      experience_distances = metric_utils.current_next_distances(
        current_state_representations=representations,
        next_state_representations=target_next_r, # online_next_r,
        distance_fn = distance_fn,)
    else:
      experience_distances = jnp.zeros_like(c51_loss)

    aux_losses = {
        'loss': loss,
        'mean_loss': jnp.mean(loss),
        'c51_loss': jnp.mean(c51_loss),
        'metric_loss': metric_loss,
        'norm_average': jnp.mean(norm_average),
        'angular_distance': jnp.mean(angular_distance),
        'experience_distances': experience_distances,
    }
    return jnp.mean(loss), aux_losses

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_distribution(
      q_target,
      states,
      next_states,
      rewards,
      terminals,
      support,
      cumulative_gamma)
  (_, aux_losses), grad = grad_fn(online_params, bellman_target,
                                  loss_weights, target_r, target_next_r)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, aux_losses


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None, None))
def target_distribution(target_network, states, next_states, rewards, terminals,
                        support, cumulative_gamma):
  """Builds the C51 target distribution as per Bellemare et al. (2017)."""
  curr_state_representation = target_network(states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  probabilities = jnp.squeeze(next_state_target_outputs.probabilities)
  next_probabilities = probabilities[next_qt_argmax]
  next_state_representation = next_state_target_outputs.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  return (
      jax.lax.stop_gradient(rainbow_agent.project_distribution(
          target_support, next_probabilities, support)),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@gin.configurable
class MetricRainbowBPERAgent(rainbow_agent.JaxRainbowAgent):
  """Rainbow Agent with the MICo loss."""

  def __init__(self, 
               num_actions, 
               summary_writer=None,
               mico_weight=0.01, 
               distance_fn=metric_utils.cosine_distance,
               method_scheme="scaling",
               log_replay_buffer_stats=False,
               batch_size_statistics=1024,
               eval_agent_ckpt=None,
               bper_weight=0):
    
    if bper_weight == 0:
      logging.info("Creating MetricRainbowBPERAgent with PER.")
    else:
      logging.info(f"Creating MetricRainbowBPERAgent with:")
      logging.info(f"  - bper_weight: {bper_weight}")
      logging.info(f"  - method_scheme: {method_scheme}")
    
    
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    network = networks.AtariRainbowNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)
    self._bper_weight = bper_weight
    self._method_scheme = method_scheme
    self.num_actions = num_actions

    self._eval_pretrained_agent = self._create_eval_agent(eval_agent_ckpt)
    self._log_replay_buffer_stats = log_replay_buffer_stats and (self._eval_pretrained_agent is not None)
    self._batch_size_statistics = batch_size_statistics

    # NOTE: I need to call again build_replay_buffer to set the prioritized version
    # coz the JaxDQNAgent set by default the uniform replay buffer
    self._replay_scheme = 'prioritized'
    self._replay = self._build_replay_buffer()

  def _create_eval_agent(self, fixed_agent_ckpt):
    
    if fixed_agent_ckpt is None:
      return None

    agent = pretrained_agent.create_agent(
          num_actions = self.num_actions,
          agent_name='metric_rainbow', 
        )

    # NOTE: Change this according to where the pretrained agent is saved
    pretrained_agent.reload_checkpoint(agent, fixed_agent_ckpt.format(self.game_name))

    return agent

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))

    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )
    sampling_distribution = samplers.PrioritizedSamplingDistribution(
        seed=self._seed
    )
    return custom_replay_buffer.CustomReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
        seed = self._seed,
    )
  
  def _sample_batch_for_statistics(self, batch_size=None):
    """Sample elements from the replay buffer."""
    tmp_replay_elements = collections.OrderedDict()
    if batch_size is None:
      elems, metadata = self._replay.sample_uniform(size = self._batch_size_statistics,
                                            with_sample_metadata=True)
    else:
      elems, metadata = self._replay.sample_uniform(size = batch_size,
                                            with_sample_metadata=True)

    tmp_replay_elements['state'] = elems.state
    tmp_replay_elements['next_state'] = elems.next_state
    tmp_replay_elements['action'] = elems.action
    tmp_replay_elements['reward'] = elems.reward
    tmp_replay_elements['terminal'] = elems.is_terminal
    if self._replay_scheme == 'prioritized':
      tmp_replay_elements['indices'] = metadata.keys

    return tmp_replay_elements


  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, aux_losses = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._support,
            self.cumulative_gamma,
            self._mico_weight,
            self._distance_fn,
            self._bper_weight)

        loss = aux_losses.pop('loss')
        experience_distances = aux_losses.pop('experience_distances')
        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.

          loss_priorities = jnp.sqrt(loss + 1e-10)
          experience_distances = experience_distances / jnp.sqrt(15488)

          if self._method_scheme == "scaling":
            priorities = (1 - self._bper_weight) * loss_priorities + self._bper_weight * experience_distances # experience_distances
          elif self._method_scheme == "softmax":
            experience_distances = jax.nn.softmax(experience_distances)
            priorities = (1 - self._bper_weight) * loss_priorities + self._bper_weight * experience_distances # experience_distances

          # NOTE: setting self._bper_weight to 0 will result in the original Rainbow with PER
          self._replay.update(self.replay_elements['indices'],
                              priorities = priorities)


        # TODO: Set the logging to save the stats during training
        if (self.summary_writer is not None and
                    self.training_steps > 0 and
                    self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            for key, value in aux_losses.items():
              tf.summary.scalar(f'Losses/{key}', value,
                                step=self.training_steps * 4)
            
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1