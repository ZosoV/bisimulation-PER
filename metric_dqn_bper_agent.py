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

"""DQN Agent with MICo loss."""

import collections
import functools
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from absl import logging


# Own scripts
# from mico.atari import metric_utils
import metric_utils
import eval_utils
import networks
import custom_replay_buffer
import pretrained_metric_dqn

@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          mico_weight, distance_fn, loss_weights, bper_weight = 0):
  """Run the training step."""
  def loss_fn(params, bellman_target, target_r, target_next_r, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    q_values = model_output.q_values
    q_values = jnp.squeeze(q_values)
    representations = model_output.representation
    representations = jnp.squeeze(representations)

    # NOTE: target online representation
    # NOTE: I can try to use this instead of target_next_r
    # tmp_model_output = jax.vmap(q_online)(next_states)
    # online_next_r = tmp_model_output.representation
    # online_next_r = jnp.squeeze(online_next_r)

    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    batch_bellman_loss = jax.vmap(losses.mse_loss)(bellman_target,
                                                      replay_chosen_q)
    bellman_loss = jnp.mean(loss_multipliers * batch_bellman_loss)
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * bellman_loss +
            mico_weight * metric_loss)
    
    # Current vs Next Distance without squarify
    # NOTE: I could try to use online next representation instead of target_next_r
    # NOTE: when bper_weight = 0 we are using PER and we don't need to compute the experience distance
    
    if bper_weight > 0:
      experience_distances = metric_utils.current_next_distances(
        current_state_representations=representations,
        next_state_representations=target_next_r, # online_next_r,
        distance_fn = distance_fn,)
    else:
      experience_distances = jnp.zeros_like(batch_bellman_loss)

    return jnp.mean(loss), (bellman_loss, metric_loss, batch_bellman_loss, experience_distances)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target,
                                           target_r, target_next_r, loss_weights)
  bellman_loss, metric_loss, batch_bellman_loss, experience_distances = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, bellman_loss, metric_loss, batch_bellman_loss, experience_distances


def target_outputs(target_network, states, next_states, rewards, terminals,
                   cumulative_gamma):
  """Compute the target Q-value."""
  curr_state_representation = jax.vmap(target_network, in_axes=(0))(
      states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  next_state_output = jax.vmap(target_network, in_axes=(0))(next_states)
  next_state_q_vals = next_state_output.q_values
  next_state_q_vals = jnp.squeeze(next_state_q_vals)
  next_state_representation = next_state_output.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  replay_next_qt_max = jnp.max(next_state_q_vals, 1)
  return (
      jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                            (1. - terminals)),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))

@gin.configurable
class MetricDQNBPERAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with the MICo loss."""

  def __init__(self, 
               num_actions, 
               summary_writer=None,
               mico_weight=0.01, 
               distance_fn=metric_utils.cosine_distance,
               replay_scheme='uniform',
               bper_weight=0, # PER: 0 and BPER: 1
               method_scheme='scaling', # 'softmax', 'softmax_weight'
               log_dynamics_stats=False,
               log_replay_buffer_stats=False,
               batch_size_statistics=1024,   #512, # 256,
               game_name=None,
               fixed_agent_ckpt=None
               ):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    self._method_scheme = method_scheme
    self.num_actions = num_actions

    self._log_dynamics_stats = log_dynamics_stats
    self._log_replay_buffer_stats = log_replay_buffer_stats
    self._batch_size_statistics = batch_size_statistics

    self.game_name = game_name
    self._fixed_pretrained_agent = self._create_fixed_agent(fixed_agent_ckpt)
    
    self._exponential_normalizer = metric_utils.ExponentialNormalizer()

    network = networks.AtariDQNNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)
    
    self._replay_scheme = replay_scheme
    self._bper_weight = bper_weight
    self._replay = self._build_replay_buffer()
    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t mico_weight: %f', self._mico_weight)
    logging.info('\t distance_fn: %s', self._distance_fn)
    logging.info('\t replay_scheme: %s', self._replay_scheme)
    logging.info('\t bper_weight: %f', bper_weight)
    logging.info('\t method_scheme: %s', self._method_scheme)
    logging.info('\t gamma: %f', self.gamma)
    logging.info('\t update_horizon: %f', self.update_horizon)
    logging.info('\t min_replay_history: %d', self.min_replay_history)
    logging.info('\t update_period: %d', self.update_period)
    logging.info('\t target_update_period: %d', self.target_update_period)
    logging.info('\t epsilon_train: %f', self.epsilon_train)
    logging.info('\t epsilon_eval: %f', self.epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', self.epsilon_decay_period)
    logging.info('\t optimizer: %s', self.optimizer)
    logging.info('\t seed: %d', self._seed)
    logging.info('\t loss_type: %s', self._loss_type)
    logging.info('\t preprocess_fn: %s', self.preprocess_fn)
    logging.info('\t summary_writing_frequency: %d', self.summary_writing_frequency)
    logging.info('\t allow_partial_reload: %s', self.allow_partial_reload)

  def _create_fixed_agent(self, fixed_agent_ckpt):
    
    if fixed_agent_ckpt is None:
      return None

    agent = pretrained_metric_dqn.create_agent(
          num_actions = self.num_actions,
          agent_name='metric_dqn', 
        )

    # NOTE: Change this according to where the pretrained agent is saved
    pretrained_metric_dqn.reload_checkpoint(agent, fixed_agent_ckpt.format(self.game_name))

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
    if self._replay_scheme == 'uniform':
      # NOTE: set a fixed uniform distribution for sampling uniformly from the experience replay
      sampling_distribution = samplers.UniformSamplingDistribution(
          seed=self._seed
      )
    elif self._replay_scheme == 'prioritized':
      sampling_distribution = samplers.PrioritizedSamplingDistribution(
              seed=self._seed
        )
    return custom_replay_buffer.CustomReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
        seed = self._seed, # Seed  fot the fixed uniform distribution
    )

  def _sample_batch_for_statistics(self, batch_size=None, uniform=True):
    """Sample elements from the replay buffer."""
    tmp_replay_elements = collections.OrderedDict()
    if batch_size is None:
      if uniform:
        elems, metadata = self._replay.sample_uniform(size = self._batch_size_statistics,
                                              with_sample_metadata=True)
      else:
        elems, metadata = self._replay.sample(size = self._batch_size_statistics,
                                              with_sample_metadata=True)
    else:
      if uniform:
        elems, metadata = self._replay.sample_uniform(size = batch_size,
                                              with_sample_metadata=True)
      else:
        elems, metadata = self._replay.sample(size = batch_size,
                                              with_sample_metadata=True)
    tmp_replay_elements['state'] = elems.state
    tmp_replay_elements['next_state'] = elems.next_state
    tmp_replay_elements['action'] = elems.action
    tmp_replay_elements['reward'] = elems.reward
    tmp_replay_elements['terminal'] = elems.is_terminal
    if self._replay_scheme == 'prioritized':
      tmp_replay_elements['indices'] = metadata.keys

    return tmp_replay_elements

  def _get_outputs(self, 
                   agent_id = "fixed", # "fixed" or "online" or "target"
                   curr_states = True,
                   next_states = False,
                   intermediates = True,
                   batch = None):
      if batch is None:
        batch = self._sample_batch_for_statistics()  # Non-JIT part
      
      if agent_id == "fixed":
        # NOTE: I can use the fixed agent to get the features
        if curr_states:
          batch['output'] =  eval_utils.get_features(
              self._fixed_pretrained_agent.network_def,
              self._fixed_pretrained_agent.online_params,
              batch['state'],
              intermediates
          )
        if next_states:
          batch['output_next']  =  eval_utils.get_features(
              self._fixed_pretrained_agent.network_def,
              self._fixed_pretrained_agent.online_params,
              batch['next_state'],
              intermediates
          )
      elif agent_id == "online":
        if curr_states:
          batch["output"] =  eval_utils.get_features(
              self.network_def,
              self.online_params,
              batch['state'],
              intermediates
          )
        if next_states:
          batch["output_next"] =  eval_utils.get_features(
              self.network_def,
              self.online_params,
              batch['next_state'],
              intermediates
          )
      elif agent_id == "target":
        if curr_states:
          batch["output"] =  eval_utils.get_features(
              self.network_def,
              self.target_network_params,
              batch['state'],
              intermediates
          )
        if next_states:
          batch["output_next"] =  eval_utils.get_features(
              self.network_def,
              self.target_network_params,
              batch['next_state'],
              intermediates
          )
          
      return batch


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
          # NOTE: they don't divide to N (size of buffer) because many optimizer 
          # are scale invariant as Adam
          # NOTE: they use sqrt(prob) instead of prob because they practically
          # are setting beta = 0.5.
          # normaly the value should (1/P)^beta => 1/sqrt(P) considering beta = 0.5
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        (self.optimizer_state, self.online_params,
         loss, bellman_loss, metric_loss, 
         batch_bellman_loss, experience_distances) = train(
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
             self.cumulative_gamma,
             self._mico_weight,
             self._distance_fn,
             loss_weights,
             self._bper_weight)
        
        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake
          # we leave it as is here, using the more direct sqrt(). Taking the
          # square root "makes sense", as we are dealing with a squared loss.
          # Add a small nonzero value to the loss to avoid 0 priority items.
          # While technically this may be okay, setting all items to 0
          # priority will cause troubles, and also result in 1.0 / 0.0 = NaN
          # correction terms.

          # NOTE: Option we can in the same way as the loss weights, use the sqrt of the
          # experience distance.
          # priorities = (1 - self._bper_weight) * jnp.sqrt(loss + 1e-10) + self._bper_weight * jnp.sqrt(experience_distances + 1e-10)
          
          batch_td_error = jnp.sqrt(batch_bellman_loss + 1e-10)

          # IDEA 3: Exponential weighted average
          if self._method_scheme == 'exponential_norm':
            experience_distances = self._exponential_normalizer.normalize(experience_distances)
            priorities = (1 - self._bper_weight) * batch_td_error + self._bper_weight * experience_distances # experience_distances
          else:
            experience_distances = experience_distances / jnp.sqrt(15488) 
            # NOTE: 15488 is the size of the representation when using the ataria network with input size 84x84x4        
            if self._method_scheme == 'scaling':
              priorities = (1 - self._bper_weight) * batch_td_error + self._bper_weight * experience_distances # experience_distances
            elif self._method_scheme == 'softmax_weight':
              experience_distances = jax.nn.softmax(experience_distances)
              # IDEA 1: Experimental method: Reweighing the priorities based on the experience distances
              priorities = batch_td_error * experience_distances
            elif self._method_scheme == 'softmax':
              experience_distances = jax.nn.softmax(experience_distances)
              # IDEA 2: Assigning the priorities relative to the softmax of the experience distances in the batch
              priorities = (1 - self._bper_weight) * batch_td_error + self._bper_weight * experience_distances # experience_distances
            elif self._method_scheme == 'td_weights':
              priorities = jax.nn.softmax(batch_td_error) * jax.nn.softmax(experience_distances)


          self._replay.update(
              self.replay_elements['indices'],
              priorities=priorities,
          )

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):

            with self.summary_writer.as_default():

                # NOTE: When I already have a well chose strategy for the experience distance
                # I can comment this part
                # Log the statistics as scalars
                if self._replay_scheme == 'prioritized':
                  pass                  
                tf.summary.scalar('Losses/Aggregate', loss, step=self.training_steps * 4)
                tf.summary.scalar('Losses/Bellman', bellman_loss, step=self.training_steps * 4)
                tf.summary.scalar('Losses/Metric', metric_loss, step=self.training_steps * 4)

                stats = collections.OrderedDict()
                    

                  
                if self._log_replay_buffer_stats:
                  
                  # NOTE: The representations are calculated using the fixed agent at 
                  # checkpoint 99 and using a uniform sample from the current
                  # replay buffer
                  
                  # NOTE: To approximate better the distribution of the bisimulation distance and euclidean distance
                  # I can sample several times batch until cover at 1% of the size of the replay buffer
                  # 20 minibatches of 512 will be 10240 samples
                  # 10 minibatches of 1024 will be 10240 samples
                  # 4 minibatches of 1024 will be 4096 samples

                  num_batches = 4 
                  batches_collected = []
                  for i in range(num_batches):
                    sampled_batch = self._sample_batch_for_statistics()
                    batches_collected.append(sampled_batch)

                  prioritized_batches_collected = []
                  for i in range(num_batches):
                    sampled_batch = self._sample_batch_for_statistics(uniform=False)
                    prioritized_batches_collected.append(sampled_batch)

                  metric_stats = eval_utils.RunningStats()
                  euclidean_stats = eval_utils.RunningStats()
                  td_residuals_stats = eval_utils.RunningStats()

                  for sampled_batch in batches_collected:
                    eval_batch = self._get_outputs(agent_id='fixed', 
                                                  next_states=True, 
                                                  intermediates=False,
                                                  batch=sampled_batch)
                  
                    curr_outputs, next_outputs = eval_batch['output'], eval_batch['output_next']

                    metric_distances = metric_utils.current_next_distances(
                        curr_outputs.representation, next_outputs.representation, self._distance_fn)
                    metric_stats.update_running_stats(metric_distances)

                    euclidean_distances = jnp.linalg.norm(curr_outputs.representation - next_outputs.representation, axis=1)
                    euclidean_stats.update_running_stats(euclidean_distances)

                    residual_diffs = jnp.linalg.norm(curr_outputs.representation - self.cumulative_gamma * next_outputs.representation, axis=1)
                    td_residuals_stats.update_running_stats(residual_diffs)

                  stats['Stats/BisimulationDistanceAvg'] = metric_stats.mean
                  stats['Stats/BisimulationDistanceStd'] = jnp.sqrt(metric_stats.variance)

                  stats['Stats/EuclideanDistanceAvg'] = euclidean_stats.mean
                  stats['Stats/EuclideanDistanceStd'] = jnp.sqrt(euclidean_stats.variance)

                  stats['Stats/TD-Residuals'] = td_residuals_stats.mean
                  stats['Stats/TD-ResidualsStd'] = jnp.sqrt(td_residuals_stats.variance)


                  prioritized_metric_stats = eval_utils.RunningStats()
                  prioritized_euclidean_stats = eval_utils.RunningStats()
                  prioritized_td_residuals_stats = eval_utils.RunningStats()

                  for sampled_batch in prioritized_batches_collected:
                    eval_batch = self._get_outputs(agent_id='fixed', 
                                                  next_states=True, 
                                                  intermediates=False,
                                                  batch=sampled_batch)
                  
                    curr_outputs, next_outputs = eval_batch['output'], eval_batch['output_next']

                    prioritized_metric_distances = metric_utils.current_next_distances(
                        curr_outputs.representation, next_outputs.representation, self._distance_fn)
                    prioritized_metric_stats.update_running_stats(prioritized_metric_distances)

                    prioritized_euclidean_distances = jnp.linalg.norm(curr_outputs.representation - next_outputs.representation, axis=1)
                    prioritized_euclidean_stats.update_running_stats(prioritized_euclidean_distances)

                    prioritized_residual_diffs = jnp.linalg.norm(curr_outputs.representation - self.cumulative_gamma * next_outputs.representation, axis=1)
                    prioritized_td_residuals_stats.update_running_stats(prioritized_residual_diffs)

                  stats['Stats/PrioritizedBisimulationDistanceAvg'] = prioritized_metric_stats.mean
                  stats['Stats/PrioritizedBisimulationDistanceStd'] = jnp.sqrt(prioritized_metric_stats.variance)
                  stats['Stats/PrioritizedEuclideanDistanceAvg'] = prioritized_euclidean_stats.mean
                  stats['Stats/PrioritizedEuclideanDistanceStd'] = jnp.sqrt(prioritized_euclidean_stats.variance)
                  stats['Stats/PrioritizedTD-ResidualsAvg'] = prioritized_td_residuals_stats.mean
                  stats['Stats/PrioritizedTD-ResidualsStd'] = jnp.sqrt(prioritized_td_residuals_stats.variance)

                  td_errors_stats = eval_utils.RunningStats()
                  online_metric_stats = eval_utils.RunningStats()
                  online_td_residuals_stats = eval_utils.RunningStats()
                  online_euclidean_stats = eval_utils.RunningStats()
                  for sampled_batch in batches_collected:
                    eval_batch = self._get_outputs(agent_id='online',
                                            next_states=True,
                                            intermediates=False,
                                            batch=sampled_batch)

                    td_errors = eval_utils.log_td_errors(
                        eval_batch,
                        self.network_def,
                        self.target_network_params,
                        self.cumulative_gamma
                        )
                    td_errors_stats.update_running_stats(td_errors)

                    online_metric_distances = metric_utils.current_next_distances(
                        eval_batch['output'].representation,
                        eval_batch['output_next'].representation,
                        self._distance_fn)
                    online_metric_stats.update_running_stats(online_metric_distances)

                    online_residuals_diff = jnp.linalg.norm(eval_batch['output'].representation - self.cumulative_gamma * eval_batch['output_next'].representation, axis=1)
                    online_td_residuals_stats.update_running_stats(online_residuals_diff)

                    online_euclidean_distances = jnp.linalg.norm(eval_batch['output'].representation - eval_batch['output_next'].representation, axis=1)
                    online_euclidean_stats.update_running_stats(online_euclidean_distances)

                  stats['Stats/TD-ErrorAvg'] = td_errors_stats.mean
                  stats['Stats/TD-ErrorStd'] = jnp.sqrt(td_errors_stats.variance)


                  stats['Stats/OnlineBisimulationDistanceAvg'] = online_metric_stats.mean
                  stats['Stats/OnlineBisimulationDistanceStd'] = jnp.sqrt(online_metric_stats.variance)
                  stats['Stats/OnlineTD-ResidualsAvg'] = online_td_residuals_stats.mean
                  stats['Stats/OnlineTD-ResidualsStd'] = jnp.sqrt(online_td_residuals_stats.variance)
                  stats['Stats/OnlineEuclideanDistanceAvg'] = online_euclidean_stats.mean
                  stats['Stats/OnlineEuclideanDistanceStd'] = jnp.sqrt(online_euclidean_stats.variance)

                  prioritized_td_errors_stats = eval_utils.RunningStats()
                  prioritized_online_metric_stats = eval_utils.RunningStats()
                  prioritized_online_td_residuals_stats = eval_utils.RunningStats()
                  prioritized_online_euclidean_stats = eval_utils.RunningStats()
                  for sampled_batch in prioritized_batches_collected:
                    eval_batch = self._get_outputs(agent_id='online',
                                            next_states=True,
                                            intermediates=False,
                                            batch=sampled_batch)

                    prioritized_td_errors = eval_utils.log_td_errors(
                        eval_batch,
                        self.network_def,
                        self.target_network_params,
                        self.cumulative_gamma
                        )
                    prioritized_td_errors_stats.update_running_stats(prioritized_td_errors)

                    prioritized_online_metric_distances = metric_utils.current_next_distances(
                        eval_batch['output'].representation,
                        eval_batch['output_next'].representation,
                        self._distance_fn)
                    prioritized_online_metric_stats.update_running_stats(prioritized_online_metric_distances)

                    prioritized_online_residuals_diff = jnp.linalg.norm(eval_batch['output'].representation - self.cumulative_gamma * eval_batch['output_next'].representation, axis=1)
                    prioritized_online_td_residuals_stats.update_running_stats(prioritized_online_residuals_diff)

                    prioritized_online_euclidean_distances = jnp.linalg.norm(eval_batch['output'].representation - eval_batch['output_next'].representation, axis=1)
                    prioritized_online_euclidean_stats.update_running_stats(prioritized_online_euclidean_distances)
                  stats['Stats/PrioritizedTD-ErrorAvg'] = prioritized_td_errors_stats.mean
                  stats['Stats/PrioritizedTD-ErrorStd'] = jnp.sqrt(prioritized_td_errors_stats.variance)
                  stats['Stats/PrioritizedOnlineBisimulationDistanceAvg'] = prioritized_online_metric_stats.mean
                  stats['Stats/PrioritizedOnlineBisimulationDistanceStd'] = jnp.sqrt(prioritized_online_metric_stats.variance)
                  stats['Stats/PrioritizedOnlineTD-ResidualsAvg'] = prioritized_online_td_residuals_stats.mean
                  stats['Stats/PrioritizedOnlineTD-ResidualsStd'] = jnp.sqrt(prioritized_online_td_residuals_stats.variance)
                  stats['Stats/PrioritizedOnlineEuclideanDistanceAvg'] = prioritized_online_euclidean_stats.mean
                  stats['Stats/PrioritizedOnlineEuclideanDistanceStd'] = jnp.sqrt(prioritized_online_euclidean_stats.variance)

        
                if self._log_dynamics_stats:
                  eval_batch = self._get_outputs(agent_id='online',
                                            next_states=False, 
                                            intermediates=True,
                                            batch=batches_collected[0])
                  stats['Stats/DormantPercentage'] = eval_utils.log_dormant_percentage(
                    eval_batch["output"][1])
                  
                  eval_batch = self._get_outputs(agent_id='online',
                                            next_states=False, 
                                            intermediates=True,
                                            batch=prioritized_batches_collected[0])
                  stats['Stats/PrioritizedDormantPercentage'] = eval_utils.log_dormant_percentage(
                    eval_batch["output"][1])

                for key, value in stats.items():
                    tf.summary.scalar(key, value, step=self.training_steps * 4)

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1


