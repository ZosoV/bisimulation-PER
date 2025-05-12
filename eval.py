from absl import app
from absl import flags
from absl import logging
import pathlib
import os.path as osp
import gin
from dopamine.discrete_domains import atari_lib
import collections

import pretrained_metric_dqn
import metric_dqn_bper_agent
import numpy as np
import eval_utils
import functools
import jax
import jax.numpy as jnp
import metric_utils
from dopamine.jax import losses
import tensorflow as tf
import os

AGENTS = [
    'metric_dqn', 'metric_dqn_bper', 'metric_dqn_per',
    'metric_dqn_bper_scaling', 'metric_dqn_bper_softmax',
]

flags.DEFINE_string('base_dir', "logs/",
                    'Base directory to host all required sub-directories.')
flags.DEFINE_enum('agent_name', "metric_dqn", AGENTS, 'Name of the agent.')
flags.DEFINE_string('checkpoint_dir', "checkpoints/Alien/metric_dqn/118398/", 'Checkpoint path to use')
flags.DEFINE_string('game_name', 'Alien', 'Name of game')
flags.DEFINE_string('seed', "118398",
                    'Random seed to use for the experiment.')


flags.DEFINE_multi_string(
    'gin_files', ["eval_metric_dqn.gin"], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@functools.partial(jax.jit, static_argnums=(0, 8, 9, 10, 12))
def step_forward(network_def, online_params, target_params,
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
    batch_metric_loss = jax.vmap(losses.huber_loss)(online_dist, target_dist)
    metric_loss = jnp.mean(batch_metric_loss)

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

    return jnp.mean(loss), (batch_bellman_loss, batch_metric_loss, experience_distances)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = metric_dqn_bper_agent.target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target,
                                           target_r, target_next_r, loss_weights)
  
  batch_bellman_loss, batch_metric_loss, experience_distances = component_losses

  return loss, batch_bellman_loss, batch_metric_loss, experience_distances, grad

def main(unused_argv):
  _ = unused_argv
  logging.set_verbosity(logging.INFO)
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)
  
#   paths = list(pathlib.Path(FLAGS.checkpoint_dir).parts)
#   run_number = paths[-1].split('_')[-1]

  ckpt_dir = osp.join(FLAGS.checkpoint_dir, 'checkpoints')
  logging.info('Checkpoint directory: %s', ckpt_dir)

  # Create the environment and agent.
  logging.info('Game: %s', FLAGS.game_name)
  environment = atari_lib.create_atari_environment(
      game_name=FLAGS.game_name, sticky_actions=True)
  
  base_dir = FLAGS.base_dir
  experiment_dir = os.path.join(base_dir, FLAGS.game_name, FLAGS.agent_name, FLAGS.seed)
  summary_writer = experiment_dir

  # NOTE: I didn't use the create_agent from pretrained because I need
   # to also load the replay buffer in this case
  agent = metric_dqn_bper_agent.MetricDQNBPERAgent(
        num_actions=environment.action_space.n, 
        summary_writer=summary_writer,
        game_name=FLAGS.game_name,)

  checkpoints = pretrained_metric_dqn.get_checkpoints(ckpt_dir, max_checkpoints=100)

  # TODO: Load experience fixed experience replay buffer
  agent._replay.load(ckpt_dir, iteration_number=1)

  # Collect batches for statistics
  # NOTE: This will be fixed along the whole experiment
  # and only the outputs will change
  num_batches = 2 # 2 * 1024 (1024 is the batch size)
  batches_collected = []
  for i in range(num_batches):
    sampled_batch = agent._sample_batch_for_statistics()
    batches_collected.append(sampled_batch)

  for idx, checkpoint in enumerate(checkpoints):
    # Load the checkpoint.
    pretrained_metric_dqn.reload_checkpoint(agent, checkpoint)
    logging.info('Checkpoint loaded: %s', checkpoint)
    logging.info('Calculating statistics...')

    stats = collections.OrderedDict()

    features = []
    residuals = []
    batch_bellman_losses = []
    batch_metric_losses = []
    experience_distances_list = []
    grads = []

    for sampled_batch in batches_collected:
      # NOTE: it adds the 'output' key to the sampled batch
      # using the online network
      sampled_batch = agent._get_outputs(agent_id='online',
                                      next_states=False, 
                                      intermediates=True,
                                      batch=sampled_batch)
      # NOTE: it adds the 'output_next' key to the sampled batch
      # using the target network
      sampled_batch = agent._get_outputs(agent_id='target',
                                      curr_states=False,
                                      next_states=True,
                                      intermediates=False,
                                      batch=sampled_batch)
      
      
      loss_weights = jnp.ones(sampled_batch['state'].shape[0])

      (loss, 
      batch_bellman_loss, 
      batch_metric_loss, 
      experience_distances, 
      grad) = step_forward(
          agent.network_def,
          agent.online_params,
          agent.target_network_params,
          sampled_batch['state'],
          sampled_batch['action'],
          sampled_batch['next_state'],
          sampled_batch['reward'],
          sampled_batch['terminal'],
          agent.cumulative_gamma,
          agent._mico_weight,
          agent._distance_fn,
          loss_weights,
          agent._bper_weight
        )
      
      features.append(sampled_batch['output'][0].representation)
      residuals.append(sampled_batch['output'][0].representation - agent.cumulative_gamma * sampled_batch['output_next'].representation)
      batch_bellman_losses.append(batch_bellman_loss)
      batch_metric_losses.append(batch_metric_loss)
      experience_distances_list.append(experience_distances)
      grads.append(grad)


    # Concatenate the features
    features_matrix = jnp.concatenate(features, axis=0)
    residuals_matrix = jnp.concatenate(residuals, axis=0)
    batch_bellman_losses_array = jnp.concatenate(batch_bellman_losses, axis=0)
    batch_metric_losses_array = jnp.concatenate(batch_metric_losses, axis=0)
    experience_distances_array = jnp.concatenate(experience_distances_list, axis=0)

    # Log srank
    stats["Eval/Srank"] = eval_utils.log_srank(features_matrix)

    # Log dormant neurons NOTE: it only uses the last batch
    stats["Eval/DormantPercentage"] = eval_utils.log_dormant_percentage(sampled_batch['output'][1])

    # Log td-Residuals norm
    stats["Eval/TD-Residuals"] = eval_utils.log_avg_norm(residuals_matrix)

    # Log representation norm
    stats["Eval/RepresentationNorm"] = eval_utils.log_avg_norm(features_matrix)

    # Log the loss variance
    stats["Eval/LossVarianceBellmanLoss"] = jnp.var(batch_bellman_losses_array)
    stats["Eval/LossVarianceMetricLoss"] = jnp.var(batch_metric_losses_array)

    # Log the average bisimulation distance
    stats["Eval/AverageBisimulationDistance"] = jnp.mean(experience_distances_array)
    stats["Eval/StdBisimulationDistance"] = jnp.var(experience_distances_array)

    with agent.summary_writer.as_default():
      for key, value in stats.items():
        tf.summary.scalar(key, value, step=(idx + 1) * 1000_000)
      
      agent.summary_writer.flush()
    
if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  app.run(main)