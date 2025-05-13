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
import numpy as onp
import eval_utils
import functools
import jax
import jax.numpy as jnp
import metric_utils
from dopamine.jax import losses
import tensorflow as tf
import os
from dopamine.discrete_domains import run_experiment
from dopamine.jax.agents.dqn import dqn_agent

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
# NOTE: we don't use this flag
flags.DEFINE_string('replay_buffer_ckpt_dir', "logs/Alien/metric_dqn/118398/", 'Replay Buffer Checkpoint path to use')


flags.DEFINE_multi_string(
    'gin_files', ["eval_metric_dqn.gin"], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@gin.configurable
class WrapperMetriDQNBPERAgent(metric_dqn_bper_agent.MetricDQNBPERAgent):
  """Wrapper for the MetricDQNBPERAgent."""

  def __init__(self, *args, **kwargs):
    super(WrapperMetriDQNBPERAgent, self).__init__(*args, **kwargs)
    self._collected_batch = collections.defaultdict(list)
    self._metric_stats = eval_utils.RunningStats()
    self._euclidean_stats = eval_utils.RunningStats()
    self._td_errors_stats = eval_utils.RunningStats()
    self._residuals_stats = eval_utils.RunningStats()

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    # Store the last transition for evaluation
    self.maybe_record_stats_and_reset_batch()
    self._last_observation = self._observation

    # NOTE: The next state is recorded after one frame is skipped.
    self._collected_batch['state'].append(self.state.copy())
    self._record_observation(observation)
    self._collected_batch['next_state'].append(self.state)

    self._collected_batch['action'].append(self.action)
    self._collected_batch['reward'].append(reward)
    self._collected_batch['terminal'].append(False)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()
   
    self._rng, self.action = dqn_agent.select_action(
        self.network_def,
        self.online_params,
        self.preprocess_fn(self.state),
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
    )
    self.action = onp.asarray(self.action)
    return self.action
  
  def maybe_record_stats_and_reset_batch(self):
    if len(self._collected_batch['state']) == self._batch_size_statistics:
        
        # Prepare the batches for calculation
        tmp_replay_elements = collections.OrderedDict()
        tmp_replay_elements['state'] = jnp.stack(self._collected_batch['state'])
        tmp_replay_elements['action'] = jnp.stack(self._collected_batch['action'])
        tmp_replay_elements['reward'] = jnp.stack(self._collected_batch['reward'])
        tmp_replay_elements['terminal'] = jnp.stack(self._collected_batch['terminal'])
        tmp_replay_elements['next_state'] = jnp.stack(self._collected_batch['next_state'])
          
        self.calculate_stats(tmp_replay_elements)
        self._collected_batch = collections.defaultdict(list)

  def calculate_stats(self, batch):

    # Calculate the statistics
    eval_batch = self._get_outputs(agent_id='fixed', 
                                    next_states=True, 
                                    intermediates=False,
                                    batch=batch)
        
    curr_outputs, next_outputs = eval_batch['output'], eval_batch['output_next']

    metric_distances = metric_utils.current_next_distances(
        curr_outputs.representation, next_outputs.representation, self._distance_fn)
    self._metric_stats.update_running_stats(metric_distances)

    euclidean_distances = jnp.sqrt(jnp.sum((curr_outputs.representation - next_outputs.representation)**2, axis=1))
    self._euclidean_stats.update_running_stats(euclidean_distances)

    eval_batch = self._get_outputs(agent_id='online',
                            next_states=False, 
                            intermediates=False,
                            batch=batch)

    td_errors = eval_utils.log_td_errors(
        eval_batch,
        self.network_def,
        self.target_network_params,
        self.cumulative_gamma
        )
    self._td_errors_stats.update_running_stats(td_errors)

@gin.configurable
def create_metric_agent(sess, environment,
                        summary_writer=None, debug_mode=False):  
    del sess
    del debug_mode

    return WrapperMetriDQNBPERAgent(
            num_actions=environment.action_space.n, 
            summary_writer=summary_writer,
            game_name=FLAGS.game_name,)

def main(unused_argv):
  _ = unused_argv
  logging.set_verbosity(logging.INFO)
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings

  addition_bindings = [
    f"atari_lib.create_atari_environment.game_name='{FLAGS.game_name}'",
    f"JaxDQNAgent.seed={FLAGS.seed}",
    f"MetricDQNBPERAgent.fixed_agent_ckpt='logs/{FLAGS.game_name}/metric_dqn/442081/checkpoints/ckpt.99'"
  ]

  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings + addition_bindings, skip_unknown=False)
  
#   paths = list(pathlib.Path(FLAGS.checkpoint_dir).parts)
#   run_number = paths[-1].split('_')[-1]

  ckpt_dir = osp.join(FLAGS.checkpoint_dir, 'checkpoints')
  logging.info('Checkpoint directory: %s', ckpt_dir)

  # Create the environment and agent.
  logging.info('Game: %s', FLAGS.game_name)
  
  base_dir = FLAGS.base_dir
  experiment_dir = os.path.join(base_dir, FLAGS.game_name, FLAGS.agent_name, FLAGS.seed, 'eval_metrics')

  # Create folder if it doesn't exist
  if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
  logging.info('Experiment directory: %s', experiment_dir)

  checkpoints = pretrained_metric_dqn.get_checkpoints(ckpt_dir, max_checkpoints=100)

  # Set a runner for collecting transitions and calculate the statistics over it
  runner = run_experiment.Runner(experiment_dir, create_metric_agent)
  
  # Number of steps to collect
  evaluation_steps = 20480 # 20481

  for idx, checkpoint in enumerate(checkpoints):
    # Load the checkpoint.
    pretrained_metric_dqn.reload_checkpoint(runner._agent, checkpoint)
    logging.info('Checkpoint loaded: %s', checkpoint)
    logging.info('Calculating statistics...')

    # Run one phase in eval mode
    # runner._agent.eval_mode = True
    runner._run_one_phase(evaluation_steps, [], 'eval')

    # Get the statistics
    stats = collections.OrderedDict()
    stats['Exploration/BisimulationDistanceAvg'] = runner._agent._metric_stats.mean
    stats['Exploration/EuclideanDistanceAvg'] = runner._agent._euclidean_stats.mean
    stats['Exploration/TD-ErrorAvg'] = runner._agent._td_errors_stats.mean
    stats['Exploration/BisimulationDistanceStd'] = jnp.sqrt(runner._agent._metric_stats.variance)
    stats['Exploration/EuclideanDistanceStd'] = jnp.sqrt(runner._agent._euclidean_stats.variance)
    stats['Exploration/TD-ErrorStd'] = jnp.sqrt(runner._agent._td_errors_stats.variance)

    with tf.device('/CPU:0'):
        with runner._agent.summary_writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=(idx + 1) * 1000_000)
            
            runner._agent.summary_writer.flush()
    
if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  app.run(main)