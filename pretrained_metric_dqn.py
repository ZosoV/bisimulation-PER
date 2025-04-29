from dopamine.jax.agents.dqn import dqn_agent
import tensorflow as tf
import pickle
from flax import core
from flax.training import checkpoints as flax_checkpoints
import os
import os.path as osp
import gin
import pathlib

from absl import app
from absl import flags
from absl import logging
import networks
from dopamine.discrete_domains import atari_lib


AGENTS = [
    'metric_dqn', 'metric_dqn_bper', 'metric_dqn_per'
]

flags.DEFINE_enum('agent_name', "metric_dqn", AGENTS, 'Name of the agent.')
flags.DEFINE_string('checkpoint_dir', "logs/Alien/metric_dqn/118398/metrics/pickle/pickle_99.pkl", 'Checkpoint path to use')
flags.DEFINE_string('game', 'Alien', 'Name of game')

# flags.DEFINE_multi_string(
#     'gin_files', ["dqn.gin"], 'List of paths to gin configuration files.')
# flags.DEFINE_multi_string(
#     'gin_bindings', [],
#     'Gin bindings to override the values set in the config files.')


FLAGS = flags.FLAGS


class PretrainedMetricDQNAgent(dqn_agent.JaxDQNAgent):

  def _build_replay_buffer(self):
    pass
  

def reload_checkpoint(agent, checkpoint_path):
  """Reload variables from a fully specified checkpoint."""
  assert checkpoint_path is not None
  with tf.io.gfile.GFile(checkpoint_path, 'rb') as fin:
    bundle_dictionary = pickle.load(fin)
  reload_jax_checkpoint(agent, bundle_dictionary)

def reload_jax_checkpoint(agent, bundle_dictionary):
  """Reload variables from a fully specified checkpoint."""
  if bundle_dictionary is not None:
    agent.state = bundle_dictionary['state']
    if isinstance(bundle_dictionary['online_params'], core.FrozenDict):
      agent.online_params = bundle_dictionary['online_params']
    else:  # Load pre-linen checkpoint.
      agent.online_params = core.FrozenDict({
          'params': flax_checkpoints.convert_pre_linen(
              bundle_dictionary['online_params']).unfreeze()
      })
    # We recreate the optimizer with the new online weights.
    # pylint: disable=protected-access
    agent.optimizer = dqn_agent.create_optimizer(agent._optimizer_name)
    # pylint: enable=protected-access
    if 'optimizer_state' in bundle_dictionary:
      agent.optimizer_state = bundle_dictionary['optimizer_state']
    else:
      agent.optimizer_state = agent.optimizer.init(agent.online_params)
    logging.info('Done restoring!')

def get_checkpoints(ckpt_dir, max_checkpoints=200):
  """Get the full path of checkpoints in `ckpt_dir`."""
  return [
      os.path.join(ckpt_dir, f'ckpt.{idx}') for idx in range(max_checkpoints)
  ]

def create_agent(environment, summary_writer=None):
  """Creates an online agent.

  Args:
    environment: An Atari 2600 environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if FLAGS.agent_name == 'metric_dqn':
    agent = PretrainedMetricDQNAgent
    network = networks.AtariDQNNetwork
#   elif FLAGS.agent_name == 'jax_rainbow':
#     agent = PretrainedRainbow
#     network = networks.RainbowNetworkWithFeatures
#   elif FLAGS.agent_name == 'jax_implicit_quantile':
#     agent = PretrainedIQN
#     network = networks.ImplicitQuantileNetworkWithFeatures
#   elif FLAGS.agent_name == 'mimplicit_quantile':
#     agent = PretrainedIQN
#     network = networks.ImplicitQuantileNetworkWithFeatures
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(
      num_actions=environment.action_space.n,
      summary_writer=summary_writer,
      network=network)

def main(unused_argv):
  _ = unused_argv
  logging.set_verbosity(logging.INFO)
#   gin_files = FLAGS.gin_files
#   gin_bindings = FLAGS.gin_bindings
#   gin.parse_config_files_and_bindings(
#       gin_files, bindings=gin_bindings, skip_unknown=False)
  
  paths = list(pathlib.Path(FLAGS.checkpoint_dir).parts)
  run_number = paths[-1].split('_')[-1]

  ckpt_dir = osp.join(FLAGS.checkpoint_dir, 'checkpoints')
  logging.info('Checkpoint directory: %s', ckpt_dir)

  # Create the environment and agent.
  logging.info('Game: %s', FLAGS.game)
  environment = atari_lib.create_atari_environment(
      game_name=FLAGS.game, sticky_actions=True)
  summary_writer = None  # Replace with actual summary writer creation.
  agent = create_agent(environment, summary_writer)

  checkpoints = get_checkpoints(ckpt_dir)

  # Load the checkpoint.
#   reload_checkpoint(agent, checkpoints[-1])
  logging.info('Checkpoint loaded successfully.')

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  app.run(main)