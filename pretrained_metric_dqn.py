from dopamine.jax.agents.dqn import dqn_agent
import tensorflow as tf
import pickle
import jax
import jax.numpy as jnp
import numpy as np
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
    agent.online_params = bundle_dictionary['online_params']
    agent.optimizer = dqn_agent.create_optimizer(agent._optimizer_name)
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

def get_features(agent, states):
  def feature_fn(state):
    return agent.network_def.apply(agent.online_params, state)
  compute_features = jax.vmap(feature_fn)
  features = []
  for state in states:
    features.append(jnp.squeeze(compute_features(state)))
  return np.concatenate(features, axis=0)

def create_agent(num_actions, 
                 summary_writer=None, 
                 agent_name=None):
  """Creates an online agent.

  Args:
    num_actions: Number of actions in the environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if agent_name == 'metric_dqn':
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
    raise ValueError('{} is not a valid agent name'.format(agent_name))

  return agent(
      num_actions=num_actions,
      summary_writer=summary_writer,
      network=network)
