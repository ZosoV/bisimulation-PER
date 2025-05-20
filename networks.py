import collections
import gin
import jax
import jax.numpy as jnp
import flax.linen as nn

NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


@gin.configurable
class AtariDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    self.sow('intermediates', 'relu_1', x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    self.sow('intermediates', 'relu_2', x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    self.sow('intermediates', 'relu_3', x)
    representation = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(representation)
    x = nn.relu(x)
    self.sow('intermediates', 'relu_4', x)
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(x)
    return NetworkType(q_values, representation)
  
@gin.configurable
class AtariRainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    representation = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(representation)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return NetworkType(q_values, logits, probabilities, representation)