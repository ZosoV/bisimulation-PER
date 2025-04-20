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

"""Utilities for computing the MICo loss."""

import functools
import gin
import jax
from jax import custom_jvp
import jax.numpy as jnp


EPSILON = 1e-9


# The following two functions were borrowed from
# https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py
# as they resolve the instabilities observed when using `jnp.arccos`.
@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.)


def l2(x, y):
  return _sqrt(jnp.sum(jnp.square(x - y)))


def cosine_distance(x, y):
  numerator = jnp.sum(x * y)
  denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
  cos_similarity = numerator / (denominator + EPSILON)
  return jnp.arctan2(_sqrt(1. - cos_similarity**2), cos_similarity)


def squarify(x):
  batch_size = x.shape[0]
  if len(x.shape) > 1:
    representation_dim = x.shape[-1]
    return jnp.reshape(jnp.tile(x, batch_size),
                       (batch_size, batch_size, representation_dim))
  return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


@gin.configurable
def representation_distances(first_representations, second_representations,
                             distance_fn, beta=0.1,
                             return_distance_components=False):
  """Compute distances between representations.

  This will compute the distances between two representations.

  Args:
    first_representations: first set of representations to use.
    second_representations: second set of representations to use.
    distance_fn: function to use for computing representation distances.
    beta: float, weight given to cosine distance between representations.
    return_distance_components: bool, whether to return the components used for
      computing the distance.

  Returns:
    The distances between representations, combining the average of the norm of
    the representations and the distance given by distance_fn.
  """
  batch_size = first_representations.shape[0]
  representation_dim = first_representations.shape[-1]
  first_squared_reps = squarify(first_representations)
  first_squared_reps = jnp.reshape(first_squared_reps,
                                   [batch_size**2, representation_dim])
  second_squared_reps = squarify(second_representations)
  second_squared_reps = jnp.transpose(second_squared_reps, axes=[1, 0, 2])
  second_squared_reps = jnp.reshape(second_squared_reps,
                                    [batch_size**2, representation_dim])
  base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(first_squared_reps,
                                                         second_squared_reps)
  norm_average = 0.5 * (jnp.sum(jnp.square(first_squared_reps), -1) +
                        jnp.sum(jnp.square(second_squared_reps), -1))
  if return_distance_components:
    return norm_average + beta * base_distances, norm_average, base_distances
  return norm_average + beta * base_distances


def absolute_reward_diff(r1, r2):
  return jnp.abs(r1 - r2)


@gin.configurable
def target_distances(representations, rewards, distance_fn, cumulative_gamma):
  """Target distance using the metric operator."""
  next_state_similarities = representation_distances(
      representations, representations, distance_fn)
  squared_rews = squarify(rewards)
  squared_rews_transp = jnp.transpose(squared_rews)
  squared_rews = squared_rews.reshape((squared_rews.shape[0]**2))
  squared_rews_transp = squared_rews_transp.reshape(
      (squared_rews_transp.shape[0]**2))
  reward_diffs = absolute_reward_diff(squared_rews, squared_rews_transp)
  return (
      jax.lax.stop_gradient(
          reward_diffs + cumulative_gamma * next_state_similarities))

@gin.configurable
def current_next_distances(
        current_state_representations,
        next_state_representations,
        distance_fn,
        beta=0.1):
    """Compute distances between current and next state representations."""
    base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(current_state_representations,
                                                           next_state_representations)

    norm_average = 0.5 * (jnp.sum(jnp.square(current_state_representations), -1) +
                          jnp.sum(jnp.square(next_state_representations), -1))

    return norm_average + beta * base_distances

@gin.configurable
class ExponentialNormalizer(object):
    def __init__(self, decay=0.99):
        self._decay: float = decay
        self._running_sum: jnp.ndarray = jnp.zeros(1)
        self._running_sqsum: jnp.ndarray = jnp.zeros(1)
        self._running_count: jnp.ndarray = jnp.zeros(1)
        # self._running_min: jnp.ndarray = jnp.array(float('inf'))  # Track min
        self._initialized: bool = False
        self._eps: float = 1e-10
    
    def normalize(self, x, update_stats=True):
        """Normalize input and optionally update running statistics"""
        # It needs to be first to set the values in the first batch
        if update_stats:
            self.update(x)

        # Calculate current mean and std
        mean, std = self.current_stats()
        
        # Normalize the input
        normalized = (x - mean) / jnp.maximum(std, self._eps)
        # Shift to positive range
        # if update_stats:
        #     # Chose between the minimum of the running minimum and the minimum of the batch
        #     # to avoid negative values
        #     self._running_min = jnp.minimum(self._running_min, jnp.min(normalized))
        
        # If the minimum is positive, we don't need to shift
        min_val = jnp.min(normalized)
        normalized = normalized - min_val + self._eps if min_val < 0 else normalized
        return normalized
    
    def current_stats(self):
        """Get current mean and standard deviation"""
        mean = self._running_sum / self._running_count
        var = (self._running_sqsum / self._running_count) - mean**2
        std = jnp.sqrt(jnp.maximum(var, self._eps))
        return mean, std
    
    def update(self, x):
        """Update running statistics"""
        batch_sum = jnp.sum(x)
        batch_sqsum = jnp.sum(x**2)
        batch_count = jnp.array(x.size, dtype=jnp.float32)
        
        if not self._initialized:
            # Initialize on first update
            self._running_sum = batch_sum
            self._running_sqsum = batch_sqsum
            self._running_count = batch_count
            self._initialized = True
        else:
            # Exponential moving average update
            new_sum = self._decay * self._running_sum + batch_sum
            new_sqsum = self._decay * self._running_sqsum + batch_sqsum
            new_count = self._decay * self._running_count + batch_count

            self._running_sum = new_sum
            self._running_sqsum = new_sqsum
            self._running_count = new_count