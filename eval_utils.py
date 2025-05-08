import jax
import jax.numpy as jnp
import gin
import functools
from dopamine.jax import losses
from typing import Tuple, NamedTuple
import numpy as np

@functools.partial(jax.jit, static_argnames=('threshold', 'sub_mean_score'))
def log_dormant_percentage(batch_activations, threshold=0.0, sub_mean_score=False):
    # NOTE: this function requires first to track the intermediates activations
    # using a the function `get_intermediates` in the class `MetricAgent`.
    """JIT-compatible dormant neuron calculation."""
    
    def process_layer(layer_act):
        # Handle both traced and concrete arrays
        layer_act = layer_act[0] if isinstance(layer_act, (list, tuple)) else layer_act
        
        # Determine reduction axes (all but last dimension)
        reduce_axes = tuple(range(layer_act.ndim - 1))
        
        if sub_mean_score:
            layer_mean = jnp.mean(layer_act, axis=reduce_axes, keepdims=True)
            layer_act = layer_act - layer_mean
        
        # Compute normalized scores
        scores = jnp.mean(jnp.abs(layer_act), axis=reduce_axes)
        scores = scores / (jnp.mean(scores) + 1e-9)
        
        # Count dormant neurons
        dormant_neurons_count = jnp.count_nonzero(scores <= threshold)
        layer_total = scores.size
        
        return dormant_neurons_count, layer_total
    
    # Process all layers
    # NOTE: tree_map is more efficient than using a for loop
    per_layer_counts = jax.tree_util.tree_map(
        process_layer,
        batch_activations
    )
    
    total_neurons, total_dormant_neurons =  0.0, 0.0
    for k, layer_counts in per_layer_counts.items():
        # Process each layer
        dormant_count, neuron_count = layer_counts[0]

        # Sum counts
        total_dormant_neurons += dormant_count
        total_neurons += neuron_count
    
    # Return percentage
    return (total_dormant_neurons / (total_neurons + 1e-9)) * 100.0

@functools.partial(jax.jit, static_argnames=('network_def', 'intermediates'))
def get_features(network_def, params, states, intermediates = True):
    def apply_data(x):
        if intermediates:
            return network_def.apply(
                params,
                x,
                mutable=['intermediates'],
            )
        else:
            return network_def.apply(
                params,
                x
            )
    if intermediates:
        output, state = jax.vmap(apply_data)(states)
        return jax.lax.stop_gradient(output), jax.lax.stop_gradient(state['intermediates'])
    else:
        return jax.lax.stop_gradient(jax.vmap(apply_data)(states))
    
def log_avg_bisimulation_distance():
    pass

def target_q_value(target_network, next_states, rewards, terminals,
                   cumulative_gamma):
  """Compute the target Q-value."""



  next_state_output = jax.vmap(target_network, in_axes=(0))(next_states)
  next_state_q_vals = next_state_output.q_values
  next_state_q_vals = jnp.squeeze(next_state_q_vals)

  replay_next_qt_max = jnp.max(next_state_q_vals, 1)
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                            (1. - terminals))

def log_td_errors(sampled_batch, network_def, target_params, cumulative_gamma):
    model_output = sampled_batch["output"]
    q_values = model_output.q_values
    q_values = jnp.squeeze(q_values)

    def q_target(state):
        return network_def.apply(target_params, state)

    bellman_target = target_q_value(q_target, sampled_batch['next_state'],
                                   sampled_batch['reward'],
                                   sampled_batch['terminal'],
                                   cumulative_gamma)
        
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, sampled_batch['action'])
    return jnp.abs(replay_chosen_q - bellman_target)

def log_srank(feature_matrix, thresh=1e-5):
    singular_vals = np.linalg.svd(
      feature_matrix, full_matrices=False, compute_uv=False)
    
    return max(np.sum(singular_vals >= thresh), 1)

def log_td_residuals_avg_norm(residuals_matrix):
    """Compute the average norm of the TD residuals."""
    # Compute the L2 norm of each row
    norms = np.linalg.norm(residuals_matrix, axis=1)
    
    # Compute the average norm
    avg_norm = np.mean(norms)
    
    return avg_norm

def log_representation_avg_norm(feature_matrix):
    """Compute the average norm of the representation."""
    # Compute the L2 norm of each row
    norms = np.linalg.norm(feature_matrix, axis=1)
    
    # Compute the average norm
    avg_norm = np.mean(norms)
    
    return avg_norm

class RunningStats():#
    def __init__(self):
        self.mean = jnp.array(0.0)
        self.variance = jnp.array(0.0)
        self.count = jnp.array(0.0)

    
    def update_running_stats(self,new_batch: jnp.ndarray):
        """Update running statistics with a new batch of data."""
        batch_mean = jnp.mean(new_batch, axis=0)
        batch_var = jnp.var(new_batch, axis=0)  # Population variance (ddof=0)
        batch_count = new_batch.shape[0]

        # Combined count
        total_count = self.count + batch_count

        # Update running mean
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)

        # Update running variance (using Chan's parallel algorithm)
        new_variance = (
            (self.variance * self.count)
            + (batch_var * batch_count)
            + (delta ** 2 * self.count * batch_count / total_count)
        ) / total_count

        self.mean = new_mean
        self.variance = new_variance
        self.count = total_count