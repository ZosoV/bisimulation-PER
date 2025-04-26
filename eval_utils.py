import jax
import jax.numpy as jnp
import gin
import functools

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

@functools.partial(jax.jit, static_argnames=('network_def',))
def compute_intermediates(network_def, params, states):
    def apply_data(x):
        return network_def.apply(
            params,
            x,
            mutable=['intermediates'],
        )
    _, state = jax.vmap(apply_data)(states)
    return state['intermediates']