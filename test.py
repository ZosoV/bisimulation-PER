import jax
import jax.numpy as jnp



class ExponentialNormalizer(object):
    def __init__(self, decay=0.99):
        self._decay: float = decay
        self._running_sum: jnp.ndarray = jnp.zeros(1)
        self._running_sqsum: jnp.ndarray = jnp.zeros(1)
        self._running_count: jnp.ndarray = jnp.zeros(1)
        self._initialized: bool = False
    
    def normalize(self, x, update_stats=True):
        """Normalize input and optionally update running statistics"""
        # Calculate current mean and std
        mean, std = self.current_stats()
        
        # Normalize the input
        normalized = (x - mean) / jnp.maximum(std, 1e-4)
        
        if update_stats:
            self.update(x)
        return normalized
    
    def current_stats(self):
        """Get current mean and standard deviation"""
        mean = self._running_sum / self._running_count
        var = (self._running_sqsum / self._running_count) - mean**2
        std = jnp.sqrt(jnp.maximum(var, 1e-4))
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
    
normalizer = ExponentialNormalizer(decay=0.90)  # Remembers ~100 batches
stats = []
for i in range(1000):
    # Simulated data: mean shifts from 0→5 at step 500
    batch = jnp.ones((32, 1)) * (5 if i >= 500 else 0)
    _ = normalizer.normalize(batch)
    if i % 50 == 0:
        mean, std = normalizer.current_stats()
        stats.append((i, float(mean), float(std)))
        print(mean,std)