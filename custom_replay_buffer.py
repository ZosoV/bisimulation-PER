from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import samplers
from typing import Any, Generic, Literal, TypeVar
import operator
import jax
import numpy as np
import gin
import pickle
import collections

# ReplayElementT = TypeVar('ReplayElementT', bound=elements.ReplayElementProtocol)


@gin.configurable
class CustomReplayBuffer(replay_buffer.ReplayBuffer[replay_buffer.ReplayElementT]):
    """Custom replay buffer that stores the experience tuples."""

    def __init__(self, 
            transition_accumulator: accumulator.Accumulator[replay_buffer.ReplayElementT],
            sampling_distribution: samplers.SamplingDistribution,
            seed: int = 0,
            *,
            batch_size: int,
            max_capacity: int,
            checkpoint_duration: int = 2,
            compress: bool = True,):
        super().__init__(
            transition_accumulator=transition_accumulator,
            sampling_distribution=sampling_distribution,
            batch_size=batch_size,
            max_capacity=max_capacity,
            checkpoint_duration=checkpoint_duration,
            compress=compress)

        # NOTE: set a fixed uniform distribution for sampling uniformly from the experience replay
        self._seed = seed
        self._uniform_distribution = samplers.UniformSamplingDistribution(
            seed=self._seed
       )

    def sample_uniform(
        self,
        size: 'int | None' = None,
        *,
        with_sample_metadata: bool = False,
    ) -> 'ReplayElementT | tuple[ReplayElementT, samplers.SampleMetadata]':
        """Sample a batch of elements uniformly from the replay buffer."""
        if self.add_count < 1:
            raise ValueError('No samples in replay buffer!')
        if size is None:
            size = self._batch_size
        if size < 1:
            raise ValueError(f'Invalid size: {size}, size must be >= 1.')

        samples = self._uniform_distribution.sample(size)
        replay_elements = operator.itemgetter(*samples.keys)(self._memory)
        
        if not isinstance(replay_elements, tuple):
            replay_elements = (replay_elements,)
        if self._compress:
            replay_elements = map(operator.methodcaller('unpack'), replay_elements)

        batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements)
        return (batch, samples) if with_sample_metadata else batch
    
    def add(self, transition: elements.TransitionElement, **kwargs: Any) -> None:
        """Add a transition to the replay buffer."""
        for replay_element in self._transition_accumulator.accumulate(transition):
            if self._compress:
                replay_element = replay_element.pack()

            # Add replay element to memory
            key = replay_buffer.ReplayItemID(self.add_count)
            self._memory[key] = replay_element
            self._sampling_distribution.add(key, **kwargs)
            self._uniform_distribution.add(key, **kwargs)
            self.add_count += 1

            # If we're beyond our capacity...
            if self.add_count > self._max_capacity:
                # Pop the oldest item from memory and keep the key
                # so we can ask the sampling distribution to remove it
                oldest_key, _ = self._memory.popitem(last=False)
                self._sampling_distribution.remove(oldest_key)
                self._uniform_distribution.remove(oldest_key)

    def update(
        self,
        keys: 'npt.NDArray[ReplayItemID] | ReplayItemID',
        **kwargs: Any,
    ) -> None:
        self._sampling_distribution.update(keys, **kwargs)
        self._uniform_distribution.update(keys, **kwargs)

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.add_count = 0
        self._memory.clear()
        self._transition_accumulator.clear()
        self._sampling_distribution.clear()
        self._uniform_distribution.clear()

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize replay buffer to a state dictionary."""
        # Serialize memory. We'll serialize keys and values separately.
        keys = list(self._memory.keys())
        # To serialize values we'll flatten each transition element.
        # This will serialize replay elements as:
        #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
        values = iter(self._memory.values())
        leaves, treedef = jax.tree_util.tree_flatten(next(values, None))
        values = [] if not leaves else [leaves, *map(treedef.flatten_up_to, values)]

        return {
            'add_count': self.add_count,
            'memory': {
                'keys': keys,
                'values': values,
                'treedef': pickle.dumps(treedef),
            },
            'sampling_distribution': self._sampling_distribution.to_state_dict(),
            'uniform_distribution': self._uniform_distribution.to_state_dict(),
            'transition_accumulator': self._transition_accumulator.to_state_dict(),
        }
    
    def from_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Deserialize and mutate replay buffer using state dictionary."""
        self.add_count = state_dict['add_count']
        self._transition_accumulator.from_state_dict(
            state_dict['transition_accumulator']
        )
        self._sampling_distribution.from_state_dict(
            state_dict['sampling_distribution']
        )

        self._uniform_distribution.from_state_dict(
            state_dict['uniform_distribution']
        )

        # Restore memory
        memory_keys = state_dict['memory']['keys']
        # Each element of the list is a flattened replay element, unflatten them
        # i.e., we have storage like:
        #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
        # and after unflattening we'll have:
        #   [ReplayElementT(...), ...]
        memory_treedef: jax.tree_util.PyTreeDef = pickle.loads(
            state_dict['memory']['treedef']
        )
        memory_values = map(
            memory_treedef.unflatten, state_dict['memory']['values']
        )

        # Create our new ordered dictionary from the restored keys and values
        self._memory = collections.OrderedDict[replay_buffer.ReplayItemID, replay_buffer.ReplayElementT](
            zip(memory_keys, memory_values, strict=True)
        )