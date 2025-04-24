import jax
import jax.numpy as jnp
from dopamine.discrete_domains import run_experiment, checkpointer
import logging
import metric_dqn_bper_agent
import metric_utils
import collections

class CustomRunner(run_experiment.Runner):
    """Custom runner to support metric agents."""

    def __init__(self, *args, **kwargs):
        super(CustomRunner, self).__init__(*args, **kwargs)
        self.frozen_agent = FrozenMetricDQNBPERAgent(checkpoint_path)
        self._checkpointer = None

    def _run_eval_phase(self, statistics):
        """Run evaluation phase.

        Args:
        statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval'
        )
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info(
            'Average undiscounted return per evaluation episode: %.2f',
            average_return,
        )
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return



class FrozenMetricDQNBPERAgent(metric_dqn_bper_agent.MetricDQNBPERAgent):
    def __init__(self, 
                num_actions, 
                summary_writer=None,
                checkpoint_path=None,
                ):

        network = metric_utils.AtariDQNNetwork
        super().__init__(num_actions, network=network,
                                summary_writer=summary_writer)
        self.eval_mode = True
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """Loads the latest checkpoint of a Jax DQN agent and runs inference.

        Args:
            checkpoint_path: Path to the directory containing the agent's checkpoints.
            num_actions: Number of possible actions in the environment.
            observation_shape: Shape of a single observation.
            stack_size: Number of stacked frames in the state.

        Returns:
            A function that takes an observation as input and returns the greedy action.
        """
        self._checkpointer = checkpointer.Checkpointer(checkpoint_path)

        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_path)
        if latest_checkpoint_version is None:
            raise ValueError(f"No checkpoint found in {checkpoint_path}")
        
        logging.info(f"Loading frozen checkpoint iteration: {latest_checkpoint_version}")
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version
            )
            if self.unbundle(checkpoint_path, latest_checkpoint_version, experiment_data):
                logging.info("Checkpoint loaded successfully.")
            else:
                raise ValueError("Unable to load checkpoint.")

    def _sample_and_get_metrics(self):
        """Sampled 20 minibatch of 512 elements from the replay buffer and
        calculated the distance between current and next state representations in average, and standard deviation.
        """

        num_samples = 10240
        num_minibatches = 20
        num_elements_per_minibatch = num_samples // num_minibatches

        def eval_sample_from_replay_buffer():

            self.dummy_replay_elements = collections.OrderedDict()
            elems, metadata = self._replay.sample(size= num_elements_per_minibatch, with_sample_metadata=True)
            
            self.dummy_replay_elements['state'] = elems.state
            self.dummy_replay_elements['next_state'] = elems.next_state
            self.dummy_replay_elements['action'] = elems.action
            self.dummy_replay_elements['reward'] = elems.reward
            self.dummy_replay_elements['terminal'] = elems.is_terminal
            if self._replay_scheme == 'prioritized':
                self.dummy_replay_elements['indices'] = metadata.keys
                self.dummy_replay_elements['sampling_probabilities'] = metadata.probabilities

        sum_distance = 0
        sqsum_distance = 0
        max_distance = 0
        min_distance = 0
        for i in range(num_minibatches):
            # Sample a batch of 512 elements from the replay buffer
            # and calculate the distance between current and next state representations
            # in average, and standard deviation.
            eval_sample_from_replay_buffer()
            states = self.dummy_replay_elements['state']
            next_states = self.dummy_replay_elements['next_state']

            metrics = self.inference_fn(states, next_states)
            sum_distance += metrics['sum_distance']
            sqsum_distance += metrics['sqsum_distance']
            max_distance = jnp.maximum(max_distance, metrics['max_distance'])
            min_distance = jnp.minimum(min_distance, metrics['min_distance'])

        # Calculate the average and standard deviation of the distances
        avg_distance = sum_distance / num_samples
        std_distance = jnp.sqrt(sqsum_distance / num_samples - avg_distance**2)

        metrics = collections.OrderedDict()
        metrics['avg_distance'] = avg_distance
        metrics['std_distance'] = std_distance
        metrics['max_distance'] = max_distance
        metrics['min_distance'] = min_distance
        return metrics

    @jax.jit
    def inference_fn(self, states, next_states):

        def q_frozen(state):
            return self.network_def.apply(self.online_params, state)
        
        # current states
        model_output = jax.vmap(q_frozen)(states)
        # q_values = model_output.q_values
        # q_values = jnp.squeeze(q_values)
        curr_representations = model_output.representation
        curr_representations = jnp.squeeze(curr_representations)

        # next states
        model_output = jax.vmap(q_frozen)(next_states)
        # q_values = model_output.q_values
        # q_values = jnp.squeeze(q_values)
        next_representations = model_output.representation
        next_representations = jnp.squeeze(next_representations)


        experience_distances = metric_utils.current_next_distances(
        current_state_representations=curr_representations,
        next_state_representations=next_representations, # online_next_r,
        distance_fn = self.distance_fn,)

        metrics = collections.OrderedDict()
        metrics['sum_distance'] = jnp.sum(experience_distances)
        metrics['sqsum_distance'] = jnp.sum(experience_distances)
        metrics['max_distance'] = jnp.max(experience_distances)
        metrics['min_distance'] = jnp.min(experience_distances)

        return metrics


