import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn
from tensorflow_probability.python.internal import distribute_lib

tfd = tfp.distributions  # pylint: disable=no-member
tfb = tfp.bijectors  # pylint: disable=no-member
mcmc = tfp.mcmc  # pylint: disable=no-member

__all__ = [
    'SamplingKernel',
    'SamplingKernelResults',
]


class SamplingKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "SamplingKernelResults",
        [
            "target_log_prob",
            "seed",
        ],
    ),
):
    __slots__ = ()


class SamplingKernel(mcmc.TransitionKernel):
    """Minimal conditional sampling Kernel for Gibbs
    """

    def __init__(self,
                 target_log_prob_fn,
                 current_full_states,
                 full_cond_dist,
                 sampling_idx,
                 experimental_shard_axis_names=None,
                 name=None):

        self._target_log_prob_fn = target_log_prob_fn
        self._sampling_idx = sampling_idx
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            current_full_states=current_full_states,
            experimental_shard_axis_names=experimental_shard_axis_names,
            full_cond_dist=full_cond_dist,
            sampling_idx=sampling_idx,
            name=name)

    @property
    def sampling_idx(self):
        return self._parameters['sampling_idx']

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def current_full_states(self):
        return self._parameters['current_full_states']

    @property
    def full_cond_dist(self):
        return self._parameters['full_cond_dist']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results, seed=None):
        with tf.name_scope(mcmc_util.make_name(self.name, 'cond_sampling', 'one_step')):
            with tf.name_scope('initialize'):
                if mcmc_util.is_list_like(current_state):
                    current_state_parts = list(current_state)
                else:
                    current_state_parts = [current_state]
                current_state_parts = [
                    tf.convert_to_tensor(s, name='current_state')
                    for s in current_state_parts
                ]

            seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
            state_fn_kwargs = {}
            if self.experimental_shard_axis_names is not None:
                state_fn_kwargs['experimental_shard_axis_names'] = (
                    self.experimental_shard_axis_names)

            next_state_parts = self.full_cond_dist(current_state=current_state_parts, # pylint: disable=not-callable
                state_parts=self.current_full_states, seed=seed, sampling_idx=self._sampling_idx)

            # User should be using a new_state_fn that does not alter the state size.
            # This will fail noisily if that is not the case.

            def _set_shape_nested(A, B):
                if isinstance(A, list) and isinstance(B, list):
                    return [_set_shape_nested(a, b) for a, b in zip(A, B)]
                else:
                   return tf.reshape(A, tf.shape(B))
            next_state_parts = _set_shape_nested(next_state_parts, self.current_full_states)

            # for next_part, current_part in zip(next_state_parts, self.current_full_states):
            #     tensorshape_util.set_shape(next_part, current_part.shape)

            # Compute `target_log_prob` so its available to MetropolisHastings.
            next_target_log_prob = self.target_log_prob_fn(next_state_parts[self.sampling_idx])  # pylint: disable=not-callable

            return [
                next_state_parts[self.sampling_idx],
                SamplingKernelResults(
                    target_log_prob=next_target_log_prob,
                    seed=seed,
                ),
            ]

    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'cond_sampling', 'bootstrap_results')):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]
            init_target_log_prob = self.target_log_prob_fn(*init_state)  # pylint:disable=not-callable
            return SamplingKernelResults(
                target_log_prob=init_target_log_prob,
                # Allow room for one_step's seed.
                seed=samplers.zeros_seed())

    @property
    def experimental_shard_axis_names(self):
        return self._parameters['experimental_shard_axis_names']

    def experimental_with_shard_axes(self, shard_axis_names):
        return self.copy(experimental_shard_axis_names=shard_axis_names)


def cond_sample_fn(inner_sample_dist):
    """ Conditional sampling function
    """

    def _fn(current_state,
            state_parts,
            seed,
            sampling_idx,
            name=None,
            experimental_shard_axis_names=None):
        with tf.name_scope(name or 'ful_cond_fn'):
            part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
            part_seeds = distribute_lib.fold_in_axis_index(
                part_seeds, experimental_shard_axis_names)

            state_parts = [
                tf.convert_to_tensor(s, name="current_state") for s in state_parts
            ]

            sample_state = state_parts[sampling_idx]
            state_parts.pop(sampling_idx)

            # sometimes sample_state is not necessary, but useful in most cases
            updated_state = inner_sample_dist(sampling_idx, sample_state, state_parts, part_seeds[sampling_idx])

            state_parts.insert(sampling_idx, updated_state)

        return state_parts

    return _fn

