"""Particle Marginal Metropolis Hastings Transition Kernel."""

import collections
# Dependency imports

import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn, random_walk_uniform_fn

__all__ = [
    'ParticleMetropolisHastings',
    'UncalibratedPMCMC',
]


class UncalibratedPMCMCResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedPMCMCResults',
        [
            'log_acceptance_correction',
            'target_log_prob',  # For "next_state".
            'seed',
            'smc_results'  # From particle filter
        ])
):
    """Internal state and diagnostics for Particle MH."""
    __slots__ = ()


class ParticleMetropolisHastings(kernel_base.TransitionKernel):
    """Runs one step of the PMCMC algorithm with symmetric proposal.

  Current class implements PMCMC for normal and uniform proposals. Alternatively,
  the user can supply any custom proposal generating function.

  The function `one_step` can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics
  are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)


  """

    def __init__(self,
                 target_log_prob_fn,
                 num_particles,
                 particle_filter_method=None,
                 proposal_theta_fn=None,
                 experimental_shard_axis_names=None,
                 name=None):
        """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      proposal_theta_fn: Python callable which takes a list of state parts and a
        seed; returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        Default value: `None` which is mapped to
          `tfp.mcmc.random_walk_normal_fn()`.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'rwm_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `scale` or a list with same length as
        `current_state`.
    """
        if proposal_theta_fn is None:
            proposal_theta_fn = random_walk_normal_fn()

        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=UncalibratedPMCMC(
                target_log_prob_fn=target_log_prob_fn,
                proposal_theta_fn=proposal_theta_fn,
                num_particles=num_particles,
                particle_filter_method=particle_filter_method,
                name=name)).experimental_with_shard_axes(
            experimental_shard_axis_names)
        self._parameters = self._impl.inner_kernel.parameters.copy()

    @property
    def target_log_prob_fn(self):
        return self._impl.inner_kernel.target_log_prob_fn

    @property
    def proposal_theta_fn(self):
        return self._impl.inner_kernel.proposal_theta_fn

    @property
    def name(self):
        return self._impl.inner_kernel.name

    @property
    def is_calibrated(self):
        return True

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._impl.inner_kernel.parameters

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Runs one iteration of Random Walk Metropolis with normal proposal.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `scale` or a list with same length as
        `current_state`.
    """
        return self._impl.one_step(current_state, previous_kernel_results,
                                   seed=seed)

    def bootstrap_results(self, init_state):
        """Creates initial `previous_kernel_results` using a supplied `state`."""
        return self._impl.bootstrap_results(init_state)

    @property
    def experimental_shard_axis_names(self):
        return self._parameters['experimental_shard_axis_names']

    def experimental_with_shard_axes(self, shard_axes):
        return self.copy(experimental_shard_axis_names=shard_axes)


class UncalibratedPMCMC(kernel_base.TransitionKernel):
    """Generate proposal for the Particle Metropolis algorithm.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedPMCMC(...))`.

  """

    @mcmc_util.set_doc(ParticleMetropolisHastings.__init__.__doc__)
    def __init__(self,
                 target_log_prob_fn,
                 num_particles,
                 particle_filter_method,
                 proposal_theta_fn=None,
                 experimental_shard_axis_names=None,
                 name=None):
        if proposal_theta_fn is None:
            proposal_theta_fn = random_walk_normal_fn()
        if particle_filter_method is None:
            particle_filter_method = 'bsf'

        self._target_log_prob_fn = target_log_prob_fn
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            proposal_theta_fn=proposal_theta_fn,
            experimental_shard_axis_names=experimental_shard_axis_names,
            particle_filter_method=particle_filter_method,
            num_particles=num_particles,
            name=name)

    @property
    def num_particles(self):
        return self._parameters['num_particles']

    @property
    def particle_filter_method(self):
        return self._parameters['particle_filter_method']

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def proposal_theta_fn(self):
        return self._parameters['proposal_theta_fn']

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

    @mcmc_util.set_doc(ParticleMetropolisHastings.one_step.__doc__)
    def one_step(self, current_state, previous_kernel_results, seed=None):
        with tf.name_scope(mcmc_util.make_name(self.name, 'rwm', 'one_step')):
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

            next_state_parts = self.proposal_theta_fn(  # pylint: disable=not-callable
                current_state_parts, seed, **state_fn_kwargs)

            # User should be using a proposal_theta_fn that does not alter the state size.
            # This will fail noisily if that is not the case.
            for next_part, current_part in zip(next_state_parts, current_state_parts):
                tensorshape_util.set_shape(next_part, current_part.shape)

            # Compute `target_log_prob` so its available to MetropolisHastings.
            # Also trace the SMC results if necessary
            next_target_log_prob, trace_smc_results = self.target_log_prob_fn(
                *next_state_parts)  # pylint: disable=not-callable

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            return [
                maybe_flatten(next_state_parts),
                UncalibratedPMCMCResults(
                    log_acceptance_correction=tf.zeros_like(next_target_log_prob),
                    target_log_prob=next_target_log_prob,
                    seed=seed,
                    smc_results=trace_smc_results
                ),
            ]

    @mcmc_util.set_doc(ParticleMetropolisHastings.bootstrap_results.__doc__)
    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'rwm', 'bootstrap_results')):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]

            init_target_log_prob, trace_smc_results = self.target_log_prob_fn(
                                                                              *init_state)  # pylint:disable=not-callable

            return UncalibratedPMCMCResults(
                log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                target_log_prob=init_target_log_prob,
                # Allow room for one_step's seed.
                seed=samplers.zeros_seed(),
                smc_results=trace_smc_results)

    @property
    def experimental_shard_axis_names(self):
        return self._parameters['experimental_shard_axis_names']

    def experimental_with_shard_axes(self, shard_axis_names):
        return self.copy(experimental_shard_axis_names=shard_axis_names)
