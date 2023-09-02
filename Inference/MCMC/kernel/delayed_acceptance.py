# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Metropolis-Hastings Transition Kernel."""

import collections
import warnings

import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import unnest

__all__ = [
    'DelayedAcceptance',
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*metropolis_hastings',
                        append=True)  # Don't override user-set filters.


class DelayedAcceptanceKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'DelayedAcceptanceKernelResults',
        [
            'accepted_results',
            'is_accepted',
            'log_accept_ratio',
            'proposed_state',
            'proposed_results',
            'extra',
            'seed',
        ])
):
    """Internal state and diagnostics for MH."""
    __slots__ = ()


class DelayedAcceptance(kernel_base.TransitionKernel):
    """Runs one step of the Delayed Acceptance Metropolis-Hastings algorithm.
  """

    def __init__(self,
                 exact_target_prob,
                 inner_kernel,
                 name=None):
        """Instantiates this object.

    Args:
      inner_kernel: `TransitionKernel`-like object which has
        `collections.namedtuple` `kernel_results` and which contains a
        `target_log_prob` member and optionally a `log_acceptance_correction`
        member.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "mh_kernel").

    Returns:
      metropolis_hastings_kernel: Instance of `TransitionKernel` which wraps the
        input transition kernel with the Metropolis-Hastings algorithm.
    """
        if inner_kernel.is_calibrated:
            warnings.warn('Supplied `TransitionKernel` is already calibrated. '
                          'Composing `MetropolisHastings` `TransitionKernel` '
                          'may not be required.')
        self._parameters = dict(exact_target_prob=exact_target_prob,
                                inner_kernel=inner_kernel,
                                name=name)

    @property
    def exact_target_prob(self):
        return self._parameters['exact_target_prob']

    @property
    def inner_kernel(self):
        return self._parameters['inner_kernel']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Takes one step of the TransitionKernel.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made within the
        previous call to this function (or as returned by `bootstrap_results`).
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.

    Raises:
      ValueError: if `inner_kernel` results doesn't contain the member
        "target_log_prob".
    """
        is_seeded = seed is not None
        seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
        proposal_seed, acceptance_seed = samplers.split_seed(seed)

        with tf.name_scope(mcmc_util.make_name(self.name, 'da', 'one_step')):

            # First stage: one step of the approximate likelihood
            inner_kwargs = dict(seed=proposal_seed) if is_seeded else {}
            [
                proposed_approx_state,
                proposed_approx_results,
            ] = self.inner_kernel.one_step(
                current_state,
                previous_kernel_results.accepted_results,
                **inner_kwargs)
            if mcmc_util.is_list_like(current_state):
                proposed_approx_state = tf.nest.pack_sequence_as(current_state, proposed_approx_state)

            if (not has_target_log_prob(proposed_approx_results) or
                    not has_target_log_prob(previous_kernel_results.accepted_results)):
                raise ValueError('"target_log_prob" must be a member of '
                                 '`inner_kernel` results.')

            # Compute log(acceptance_ratio).
            to_sum = [proposed_approx_results.target_log_prob,
                      -previous_kernel_results.accepted_results.target_log_prob]
            try:
                if (not mcmc_util.is_list_like(
                        proposed_approx_results.log_acceptance_correction)
                        or proposed_approx_results.log_acceptance_correction):
                    to_sum.append(proposed_approx_results.log_acceptance_correction)
            except AttributeError:
                warnings.warn('Supplied inner `TransitionKernel` does not have a '
                              '`log_acceptance_correction`. Assuming its value is `0.`')

            def _compute_accept_ratio(sum_list):

                accept_ratio = mcmc_util.safe_sum(
                    sum_list, name='compute_log_accept_ratio')

                # If proposed state reduces likelihood: randomly accept.
                # If proposed state increases likelihood: always accept.
                # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
                #       ==> log(u) < log_accept_ratio
                log_uniform = tf.math.log(
                    samplers.uniform(
                        shape=prefer_static.shape(proposed_approx_results.target_log_prob),
                        dtype=dtype_util.base_dtype(
                            proposed_approx_results.target_log_prob.dtype),
                        seed=acceptance_seed))
                accepted = log_uniform < log_accept_ratio

                return accepted, accept_ratio

            is_accepted, log_accept_ratio = _compute_accept_ratio(to_sum)

            # if is_accepted is true, then go to second stage, to calculate the exact probability
            next_state = mcmc_util.choose(
                is_accepted,
                proposed_approx_state,
                current_state,
                name='choose_next_state')

            kernel_results = DelayedAcceptanceKernelResults(
                accepted_results=mcmc_util.choose(
                    is_accepted,
                    # We strip seeds when populating `accepted_results` because unlike
                    # other kernel result fields, seeds are not a per-chain value.
                    # Thus it is impossible to choose between a previously accepted
                    # seed value and a proposed seed, since said choice would need to
                    # be made on a per-chain basis.
                    mcmc_util.strip_seeds(proposed_approx_results),
                    previous_kernel_results.accepted_results,
                    name='choose_inner_results'),
                is_accepted=is_accepted,
                log_accept_ratio=log_accept_ratio,
                proposed_state=proposed_approx_state,
                proposed_results=proposed_approx_results,
                extra=[],
                seed=seed,
            )

            # second stage: no need to propose the new stage, just calculate
            if is_accepted:
                exact_propose_target_log_prob = self.exact_target_prob(*proposed_approx_state)
                exact_current_target_log_prob = self.exact_target_prob(*current_state)

                # TODO: need to backtrace the last accepted?
                # Compute log(acceptance_ratio).
                to_sum = [exact_propose_target_log_prob,
                          -exact_current_target_log_prob,
                          -proposed_approx_results.target_log_prob,
                          previous_kernel_results.accepted_results.target_log_prob]

                is_accepted_exact, log_accept_ratio_exact = _compute_accept_ratio(to_sum)

                next_state = mcmc_util.choose(
                    is_accepted_exact,
                    proposed_approx_state,
                    current_state,
                    name='choose_next_state')

                target_prob_setter_fn(proposed_approx_results, exact_propose_target_log_prob)

                kernel_results = kernel_results._replace(is_accepted=is_accepted_exact,
                                                         log_accept_ratio=log_accept_ratio_exact,
                                                         proposed_results=proposed_approx_results,
                                                         accepted_results=mcmc_util.choose(
                                                             is_accepted_exact,
                                                             mcmc_util.strip_seeds(proposed_approx_results),
                                                             previous_kernel_results.accepted_results,
                                                             name='choose_inner_results')
                                                         )

            return next_state, kernel_results

    def bootstrap_results(self, init_state):
        """Returns an object with the same type as returned by `one_step`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        initial state(s) of the Markov chain(s).

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.

    Raises:
      ValueError: if `inner_kernel` results doesn't contain the member
        "target_log_prob".
    """
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'da', 'bootstrap_results')):
            pkr = self.inner_kernel.bootstrap_results(init_state)
            if not has_target_log_prob(pkr):
                raise ValueError(
                    '"target_log_prob" must be a member of `inner_kernel` results.')
            x = pkr.target_log_prob
            return DelayedAcceptanceKernelResults(
                # See note regarding `strip_seeds` above in `one_step`.
                accepted_results=mcmc_util.strip_seeds(pkr),
                is_accepted=tf.ones_like(x, dtype=tf.bool),
                log_accept_ratio=tf.zeros_like(x),
                proposed_state=init_state,
                proposed_results=pkr,
                extra=[],
                # Allow room for one_step's seed.
                seed=samplers.zeros_seed(),
            )

    @property
    def experimental_shard_axis_names(self):
        return self.inner_kernel.experimental_shard_axis_names

    def experimental_with_shard_axes(self, shard_axis_names):
        return self.copy(
            inner_kernel=self.inner_kernel.experimental_with_shard_axes(
                shard_axis_names))


def has_target_log_prob(kernel_results):
    """Returns `True` if `target_log_prob` is a member of input."""
    return getattr(kernel_results, 'target_log_prob', None) is not None


def target_prob_setter_fn(kernel_results, new_target_prob):
    """Setter for `new_target_prob`"""
    return unnest.replace_innermost(kernel_results, target_log_prob=new_target_prob)
