# Copyright 2020 The TensorFlow Probability Authors.
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
"""Particle filtering."""

import numpy as np
import tensorflow as tf
from . import smc_kernel
from . import resampling as weighted_resampling
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import loop_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

__all__ = [
    'particle_filter',
    '_check_resample_fn'
]

# Default trace criterion.
_always_trace = lambda *_: True

particle_filter_arg_str = """\
Each latent state is a `Tensor` or nested structure of `Tensor`s, as defined
by the `initial_state_prior`.

The `transition_fn` and `proposal_fn` args, if specified, have signature
`next_state_dist = fn(step, state)`, where `step` is an `int` `Tensor` index
of the current time step (beginning at zero), and `state` represents
the latent state at time `step`. The return value is a `tfd.Distribution`
instance over the state at time `step + 1`.

Similarly, the `observation_fn` has signature
`observation_dist = observation_fn(step, state)`, where the return value
is a distribution over the value(s) observed at time `step`.

Args:
  observations: a (structure of) Tensors, each of shape
    `concat([[num_observation_steps, b1, ..., bN], event_shape])` with
    optional batch dimensions `b1, ..., bN`.
  initial_state_prior: a (joint) distribution over the initial latent state,
    with optional batch shape `[b1, ..., bN]`.
  transition_fn: callable returning a (joint) distribution over the next
    latent state.
  observation_fn: callable returning a (joint) distribution over the current
    observation.
  num_particles: `int` `Tensor` number of particles.
  initial_state_proposal: a (joint) distribution over the initial latent
    state, with optional batch shape `[b1, ..., bN]`. If `None`, the initial
    particles are proposed from the `initial_state_prior`.
    Default value: `None`.
  proposal_fn: callable returning a (joint) proposal distribution over the
    next latent state. If `None`, the dynamics model is used (
    `proposal_fn == transition_fn`).
    Default value: `None`.
  resample_fn: Python `callable` to generate the indices of resampled
    particles, given their weights. Generally, one of
    `tfp.experimental.mcmc.resample_independent` or
    `tfp.experimental.mcmc.resample_systematic`, or any function
    with the same signature, `resampled_indices = f(log_probs, event_size, '
    'sample_shape, seed)`.
    Default: `tfp.experimental.mcmc.resample_systematic`.
  resample_criterion_fn: optional Python `callable` with signature
    `do_resample = resample_criterion_fn(log_weights)`,
    where `log_weights` is a float `Tensor` of shape
    `[b1, ..., bN, num_particles]` containing log (unnormalized) weights for
    all particles at the current step. The return value `do_resample`
    determines whether particles are resampled at the current step. In the
    case `resample_criterion_fn==None`, particles are resampled at every step.
    The default behavior resamples particles when the current effective
    sample size falls below half the total number of particles.
    Default value: `tfp.experimental.mcmc.ess_below_threshold`.
  unbiased_gradients: If `True`, use the stop-gradient
    resampling trick of Scibior, Masrani, and Wood [{scibor_ref_idx}] to
    correct for gradient bias introduced by the discrete resampling step. This
    will generally increase the variance of stochastic gradients.
    Default value: `True`.
  rejuvenation_kernel_fn: optional Python `callable` with signature
    `transition_kernel = rejuvenation_kernel_fn(target_log_prob_fn)`
    where `target_log_prob_fn` is a provided callable evaluating
    `p(x[t] | y[t], x[t-1])` at each step `t`, and `transition_kernel`
    should be an instance of `tfp.mcmc.TransitionKernel`.
    Default value: `None`.  # TODO(davmre): not yet supported.
  num_transitions_per_observation: scalar Tensor positive `int` number of
    state transitions between regular observation points. A value of `1`
    indicates that there is an observation at every timestep,
    `2` that every other step is observed, and so on. Values greater than `1`
    may be used with an appropriately-chosen transition function to
    approximate continuous-time dynamics. The initial and final steps
    (steps `0` and `num_timesteps - 1`) are always observed.
    Default value: `None`.
"""

"""
Particle Filter Main Functions: Bootstrap, EKPF, APF
"""


def _default_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.parent_indices,
            kernel_results.incremental_log_marginal_likelihood,
            kernel_results.accumulated_log_marginal_likelihood)


@docstring_util.expand_docstring(
    particle_filter_arg_str=particle_filter_arg_str.format(scibor_ref_idx=1))
def particle_filter(observations,
                    initial_state_prior,
                    transition_fn,
                    observation_fn,
                    num_particles,
                    initial_state_proposal=None,
                    proposal_fn=None,
                    resample_fn='systematic',
                    resample_ess_num=0.5,
                    unbiased_gradients=False,
                    filter_method=None,
                    num_transitions_per_observation=1,
                    trace_fn=_default_trace_fn,
                    trace_criterion_fn=_always_trace,
                    static_trace_allocation_size=None,
                    parallel_iterations=1,
                    is_conditional=False,
                    conditional_sample=None,
                    seed=None,
                    name=None,
                    auxiliary_fn=None,
                    **kwargs):  # pylint: disable=g-doc-args
    """Samples a series of particles representing filtered latent states.

  The particle filter samples from the sequence of "filtering" distributions
  `p(state[t] | observations[:t])` over latent
  states: at each point in time, this is the distribution conditioned on all
  observations *up to that time*. Because particles may be resampled, a particle
  at time `t` may be different from the particle with the same index at time
  `t + 1`. To reconstruct trajectories by tracing back through the resampling
  process, see `tfp.mcmc.experimental.reconstruct_trajectories`.

  ${particle_filter_arg_str}
    trace_fn: Python `callable` defining the values to be traced at each step,
      with signature `traced_values = trace_fn(weighted_particles, results)`
      in which the first argument is an instance of
      `tfp.experimental.mcmc.WeightedParticles` and the second an instance of
      `SequentialMonteCarloResults` tuple, and the return value is a structure
      of `Tensor`s.
      Default value: `lambda s, r: (s.particles, s.log_weights,
      r.parent_indices, r.incremental_log_marginal_likelihood)`
    trace_criterion_fn: optional Python `callable` with signature
      `trace_this_step = trace_criterion_fn(weighted_particles, results)` taking
      the same arguments as `trace_fn` and returning a boolean `Tensor`. If
      `None`, only values from the final step are returned.
      Default value: `lambda *_: True` (trace every step).
    static_trace_allocation_size: Optional Python `int` size of trace to
      allocate statically. This should be an upper bound on the number of steps
      traced and is used only when the length cannot be
      statically inferred (for example, if a `trace_criterion_fn` is specified).
      It is primarily intended for contexts where static shapes are required,
      such as in XLA-compiled code.
      Default value: `None`.
    is_conditional: whether the conditional particle filter is conducted, which
      is used in the particle MCMC (particle Gibbs sampling) algorithm.
      Default value: 'False'
    conditional_sample: the conditional particle filter samples and trajectory
      Default value: None
    parallel_iterations: Passed to the internal `tf.while_loop`.
      Default value: `1`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'particle_filter'`).
  Returns:
    traced_results: A structure of Tensors as returned by `trace_fn`. If
      `trace_criterion_fn==None`, this is computed from the final step;
      otherwise, each Tensor will have initial dimension `num_steps_traced`
      and stacks the traced results across all steps.

  #### References

  [1] Adam Scibior, Vaden Masrani, and Frank Wood. Differentiable Particle
      Filtering without Modifying the Forward Pass. _arXiv preprint
      arXiv:2106.10314_, 2021. https://arxiv.org/abs/2106.10314
  """
    with tf.name_scope(name or 'particle_filter'):

        resample_fn = _check_resample_fn(resample_fn)

        # if aux not exist then reset it
        is_apf = False if auxiliary_fn is None else True
        auxiliary_fn = _dummy_auxiliary_fn if auxiliary_fn is None else auxiliary_fn

        if filter_method == 'bsf' and proposal_fn:
            raise AssertionError('Bootstrap Filter does not need extra proposal distribution!')
        if filter_method == 'ekf':
            transition_fn_grad = kwargs['transition_fn_grad']
            observation_fn_grad = kwargs['observation_fn_grad']
        else:
            transition_fn_grad = None
            observation_fn_grad = None
        if is_conditional and conditional_sample is None:
            raise AttributeError("No conditional sample and trajectory is provided!")

        init_seed, loop_seed = samplers.split_seed(seed, salt='particle_filter')
        num_observation_steps = ps.size0(tf.nest.flatten(observations)[0])
        num_timesteps = (
                1 + num_transitions_per_observation * (num_observation_steps - 1))

        # If trace criterion is `None`, we'll return only the final results.
        never_trace = lambda *_: False
        if trace_criterion_fn is None:
            static_trace_allocation_size = 0
            trace_criterion_fn = never_trace

        # Note: the initial weighted particles are already one-step, but without resampling
        initial_weighted_particles = _particle_filter_initial_weighted_particles(
            observations=observations,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            transition_fn_grad=transition_fn_grad,
            observation_fn_grad=observation_fn_grad,
            filter_method=filter_method,
            initial_state_prior=initial_state_prior,
            initial_state_proposal=initial_state_proposal,
            num_particles=num_particles,
            auxiliary_fn=auxiliary_fn,
            seed=init_seed)
        propose_and_update_log_weights_fn = (
            _particle_filter_propose_and_update_log_weights_fn(
                observations=observations,
                transition_fn=transition_fn,
                proposal_fn=proposal_fn,
                observation_fn=observation_fn,
                filter_method=filter_method,
                transition_fn_grad=transition_fn_grad,
                observation_fn_grad=observation_fn_grad,
                auxiliary_fn=auxiliary_fn,
                num_transitions_per_observation=num_transitions_per_observation,
            ))

        kernel = smc_kernel.SequentialMonteCarlo(
            propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
            resample_fn=resample_fn,
            resample_ess_num=resample_ess_num,
            unbiased_gradients=unbiased_gradients,
            is_conditional=is_conditional,
            conditional_sample=conditional_sample)

        # Use `trace_scan` rather than `sample_chain` directly because the latter
        # would force us to trace the state history (with or without thinning),
        # which is not always appropriate.
        def seeded_one_step(seed_state_results, _):
            seed, state, results = seed_state_results
            one_step_seed, next_seed = samplers.split_seed(seed)
            next_state, next_results = kernel.one_step(
                state, results, is_apf=is_apf, seed=one_step_seed)
            return next_seed, next_state, next_results

        final_seed_state_result, traced_results = loop_util.trace_scan(
            loop_fn=seeded_one_step,
            initial_state=(loop_seed,
                           initial_weighted_particles,
                           kernel.bootstrap_results(initial_weighted_particles)),
            elems=tf.ones([num_timesteps]),  # already executed the first
            trace_fn=lambda seed_state_results: trace_fn(*seed_state_results[1:]),
            trace_criterion_fn=(
                lambda seed_state_results: trace_criterion_fn(  # pylint: disable=g-long-lambda
                    *seed_state_results[1:])),
            static_trace_allocation_size=static_trace_allocation_size,
            parallel_iterations=parallel_iterations)

        if trace_criterion_fn is never_trace:
            # Return results from just the final step.
            traced_results = trace_fn(*final_seed_state_result[1:])
        # else:
        #     traced_results = _combine_initial_state(initial_weighted_particles, traced_results)

        return traced_results


def _check_resample_fn(method):
    resample_method = f"_resample_{method}"
    if hasattr(weighted_resampling, resample_method):
        return getattr(weighted_resampling, resample_method)
    else:
        raise AssertionError(f'No {method} resampling is implemented!')


def _particle_filter_initial_weighted_particles(observations,
                                                transition_fn,
                                                observation_fn,
                                                filter_method,
                                                transition_fn_grad,
                                                observation_fn_grad,
                                                initial_state_prior,
                                                initial_state_proposal,
                                                num_particles,
                                                auxiliary_fn,
                                                seed=None):
    """Initialize a set of weighted particles including the first observation."""
    # Propose an initial state.
    # No need to multiple with auxiliary function in the initial step
    if initial_state_proposal is None:
        initial_state = initial_state_prior.sample(num_particles, seed=seed)
        initial_log_weights = ps.zeros_like(
            initial_state_prior.log_prob(initial_state))
    else:
        observation = tf.nest.map_structure(
            lambda x, step=0: tf.gather(x, 0), observations)

        if filter_method == 'ekf':
            initial_state_dist = initial_state_proposal(initial_state_prior, observation,
                                                                    transition_fn, observation_fn,
                                                                    transition_fn_grad, observation_fn_grad)
        else:
            initial_state_dist = initial_state_proposal(observation)

        initial_state = initial_state_dist.sample(num_particles, seed=seed)
        initial_log_weights = (initial_state_prior.log_prob(initial_state) -
                               initial_state_dist.log_prob(initial_state))

    log_aux_weights = auxiliary_fn(0, initial_state, initial_log_weights, observations)

    # Return particles weighted by the initial observation.
    return smc_kernel.WeightedParticles(
        particles=initial_state,
        log_weights=initial_log_weights + _compute_observation_log_weights(
            step=0,
            particles=initial_state,
            observations=observations,
            observation_fn=observation_fn) + log_aux_weights)


def _particle_filter_propose_and_update_log_weights_fn(
        observations,
        transition_fn,
        proposal_fn,
        observation_fn,
        filter_method,
        transition_fn_grad,
        observation_fn_grad,
        auxiliary_fn,
        num_transitions_per_observation=1):
    """Build a function specifying a particle filter update step."""

    def propose_and_update_log_weights_fn(step, state, seed=None):
        particles, log_weights = state.particles, state.log_weights
        transition_dist = transition_fn(step, particles)
        assertions = _assert_batch_shape_matches_weights(
            distribution=transition_dist,
            weights_shape=ps.shape(log_weights),
            diststr='transition')

        observation_idx = step // num_transitions_per_observation
        observation = tf.nest.map_structure(
            lambda x, step=step: tf.gather(x, observation_idx), observations)

        if proposal_fn:
            if filter_method == 'ekf':
                proposal_dist = proposal_fn(step, particles, observation,
                                            transition_fn, observation_fn,
                                            transition_fn_grad, observation_fn_grad)
            else:
                proposal_dist = proposal_fn(step, particles, observation)
            assertions += _assert_batch_shape_matches_weights(
                distribution=proposal_dist,
                weights_shape=ps.shape(log_weights),
                diststr='proposal')
            proposed_particles = proposal_dist.sample(seed=seed)

            log_weights += (transition_dist.log_prob(proposed_particles) -
                            proposal_dist.log_prob(proposed_particles))
            # The normalizing constant E~q[p(x)/q(x)] is 1 in expectation,
            # so we reduce variance by dividing it out. Intuitively: the marginal
            # likelihood of a model with no observations is constant
            # (equal to 1.), so the transition and proposal distributions shouldn't
            # affect it.
            # log_weights = tf.nn.log_softmax(log_weights, axis=0)
        else:
            proposed_particles = transition_dist.sample(seed=seed)

        log_aux_weights = auxiliary_fn(step + 1, proposed_particles, log_weights, observations)

        with tf.control_dependencies(assertions):
            return smc_kernel.WeightedParticles(
                particles=proposed_particles,
                log_weights=log_weights + _compute_observation_log_weights(
                    step + 1, proposed_particles, observations, observation_fn,
                    num_transitions_per_observation=num_transitions_per_observation) + log_aux_weights)

    return propose_and_update_log_weights_fn


def _compute_observation_log_weights(step,
                                     particles,
                                     observations,
                                     observation_fn,
                                     num_transitions_per_observation=1):
    """Computes particle importance weights from an observation step.

    Args:
    step: int `Tensor` current step.
    particles: Nested structure of `Tensor`s, each of shape
      `concat([[num_particles, b1, ..., bN], event_shape])`, where
      `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observations: Nested structure of `Tensor`s, each of shape
      `concat([[num_observations, b1, ..., bN], event_shape])`
      where `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observation_fn: callable with signature
      `observation_dist = observation_fn(step, particles)`, producing
      a batch of distributions over the `observation` at the given `step`,
      one for each particle.
    num_transitions_per_observation: optional int `Tensor` number of times
      to apply the transition model between successive observation steps.
      Default value: `1`.
    Returns:
    log_weights: `Tensor` of shape `concat([num_particles, b1, ..., bN])`.
    """

    with tf.name_scope('compute_observation_log_weights'):
        step_has_observation = (
            # The second of these conditions subsumes the first, but both are
            # useful because the first can often be evaluated statically.
                ps.equal(num_transitions_per_observation, 1) |
                ps.equal(step % num_transitions_per_observation, 0))
        observation_idx = step // num_transitions_per_observation
        observation = tf.nest.map_structure(
            lambda x, step=step: tf.gather(x, observation_idx), observations)

        log_weights = observation_fn(step, particles).log_prob(observation)

        return tf.where(step_has_observation,
                        log_weights,
                        tf.zeros_like(log_weights))


def _assert_batch_shape_matches_weights(distribution, weights_shape, diststr):
    """Checks that all parts of a distribution have the expected batch shape."""
    shapes = [weights_shape] + tf.nest.flatten(distribution.batch_shape_tensor())
    static_shapes = [tf.get_static_value(ps.convert_to_shape_tensor(s))
                     for s in shapes]
    static_shapes_not_none = [s for s in static_shapes if s is not None]
    static_shapes_match = all([
        np.all(a == b)  # Also need to check for rank mismatch (below).
        for (a, b) in zip(static_shapes_not_none[1:],
                          static_shapes_not_none[:-1])])

    # Build a separate list of static ranks, since rank is often static even when
    # shape is not.
    ranks = [ps.rank_from_shape(s) for s in shapes]
    static_ranks = [int(r) for r in ranks if not tf.is_tensor(r)]
    static_ranks_match = all([a == b for (a, b) in zip(static_ranks[1:],
                                                       static_ranks[:-1])])

    msg = (
        "The {diststr} distribution's batch shape does not match the particle "
        "weights; a correct {diststr} distribution must return an independent "
        "log-density for each particle. You may be "
        "creating a joint distribution in which some parts do not depend on the "
        "previous particles, and/or you are creating an autobatched joint "
        "distribution without setting `batch_ndims`.".format(
            diststr=diststr))
    if not (static_ranks_match and static_shapes_match):
        raise ValueError(msg + ' ' +
                         'Weights have shape {}, but the distribution has batch '
                         'shape {}.'.format(
                             weights_shape, distribution.batch_shape))

    assertions = []
    if distribution.validate_args and any([s is None for s in static_shapes]):
        assertions = [assert_util.assert_equal(a, b, message=msg)
                      for a, b in zip(shapes[1:], shapes[:-1])]
    return assertions


def _dummy_auxiliary_fn(step, proposed_particles, log_weights, observations):
    return tf.zeros(log_weights.shape, dtype=log_weights.dtype)