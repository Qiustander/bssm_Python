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
"""Sequential Monte Carlo."""

import collections
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensorshape_util
import tensorflow as tf
from . import resampling as weighted_resampling
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'SequentialMonteCarlo',
    'SequentialMonteCarloResults',
    'WeightedParticles',
    'log_ess_from_log_weights',
    'adaptive_resampling',
]


# SequentialMonteCarlo `state` structure.
class WeightedParticles(collections.namedtuple(
    'WeightedParticles', ['particles', 'log_weights'])):
    """Particles with corresponding log weights.

  This structure serves as the `state` for the `SequentialMonteCarlo` transition
  kernel.

  Elements:
    particles: a (structure of) Tensor(s) each of shape
      `concat([[num_particles, b1, ..., bN], event_shape])`, where `event_shape`
      may differ across component `Tensor`s.
    log_weights: `float` `Tensor` of shape
      `[num_particles, b1, ..., bN]` containing a log importance weight for
      each particle, typically normalized so that
      `exp(reduce_logsumexp(log_weights, axis=0)) == 1.`. These must be used in
      conjunction with `particles` to compute expectations under the target
      distribution.

  In some contexts, particles may be stacked across multiple inference steps,
  in which case all `Tensor` shapes will be prefixed by an additional dimension
  of size `num_steps`.
  """


# SequentialMonteCarlo `kernel_results` structure.
class SequentialMonteCarloResults(collections.namedtuple(
    'SequentialMonteCarloResults',
    ['steps',
     'parent_indices',
     'incremental_log_marginal_likelihood',
     # Track both incremental and accumulated likelihoods so that users can get
     # the accumulated likelihood without needing to trace every step.
     'accumulated_log_marginal_likelihood',
     'seed',
     ])):
    """Auxiliary results from a Sequential Monte Carlo step.

  This structure serves as the `kernel_results` for the `SequentialMonteCarlo`
  transition kernel.

  Elements:
    steps: scalar int `Tensor` number of inference steps completed so far.
    parent_indices: `int` `Tensor` of shape `[num_particles, b1, ..., bN]`,
      such that `parent_indices[k]` gives the indice(s) of the particle(s) at
      the previous step from which the the `k`th current particle is
      immediately descended. See also
      `tfp.experimental.mcmc.reconstruct_trajectories`.
    incremental_log_marginal_likelihood: float `Tensor` of shape
      `[b1, ..., bN]`, giving the natural logarithm of an unbiased estimate of
      the ratio in normalizing constants incurred in the most recent step
      (typically this is the likelihood of observed data).
      Note that (by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true log ratio.
    cumulative_log_marginal_likelihood: float `Tensor` of shape
      `[b1, ..., bN]`, giving the natural logarithm of an unbiased estimate of
      the ratio in normalizing constants incurred since the initial step
      (typically this is the likelihood of observed data).
      Note that (by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true log ratio.
    seed: The seed used in one_step.

  In some contexts, results may be stacked across multiple inference steps,
  in which case all `Tensor` shapes will be prefixed by an additional dimension
  of size `num_steps`.
  """
    __slots__ = ()


def _dummy_indices_like(indices):
    """Returns dummy indices ([0, 1, 2, ...]) with batch shape like `indices`."""
    indices_shape = ps.shape(indices)
    num_particles = indices_shape[0]
    return tf.broadcast_to(
        ps.reshape(
            ps.range(num_particles),
            ps.pad([num_particles],
                   paddings=[[0, ps.rank_from_shape(indices_shape) - 1]],
                   constant_values=1)),
        indices_shape)


def _broadcast_resample(resample, particles):
    """Returns do_resample  with the same shape like `particles`."""
    broadcast_shape = ps.shape(particles)

    rank_diff = ps.rank(particles) - ps.rank(resample)  # 2 or 1

    resample = resample[..., tf.newaxis] if rank_diff == 2 else resample

    # Broadcast do_resample to the full shape of particles
    return tf.broadcast_to(resample, broadcast_shape)


def log_ess_from_log_weights(log_weights):
    """Computes log-ESS estimate from log-weights along axis=0."""
    with tf.name_scope('ess_from_log_weights'):
        log_weights = tf.math.log_softmax(log_weights, axis=0)
        return -tf.math.reduce_logsumexp(2 * log_weights, axis=0)


def adaptive_resampling(threshold):
    def ess_below_threshold(weighted_particles):
        """Determines if the effective sample size is much less than num_particles."""
        with tf.name_scope('ess_below_threshold'):
            num_particles = ps.size0(weighted_particles.log_weights)
            log_ess = log_ess_from_log_weights(weighted_particles.log_weights)
            return log_ess < (ps.log(num_particles) + ps.log(threshold))

    return ess_below_threshold


class SequentialMonteCarlo(kernel_base.TransitionKernel):
    """Sequential Monte Carlo transition kernel.

  Sequential Monte Carlo maintains a population of weighted particles
  representing samples from a sequence of target distributions. It is
  *not* a calibrated MCMC kernel: the transitions step through a sequence of
  target distributions, rather than trying to maintain a stationary
  distribution.
  """

    def __init__(self,
                 propose_and_update_log_weights_fn,
                 resample_fn,
                 resample_ess_num=0.5,
                 unbiased_gradients=True,
                 is_conditional=False,
                 conditional_sample=None,
                 name=None):
        """Initializes a sequential Monte Carlo transition kernel.

    Args:
      propose_and_update_log_weights_fn: Python `callable` with signature
        `new_weighted_particles = propose_and_update_log_weights_fn(step,
        weighted_particles, seed=None)`. Its input is a
        `tfp.experimental.mcmc.WeightedParticles` structure representing
        weighted samples (with normalized weights) from the `step`th
        target distribution, and it returns another such structure representing
        unnormalized weighted samples from the next (`step + 1`th) target
        distribution. This will typically include particles
        sampled from a proposal distribution `q(x[step + 1] | x[step])`, and
        weights that account for some or all of: the proposal density,
        a transition density `p(x[step + 1] | x[step]),
        observation weights `p(y[step + 1] | x[step + 1])`, and/or a backwards
        or 'L'-kernel `L(x[step] | x[step + 1])`. The (log) normalization
        constant of the weights is interpreted as the incremental (log) marginal
        likelihood.
      resample_fn: Resampling scheme specified as a `callable` with signature
        `indices = resample_fn(log_probs, event_size, sample_shape, seed)`,
        where `log_probs` is a `Tensor` of the same shape as `state.log_weights`
        containing a normalized log-probability for every current
        particle, `event_size` is the number of new particle indices to
        generate,  `sample_shape` is the number of independent index sets to
        return, and the  return value `indices` is an `int` Tensor of shape
        `concat([sample_shape, [event_size, B1, ..., BN])`. Typically one of
        `tfp.experimental.mcmc.resample_deterministic_minimum_error`,
        `tfp.experimental.mcmc.resample_independent`,
        `tfp.experimental.mcmc.resample_stratified`, or
        `tfp.experimental.mcmc.resample_systematic`.
        Default value: `tfp.experimental.mcmc.resample_systematic`.
      resample_criterion_fn: optional Python `callable` with signature
        `do_resample = resample_criterion_fn(weighted_particles)`,
        passed an instance of `tfp.experimental.mcmc.WeightedParticles`. The
        return value `do_resample`
        determines whether particles are resampled at the current step. The
        default behavior is to resample particles when the effective
        sample size falls below half of the total number of particles.
        Default value: `tfp.experimental.mcmc.ess_below_threshold`.
      unbiased_gradients: If `True`, use the stop-gradient
        resampling trick of Scibior, Masrani, and Wood [{scibor_ref_idx}] to
        correct for gradient bias introduced by the discrete resampling step.
        This will generally increase the variance of stochastic gradients.
        Default value: `True`.
      name: Python `str` name for ops created by this kernel.

    #### References

    [1] Adam Scibior, Vaden Masrani, and Frank Wood. Differentiable Particle
        Filtering without Modifying the Forward Pass. _arXiv preprint
        arXiv:2106.10314_, 2021. https://arxiv.org/abs/2106.10314
    """
        self._propose_and_update_log_weights_fn = propose_and_update_log_weights_fn
        self._resample_fn = resample_fn
        self._resample_criterion_fn = adaptive_resampling(resample_ess_num)
        self._unbiased_gradients = unbiased_gradients
        self._is_conditional = is_conditional
        self._conditional_sample = conditional_sample
        self._name = name or 'SequentialMonteCarlo'

    @property
    def is_calibrated(self):
        return False

    @property
    def name(self):
        return self._name

    @property
    def is_conditional(self):
        return self._is_conditional

    @property
    def conditional_sample(self):
        return self._conditional_sample

    @property
    def propose_and_update_log_weights_fn(self):
        return self._propose_and_update_log_weights_fn

    @property
    def resample_criterion_fn(self):
        return self._resample_criterion_fn

    @property
    def unbiased_gradients(self):
        return self._unbiased_gradients

    @property
    def resample_fn(self):
        return self._resample_fn

    def one_step(self, state, kernel_results, is_apf=False, seed=None):
        """Takes one Sequential Monte Carlo inference step.

    Args:
        is_apf:
      state: instance of `tfp.experimental.mcmc.WeightedParticles` representing
        the current particles with (log) weights. The `log_weights` must be
        a float `Tensor` of shape `[num_particles, b1, ..., bN]`. The
        `particles` may be any structure of `Tensor`s, each of which
        must have shape `concat([log_weights.shape, event_shape])` for some
        `event_shape`, which may vary across components.
      kernel_results: instance of
        `tfp.experimental.mcmc.SequentialMonteCarloResults` representing results
        from a previous step.
      auxiliary_fn: auxiliary function used in the Auxiliary Particle Filter. Dafault: None
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      state: instance of `tfp.experimental.mcmc.WeightedParticles` representing
        new particles with (log) weights.
      kernel_results: instance of
        `tfp.experimental.mcmc.SequentialMonteCarloResults`.
    """
        with tf.name_scope(self.name):
            with tf.name_scope('one_step'):
                state = WeightedParticles(*state)  # Canonicalize.

                seed = samplers.sanitize_seed(seed)
                proposal_seed, resample_seed = samplers.split_seed(seed)
                last_weights = state.log_weights

                # Propose new particles and update weights for this step, unless it's
                # the initial step, in which case, use the user-provided initial
                # particles and weights.
                proposed_state = self.propose_and_update_log_weights_fn(
                    # Propose state[t] from state[t - 1].
                    ps.maximum(0, kernel_results.steps - 1),
                    state,
                    seed=proposal_seed)

                is_initial_step = ps.equal(kernel_results.steps, 0)
                state = tf.nest.map_structure(
                    lambda a, b: tf.where(is_initial_step, a, b), state, proposed_state)

                if self.is_conditional:
                    # dim: # particles, state_dim
                    replace_sample = tf.gather(self.conditional_sample, indices=kernel_results.steps,
                                               axis=0)
                    state = state._replace(particles=tf.tensor_scatter_nd_update(state.particles,
                                                                                 indices=tf.constant([[0]]),
                                                                                 updates=[replace_sample]))

                incremental_log_marginal_likelihood = tf.nest.map_structure(
                    lambda a, b: tf.where(is_initial_step, a, b), tfp.math.reduce_logmeanexp(state.log_weights, axis=0),
                    tfp.math.reduce_logmeanexp(state.log_weights, axis=0) - \
                    tfp.math.reduce_logmeanexp(last_weights, axis=0))

                # do resampling first for x_{t-1} -> w_{t-1}
                do_resample = self.resample_criterion_fn(state)
                # Some batch elements may require resampling and others not, so
                # we first do the resampling for all elements, then select whether to
                # use the resampled values for each batch element according to
                # `do_resample`. If there were no batching, we might prefer to use
                # `tf.cond` to avoid the resampling computation on steps where it's not
                # needed---but we're ultimately interested in adaptive resampling
                # for statistical (not computational) purposes, so this isn't a
                # dealbreaker.
                # log weights_after_resampling = 0
                [
                    resampled_particles,
                    resample_indices,
                    weights_after_resampling
                ] = weighted_resampling.resample(
                    particles=state.particles,
                    # The `stop_gradient` here does not affect discrete resampling
                    # (which is nondifferentiable anyway), but avoids canceling out
                    # the gradient signal from the 'target' log weights, as described in
                    # Scibior, Masrani, and Wood (2021).
                    log_weights=tf.stop_gradient(state.log_weights),
                    resample_fn=self.resample_fn,
                    is_conditional=self.is_conditional,
                    target_log_weights=(state.log_weights
                                        if self.unbiased_gradients else None),
                    seed=resample_seed)
                do_resample_broadcasted = _broadcast_resample(do_resample, resampled_particles)
                resampled_particles = tf.nest.map_structure(
                    lambda r, p: tf.where(do_resample_broadcasted, r, p),
                    resampled_particles, state.particles)
                (resample_indices,
                 log_weights) = tf.nest.map_structure(
                    lambda r, p: tf.where(do_resample, r, p),
                    (resample_indices, weights_after_resampling),
                    (_dummy_indices_like(resample_indices),
                     state.log_weights))

                # In APF, the weights need to be adjusted, we place it after resamplinig
                gather_ancestors = lambda x: (  # pylint: disable=g-long-lambda
                    mcmc_util.index_remapping_gather(x, resample_indices, axis=0))
                adjust_apf_weights = tf.cond(tf.constant(is_apf, dtype=tf.bool),
                                             lambda: tf.nest.map_structure(gather_ancestors,
                                                                           tf.nn.log_softmax(log_weights, axis=0) -
                                                                           tf.nn.log_softmax(state.log_weights,
                                                                                             axis=0)),
                                             lambda: log_weights)

                # unnormalized
                last_adjust_aux_weights = tf.nest.map_structure(
                    lambda r, p: tf.where(do_resample, r, p),
                    adjust_apf_weights,
                    log_weights)

                tensorshape_util.set_shape(last_adjust_aux_weights, state.log_weights.shape)

                # if self.is_conditional:
                #     # update the ancestor
                #     resample_indices = tf.concat([tf.zeros([1, *ps.shape(resample_indices)[1:]],
                #                                            dtype=resample_indices.dtype),
                #                                   resample_indices[1:]],
                #                                  axis=0)

            return (WeightedParticles(particles=resampled_particles,
                                      log_weights=last_adjust_aux_weights),
                    SequentialMonteCarloResults(
                        steps=kernel_results.steps + 1,
                        parent_indices=resample_indices,
                        incremental_log_marginal_likelihood=(
                            incremental_log_marginal_likelihood),
                        accumulated_log_marginal_likelihood=(
                                kernel_results.accumulated_log_marginal_likelihood +
                                incremental_log_marginal_likelihood),
                        seed=seed))

    def bootstrap_results(self, init_state):
        """
        Result at the x_0
        Args:
            init_state: normalized log weights and particles
        Returns: SMC result at x_0
        """
        with tf.name_scope(self.name):
            with tf.name_scope('bootstrap_results'):
                init_state = WeightedParticles(*init_state)

                batch_zeros = tf.zeros(
                    ps.shape(init_state.log_weights)[1:],
                    dtype=init_state.log_weights.dtype)

                return SequentialMonteCarloResults(
                    steps=0,
                    parent_indices=_dummy_indices_like(init_state.log_weights),
                    incremental_log_marginal_likelihood=batch_zeros,
                    accumulated_log_marginal_likelihood=batch_zeros,
                    seed=samplers.zeros_seed())
