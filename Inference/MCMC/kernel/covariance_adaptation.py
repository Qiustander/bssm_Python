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
"""SimpleStepSizeAdaptation TransitionKernel."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn
from tensorflow_probability.python.internal import distribute_lib


def _covariance_setter_fn(kernel_results, new_covariance):
    """Setter for `random walk state function` so it can be adapted."""
    return unnest.replace_innermost(kernel_results, new_state_fn=random_walk_normal_fn(new_covariance))


# def _covariance_getter_fn(kernel_results):
#     """Getter for `random walk state function` so it can be inspected."""
#     return unnest.get_innermost(kernel_results, 'proposed_state')


def _covariance_getter_fn(kernel_results, state_parts, experimental_shard_axis_names=None):
    """Getter for `random walk state function` so it can be inspected."""
    seed = unnest.get_innermost(kernel_results, 'seed')
    part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
    part_seeds = distribute_lib.fold_in_axis_index(
        part_seeds, experimental_shard_axis_names)
    return [
          samplers.normal(  # pylint: disable=g-complex-comprehension
              mean=0.,
              stddev=1.0,
              shape=prefer_static.shape(state_part),
              dtype=dtype_util.base_dtype(state_part.dtype),
              seed=seed_part)
          for state_part, seed_part
          in zip(state_parts, part_seeds)
      ]


def _log_accept_prob_getter_fn(kernel_results):
    log_accept_ratio = unnest.get_innermost(kernel_results, 'log_accept_ratio')
    safe_accept_ratio = tf.where(
        tf.math.is_finite(log_accept_ratio),
        log_accept_ratio,
        tf.constant(-np.inf, dtype=log_accept_ratio.dtype))
    return tf.minimum(safe_accept_ratio, 0.)


def get_differing_dims(a, b):
    # Get the indices of dimensions where shapes of `a` and `b` differ.
    # `a` is allowed to have fewer dimensions than `b`.
    if (not tensorshape_util.is_fully_defined(a.shape) or
            not tensorshape_util.is_fully_defined(b.shape)):
        return tf.where(
            tf.not_equal(tf.shape(a), tf.shape(b)[:tf.rank(a)]))[:, 0]
    a_shape = np.int32(a.shape)
    b_shape = np.int32(b.shape)
    return np.where(a_shape != b_shape[:len(a_shape)])[0]


class CovarianceAdaptationResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'CovarianceAdaptation',
        [
            'inner_results',
            'target_accept_prob',
            'adaptation_rate',
            'step',
            'new_covariance',
        ])):
    """Results of the CovarianceAdaptation TransitionKernel.

  Attributes:
    inner_results: Results of the inner kernel.
    target_accept_prob: Floating point scalar `Tensor`. Target accept
      probability.
    adaptation_rate: Floating point scalar `Tensor`. Fraction by which to adjust
      the step size during each step.
    step: Int32 scalar `Tensor`. The current step number as perceived by this
      kernel. Increases by 1 for every call to `one_step`.
    new_covariance:  Floating point scalar `Tensor` or a list thereof (one for
      each `state_part`). Step size that will be passed to the inner kernel
      during the next step.
  """
    __slots__ = ()


class CovarianceAdaptation(kernel_base.TransitionKernel):
    """Adapts the inner kernel's `step_size` based on `log_accept_prob`.

  The simple policy multiplicatively increases or decreases the `step_size` of
  the inner kernel based on the value of `log_accept_prob`. It is based on
  [equation 19 of Andrieu and Thoms (2008)][1]. Given enough steps and small
  enough `adaptation_rate` the median of the distribution of the acceptance
  probability will converge to the `target_accept_prob`. A good target
  acceptance probability depends on the inner kernel. If this kernel is
  `HamiltonianMonteCarlo`, then 0.6-0.9 is a good range to aim for. For
  `RandomWalkMetropolis` this should be closer to 0.25. See the individual
  kernels' docstrings for guidance.

  In general, adaptation prevents the chain from reaching a stationary
  distribution, so obtaining consistent samples requires `num_adaptation_steps`
  be set to a value [somewhat smaller][2] than the number of burnin steps.
  However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
  value during development in order to inspect the behavior of the chain during
  adaptation.

  The step size is assumed to broadcast with the chain state, potentially having
  leading dimensions corresponding to multiple chains. When there are fewer of
  those leading dimensions than there are chain dimensions, the corresponding
  dimensions in the `log_accept_prob` are averaged (in the direct space, rather
  than the log space) before being used to adjust the step size. This means that
  this kernel can do both cross-chain adaptation, or per-chain step size
  adaptation, depending on the shape of the step size.

  For example, if your problem has a state with shape `[S]`, your chain state
  has shape `[C0, C1, Y]` (meaning that there are `C0 * C1` total chains) and
  `log_accept_prob` has shape `[C0, C1]` (one acceptance probability per chain),
  then depending on the shape of the step size, the following will happen:

  - Step size has shape [], [S] or [1], the `log_accept_prob` will be averaged
    across its `C0` and `C1` dimensions. This means that you will learn a shared
    step size based on the mean acceptance probability across all chains. This
    can be useful if you don't have a lot of steps to adapt and want to average
    away the noise.

  - Step size has shape [C1, 1] or [C1, S], the `log_accept_prob` will be
    averaged across its `C0` dimension. This means that you will learn a shared
    step size based on the mean acceptance probability across chains that share
    the coordinate across the `C1` dimension. This can be useful when the `C1`
    dimension indexes different distributions, while `C0` indexes replicas of a
    single distribution, all sampled in parallel.

  - Step size has shape [C0, C1, 1] or [C0, C1, S], then no averaging will
    happen. This means that each chain will learn its own step size. This can be
    useful when all chains are sampling from different distributions. Even when
    all chains are for the same distribution, this can help during the initial
    warmup period.

  - Step size has shape [C0, 1, 1] or [C0, 1, S], the `log_accept_prob` will be
    averaged across its `C1` dimension. This means that you will learn a shared
    step size based on the mean acceptance probability across chains that share
    the coordinate across the `C0` dimension. This can be useful when the `C0`
    dimension indexes different distributions, while `C1` indexes replicas of a
    single distribution, all sampled in parallel.

  By default, the averaging function used above is the arithmetic mean, which is
  not robust to stuck chains (e.g. average of one chain with `p_accept = 0` and
  three chains with `p_accept = 1` will result in an average `p_accept = 0.75`,
  which will cause this kernel keep the step size roughly the same rather than
  reducing it to unstick the stuck chain). A more robust choice would be to set
  `reduce_fn` argument to `tfp.math.reduce_log_harmonic_mean_exp` [3]. Note,
  however, that the harmonic mean of a set of numbers is usually smaller than
  the arithmetic mean, so its use will typically produce smaller than optimal
  step sizes even for well behaved target distributions.

  #### References

  [1]: Andrieu, Christophe, Thoms, Johannes. A tutorial on adaptive MCMC.
       _Statistics and Computing_, 2008.
       https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf

  """

    def __init__(self,
                 inner_kernel,
                 num_adaptation_steps,
                 target_accept_prob=0.234,
                 adaptation_rate=0.01, # adaption rate should be a fixed constant
                 covariance_setter_fn=_covariance_setter_fn,
                 covariance_getter_fn=_covariance_getter_fn,
                 log_accept_prob_getter_fn=_log_accept_prob_getter_fn,
                 reduce_fn=reduce_logmeanexp,
                 experimental_reduce_chain_axis_names=None,
                 validate_args=False,
                 name=None):
        """Implement robust adaptive Metropolis Algorithm.

    The default setter_fn and the getter_fn callbacks assume that the inner
    kernel produces kernel results structurally the same as the inner kernel.

    Args:
      inner_kernel: `TransitionKernel`-like object.
      num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
        during which to adjust the step size. This may be greater, less than, or
        equal to the number of burnin steps.
      target_accept_prob: A floating point `Tensor` representing desired
        acceptance probability. Must be a positive number less than 1. This can
        either be a scalar, or have shape [num_chains]. Default value: `0.75`
        (the [center of asymptotically optimal rate for HMC][1]).
      adaptation_rate: `Tensor` representing amount to scale the current
        `step_size`.
      covariance_setter_fn: A callable with the signature
        `(kernel_results, new_covariance) -> new_kernel_results` where
        `kernel_results` are the results of the `inner_kernel`, `new_covariance`
        is a `Tensor` or a nested collection of `Tensor`s with the same
        structure as returned by the `covariance_getter_fn`, and
        `new_kernel_results` are a copy of `kernel_results` with the step
        size(s) set.
      covariance_getter_fn: A callable with the signature
        `(kernel_results) -> step_size` where `kernel_results` are the results
        of the `inner_kernel`, and `step_size` is a floating point `Tensor` or a
        nested collection of such `Tensor`s.
      log_accept_prob_getter_fn: A callable with the signature
        `(kernel_results) -> log_accept_prob` where `kernel_results` are the
        results of the `inner_kernel`, and `log_accept_prob` is a floating point
        `Tensor`. `log_accept_prob` can either be a scalar, or have shape
        [num_chains]. If it's the latter, `step_size` should also have the same
        leading dimension.
      reduce_fn: A callable with signature `(input_tensor, axis, keepdims) ->
        tensor` that returns a log-reduction of `log_accept_prob`, typically
        some sort of mean. By default, this performs an arithmetic mean.
      experimental_reduce_chain_axis_names: A `str` or list of `str`s indicating
        the named axes that should additionally reduced during the log-reduction
        of `log_accept_prob`.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'simple_step_size_adaptation'.

    #### References

    [1]:
    """

        inner_kernel = mcmc_util.enable_store_parameters_in_results(inner_kernel)

        with tf.name_scope(mcmc_util.make_name(
                name, 'covariance_adaptation', '__init__')) as name:
            dtype = dtype_util.common_dtype([target_accept_prob, adaptation_rate],
                                            tf.float32)
            target_accept_prob = tf.convert_to_tensor(
                target_accept_prob, dtype=dtype, name='target_accept_prob')
            adaptation_rate = tf.convert_to_tensor(
                adaptation_rate, dtype=dtype, name='adaptation_rate')
            num_adaptation_steps = tf.convert_to_tensor(
                num_adaptation_steps,
                dtype=tf.int32,
                name='num_adaptation_steps')

            target_accept_prob = _maybe_validate_target_accept_prob(
                target_accept_prob, validate_args)

        self._parameters = dict(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
            adaptation_rate=adaptation_rate,
            covariance_setter_fn=covariance_setter_fn,
            covariance_getter_fn=covariance_getter_fn,
            log_accept_prob_getter_fn=log_accept_prob_getter_fn,
            reduce_fn=reduce_fn,
            experimental_reduce_chain_axis_names=(
                experimental_reduce_chain_axis_names),
            name=name,
        )

    @property
    def inner_kernel(self):
        return self._parameters['inner_kernel']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def num_adaptation_steps(self):
        return self._parameters['num_adaptation_steps']

    def covariance_setter_fn(self, kernel_results, new_covariance):
        return self._parameters['covariance_setter_fn'](kernel_results,
                                                        new_covariance)

    def covariance_getter_fn(self, kernel_results, state_parts):
        return self._parameters['covariance_getter_fn'](kernel_results, state_parts)

    def log_accept_prob_getter_fn(self, kernel_results):
        return self._parameters['log_accept_prob_getter_fn'](kernel_results)

    def reduce_fn(self, input_tensor, axis, keepdims,
                  experimental_named_axis=None):
        if experimental_named_axis is None:
            return self._parameters['reduce_fn'](
                input_tensor, axis=axis, keepdims=keepdims)
        return self._parameters['reduce_fn'](
            input_tensor, axis=axis, keepdims=keepdims,
            experimental_named_axis=experimental_named_axis)

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    def one_step(self, current_state, previous_kernel_results, seed=None):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'covariance_adaptation', 'one_step')):
            # Set the covariance.
            covariance_parts = previous_kernel_results.new_covariance
            inner_results = self.covariance_setter_fn(
                previous_kernel_results.inner_results,
                covariance_parts)

            # Step the inner kernel.
            inner_kwargs = {} if seed is None else dict(seed=seed)
            new_state, new_inner_results = self.inner_kernel.one_step(
                current_state, inner_results, **inner_kwargs)

            # Get the new covariance
            log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
            log_target_accept_prob = tf.math.log(
                tf.cast(previous_kernel_results.target_accept_prob,
                        dtype=log_accept_prob.dtype))

            # in RAM, we directly use the current proposed state to update the covariance
            state_parts = tf.nest.flatten(current_state)
            random_walk_parts = self.covariance_getter_fn(new_inner_results, state_parts)
            log_accept_prob_rank = prefer_static.rank(log_accept_prob)

            new_covariance_parts = []
            for covariance_part, state_part, random_walk_part in zip(covariance_parts, state_parts, random_walk_parts):
                # Compute new step sizes for each step size part. If step size part has
                # smaller rank than the corresponding state part, then the difference is
                # averaged away in the log accept prob.
                #
                # Example:
                # state_part has shape       [2, 3, 4] + [5, 6]  (batch + event)
                # covariance_part has shape          [4] + [5, 6]
                # log_accept_prob has shape         [2, 3, 4]
                #
                # i.e., we have a batch of chains of shape [2, 3, 4], and 4 mass
                # matrices, each being shared across a [2, 3]-batch of chains. Note
                # this division is inferred from the shapes of the state part, the
                # log_prob, and the user-provided  covariance.

                num_reduce_dims = prefer_static.minimum(
                    log_accept_prob_rank,
                    prefer_static.rank(state_part) - prefer_static.rank(covariance_part))
                reduced_log_accept_prob = self.reduce_fn(
                    log_accept_prob,
                    axis=prefer_static.range(num_reduce_dims),
                    keepdims=False,
                    experimental_named_axis=self.experimental_reduce_chain_axis_names)
                # reduced_log_accept_prob must broadcast into step_size_part on the
                # left, so we do an additional reduction over dimensions where their
                # shapes differ.
                reduce_indices = get_differing_dims(reduced_log_accept_prob,
                                                    covariance_part)
                reduced_log_accept_prob = self.reduce_fn(
                    reduced_log_accept_prob, axis=reduce_indices, keepdims=True)

                adapted_state = covariance_part @ tf.math.l2_normalize(random_walk_part) * tf.sqrt(tf.math.minimum(
                    1., prefer_static.size(random_walk_part)*tf.pow(previous_kernel_results.step, -previous_kernel_results.adaptation_rate)
                )* tf.abs(reduced_log_accept_prob - log_target_accept_prob))

                new_covariance_part = mcmc_util.choose(
                    reduced_log_accept_prob > log_target_accept_prob,
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=1.0),
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=-1.0))

                new_covariance_parts.append(
                    tf.where(previous_kernel_results.step < self.num_adaptation_steps,
                             new_covariance_part,
                             step_size_part))
            new_covariance = tf.nest.pack_sequence_as(step_size, new_covariance_parts)

            return new_state, previous_kernel_results._replace(
                inner_results=new_inner_results,
                step=1 + previous_kernel_results.step,
                new_covariance=new_covariance)

    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'simple_step_size_adaptation', 'bootstrap_results')):
            inner_results = self.inner_kernel.bootstrap_results(init_state)
            step_size = self.covariance_getter_fn(inner_results)
            return CovarianceAdaptationResults(
                inner_results=inner_results,
                step=tf.constant(0, dtype=tf.int32),
                target_accept_prob=self.parameters['target_accept_prob'],
                adaptation_rate=self.parameters['adaptation_rate'],
                new_covariance=step_size)

    @property
    def is_calibrated(self):
        return self.inner_kernel.is_calibrated

    @property
    def experimental_shard_axis_names(self):
        return self.inner_kernel.experimental_shard_axis_names

    def experimental_with_shard_axes(self, shard_axis_names):
        return self.copy(
            inner_kernel=self.inner_kernel.experimental_with_shard_axes(
                shard_axis_names))

    @property
    def experimental_reduce_chain_axis_names(self):
        return self._parameters['experimental_reduce_chain_axis_names']


def _maybe_validate_target_accept_prob(target_accept_prob, validate_args):
    """Validates that target_accept_prob is in (0, 1)."""
    if not validate_args:
        return target_accept_prob
    with tf.control_dependencies([
        assert_util.assert_positive(
            target_accept_prob, message='`target_accept_prob` must be > 0.'),
        assert_util.assert_less(
            target_accept_prob,
            tf.ones_like(target_accept_prob),
            message='`target_accept_prob` must be < 1.')
    ]):
        return tf.identity(target_accept_prob)
