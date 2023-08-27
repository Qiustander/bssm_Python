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


def _covariance_getter_fn(kernel_results):
    """Getter for `random walk state function` so it can be inspected."""
    return unnest.get_innermost(kernel_results, 'proposed_state')


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
    """Adapts the inner kernel's `covariance matrix` based on `log_accept_prob`.

  Modification modifies the `covariance matrix` of
  the inner kernel based on the value of `log_accept_prob`.

  In general, adaptation prevents the chain from reaching a stationary
  distribution, so obtaining consistent samples requires `num_adaptation_steps`
  be set to a value [somewhat smaller][2] than the number of burnin steps.
  However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
  value during development in order to inspect the behavior of the chain during
  adaptation.


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

  [2]"

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

    def covariance_getter_fn(self, kernel_results):
        return self._parameters['covariance_getter_fn'](kernel_results)

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
            seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
            inner_kwargs = dict(seed=seed)
            # inner_kwargs = {} if seed is None else dict(seed=seed)
            new_state, new_inner_results = self.inner_kernel.one_step(
                current_state, inner_results, **inner_kwargs)

            # Get the new results
            log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
            log_target_accept_prob = tf.math.log(
                tf.cast(previous_kernel_results.target_accept_prob,
                        dtype=log_accept_prob.dtype))

            # in RAM, we directly use the current proposed state to update the covariance
            state_parts = tf.nest.flatten(current_state)
            # get the new state_results
            new_state_parts = tf.nest.flatten(new_state)
            log_accept_prob_rank = prefer_static.rank(log_accept_prob)

            new_covariance_parts = []
            for covariance_part, state_part, new_state_part in zip(covariance_parts, state_parts, new_state_parts):
                # Compute new covariance matrix for each covariance part.
                # It is worth noting that the covariance matrix should be [1,1] matrix for multidimensional state part.
                # Example:
                # state_part has shape   [2, 3, 4] + [5, 5] (batch + event, state_part is a matrix, e.g., covariance matrix)
                # covariance_part has shape   [2, 3, 4] + [1, 1] (the covariance_part is a 1*1 matrix for the `scalar' parameter)
                # state_part has shape   [2, 3, 4] + [5, 1]/ [5] (batch + event, state_part is a stack of scalar parameters)
                # covariance_part has shape   [2, 3, 4] + [5, 5] (the covariance_part is a 5*5 matrix)
                # The size of the covariance_part is fixed from the initialization
                state_part_rank = prefer_static.rank(state_part)
                if state_part_rank == log_accept_prob_rank:
                    raise ValueError("Rank of parameters must be larger than the log probability")
                rank_diff = state_part_rank - log_accept_prob_rank
                shape_state = prefer_static.shape(state_part)[-1]
                is_scalar_cov = True if rank_diff == 2 and shape_state != 1 else False

                # random walk at the newest state, need to do inverse since no trace of this part,
                random_walk_part = tf.linalg.triangular_solve(covariance_part, new_state_part - state_part, lower=True)

                adapted_state = tf.where(is_scalar_cov,
                                         covariance_part * tf.math.l2_normalize(random_walk_part,
                                                                                axis=prefer_static.range(start=-1,
                                                                                                         limit=-rank_diff - 1,
                                                                                                         delta=-1)),
                                         tf.matmul(covariance_part, tf.math.l2_normalize(random_walk_part,
                                                                                axis=prefer_static.range(start=-1,
                                                                                                         limit=-rank_diff - 1,
                                                                                                         delta=-1)))
                                         ) \
                                * tf.sqrt(tf.math.minimum(
                    1., prefer_static.size(random_walk_part)*tf.pow(previous_kernel_results.step, -previous_kernel_results.adaptation_rate)
                )* tf.abs(log_accept_prob - log_target_accept_prob))

                # reduce the last dimension if necessary.
                adapted_state = tf.cond(tf.equal(shape_state, 1),
                                        lambda: tf.squeeze(shape_state, axis=-1),
                                        lambda: adapted_state)

                new_covariance_part = mcmc_util.choose(
                    log_accept_prob > log_target_accept_prob,
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=1.0),
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=-1.0))
                # TODO: replace the negative value in the diagonal term to ensure the positive value

                new_covariance_parts.append(
                    tf.where(previous_kernel_results.step < self.num_adaptation_steps,
                             new_covariance_part,
                             covariance_part))
            new_covariance = tf.nest.pack_sequence_as(covariance_parts, new_covariance_parts)

            return new_state, previous_kernel_results._replace(
                inner_results=new_inner_results,
                step=1 + previous_kernel_results.step,
                new_covariance=new_covariance)

    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'covariance_adaptation', 'bootstrap_results')):
            inner_results = self.inner_kernel.bootstrap_results(init_state)
            log_accept_prob = self.log_accept_prob_getter_fn(inner_results)
            log_accept_prob_rank = prefer_static.rank(log_accept_prob)
            # TODO: Need boardcasting
            covariance_parts = []
            for state_part in zip(init_state):
                state_part_rank = prefer_static.rank(state_part)
                if state_part_rank == log_accept_prob_rank:
                    raise ValueError("Rank of parameters must be larger than the log probability")
                rank_diff = state_part_rank - log_accept_prob_rank
                shape_state = prefer_static.shape(state_part)[-1]
                is_scalar_cov = True if rank_diff == 2 and shape_state != 1 else False
                covariance_parts.append(tf.cond(is_scalar_cov,
                                        lambda: tf.eye(1),
                                        lambda: tf.eye(shape_state)))
            return CovarianceAdaptationResults(
                inner_results=inner_results,
                step=tf.constant(0, dtype=tf.int32),
                target_accept_prob=self.parameters['target_accept_prob'],
                adaptation_rate=self.parameters['adaptation_rate'],
                new_covariance=covariance_parts)

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
