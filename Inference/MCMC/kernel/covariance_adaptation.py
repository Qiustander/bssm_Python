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
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


def _covariance_setter_fn(kernel_results, new_covariance):
    """Setter for `random walk state function` so it can be adapted."""
    return unnest.replace_innermost(kernel_results, random_walk_cov=new_covariance)


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
                 initial_scale=0.2,
                 adaptation_rate=2/3, # adaption rate should be a fixed constant
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
            initial_scale = tf.convert_to_tensor(
                initial_scale, dtype=dtype, name='initial_scale')
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
            initial_scale=initial_scale,
            experimental_reduce_chain_axis_names=(
                experimental_reduce_chain_axis_names),
            name=name,
        )

    @property
    def inner_kernel(self):
        return self._parameters['inner_kernel']

    @property
    def initial_scale(self):
        return self._parameters['initial_scale']

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
            new_state, new_inner_results = self.inner_kernel.one_step(
                current_state, inner_results, **inner_kwargs)
            is_accept_propose = unnest.get_innermost(new_inner_results, "is_accepted")

            # get the new state for random walk
            new_proposed_state = self.covariance_getter_fn(new_inner_results)

            # Use the probability in original form
            log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
            accept_prob = tf.exp(log_accept_prob)
            accept_prob = mcmc_util.choose(is_accept_propose,
                                           accept_prob,
                                           tf.zeros_like(accept_prob, dtype=accept_prob.dtype))

            # in RAM, we directly use the current proposed state to update the covariance
            state_parts = tf.nest.flatten(current_state)
            new_state_parts = tf.nest.flatten(new_proposed_state)
            log_accept_prob_rank = prefer_static.rank(log_accept_prob)

            new_covariance_parts = []
            for covariance_part, state_part, new_state_part in zip(covariance_parts, state_parts, new_state_parts):
                # Compute new covariance matrix for each covariance part.
                # It is worth noting that the covariance matrix should be [1,1] matrix for multidimensional state part.
                # Example:
                # state_part has shape   [2, 3, 4] + [5] (batch + event, state_part is a stack of scalar parameters)
                # covariance_part has shape   [2, 3, 4] + [5, 5] (the covariance_part is a 5*5 matrix)
                # The size of the covariance_part is fixed from the initialization
                state_part_rank = prefer_static.rank(state_part)
                rank_diff = state_part_rank - log_accept_prob_rank
                shape_state = prefer_static.shape(state_part)[-1]
                is_scalar_cov = True if rank_diff == 2 and shape_state != 1 else False
                if is_scalar_cov:
                    raise ValueError("Input state is a matrix, which is not applicable to this adaption. Please"
                                     "flatten to vector, or use naive random walk Metropolis.")

                # add one dimension to smoothly conduct matrix multiplication
                state_part = state_part[..., tf.newaxis]
                new_state_part = new_state_part[..., tf.newaxis]

                # random walk at the newest state, need to do inverse since no trace of this part,
                random_walk_part = tf.linalg.triangular_solve(covariance_part, new_state_part - state_part, lower=True)

                # RAM formula
                left_multiply = tf.matmul(covariance_part, tf.math.l2_normalize(random_walk_part,
                                                                                axis=1))
                power_part = tf.pow(tf.cast(previous_kernel_results.step+1,
                                                                            dtype=random_walk_part.dtype), -previous_kernel_results.adaptation_rate)
                scale_multiply = tf.sqrt(tf.math.minimum(
                    1., prefer_static.shape(random_walk_part)[1]*power_part
                )* tf.abs(accept_prob - previous_kernel_results.target_accept_prob))

                adapted_state = tf.einsum('b..., b -> b...', left_multiply, scale_multiply)

                # adapted_state = tf.cond(state_part_rank < 2,
                #                          lambda: left_multiply*scale_multiply,
                #                          lambda: tf.einsum('b..., b -> b...', left_multiply, scale_multiply))
                # reduce the last dimension
                adapted_state = tf.squeeze(adapted_state, axis=-1)

                new_covariance_part = mcmc_util.choose(
                    accept_prob > previous_kernel_results.target_accept_prob,
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=1.0),
                    tfp.math.cholesky_update(covariance_part, adapted_state, multiplier=-1.0))

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
        # we have to store the covariance from the beginning since random walk Metropolis
        # does not store results of the matrix
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'covariance_adaptation', 'bootstrap_results')):
            # should use inner results as init state (if inner kernel is transformed results)
            inner_results = self.inner_kernel.bootstrap_results(init_state)
            log_accept_prob = self.log_accept_prob_getter_fn(inner_results)
            log_accept_prob_rank = prefer_static.rank(log_accept_prob)
            state_parts = tf.nest.flatten(init_state)
            # if _is_transformed(inner_results):
            #     state_parts = tf.nest.flatten(_get_transformed_states(inner_results))
            # else:
            #     state_parts = tf.nest.flatten(init_state)
            covariance_parts = []
            # can not use zip
            for state_part in state_parts:
                state_part_rank = prefer_static.rank(state_part)

                rank_diff = state_part_rank - log_accept_prob_rank
                shape_state = prefer_static.shape(state_part)[-1]
                is_scalar_cov = True if rank_diff == 2 else False
                if is_scalar_cov:
                    raise ValueError("Input state is a matrix, which is not applicable to this adaption. Please"
                                     "flatten to vector, or use naive random walk Metropolis.")
                covariance_mtx = self.initial_scale*tf.where(state_part_rank < 2,
                                          tf.eye(shape_state),
                                         tf.eye(shape_state, batch_shape=prefer_static.shape(state_part)[:-1]))
                covariance_parts.append(covariance_mtx)

                # need to replace all the covariance matrix to ensure the consistence
                inner_results = _deep_replace(inner_results, 'random_walk_cov', covariance_parts)

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


def _deep_replace(named_tuple, attr_to_replace, new_value):
    """Function to recursively replace all existing attributes in nested namedtuples"""
    if hasattr(named_tuple, attr_to_replace):
        return named_tuple._replace(**{attr_to_replace: new_value})
    else:
        updated_dict = {}
        for field_name, field_value in named_tuple._asdict().items():
            if isinstance(field_value, tuple) and hasattr(field_value, '_fields'):  # It's a namedtuple
                updated_dict[field_name] = _deep_replace(field_value, attr_to_replace, new_value)
            else:
                updated_dict[field_name] = field_value
        return named_tuple._replace(**updated_dict)


def _is_transformed(results):
    return unnest.has_nested(results, "transformed_state")


def _get_transformed_states(results):
    return unnest.get_innermost(results, "transformed_state")