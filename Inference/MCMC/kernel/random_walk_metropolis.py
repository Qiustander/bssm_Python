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
"""Random Walk Metropolis (RWM) Transition Kernel."""

import collections
# Dependency imports

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import RandomWalkMetropolis as RandomWalkMetropolisBase
from tensorflow_probability.python.mcmc.random_walk_metropolis import UncalibratedRandomWalk as RandomWalkBase

tfd = tfp.distributions

__all__ = [
    'random_walk_normal_fn',
    'random_walk_uniform_fn',
    'RandomWalkMetropolis',
    'UncalibratedRandomWalk',
]


class UncalibratedRandomWalkResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedRandomWalkResults',
        [
            'log_acceptance_correction',
            'target_log_prob',  # For "next_state".
            'seed',
            'random_walk_cov',
        ])
):
    """Internal state and diagnostics for Random Walk MH."""
    __slots__ = ()


def random_walk_normal_fn(scale=1., name=None):
    """Returns a callable that adds a random normal perturbation to the input.

  This function returns a callable that accepts a Python `list` of `Tensor`s of
  any shapes and `dtypes`  representing the state parts of the `current_state`
  and a random seed. The supplied argument `scale` must be a `Tensor` or Python
  `list` of `Tensor`s representing the scale of the generated
  proposal. `scale` must broadcast with the state parts of `current_state`.
  The callable adds a sample from a zero-mean normal distribution with the
  supplied scales to each state part and returns a same-type `list` of `Tensor`s
  as the state parts of `current_state`.

  Args:
    scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
      controlling the scale of the normal proposal distribution.
    name: Python `str` name prefixed to Ops created by this function.
        Default value: 'random_walk_normal_fn'.

  Returns:
    random_walk_normal_fn: A callable accepting a Python `list` of `Tensor`s
      representing the state parts of the `current_state` and an `int`
      representing the random seed to be used to generate the proposal. The
      callable returns the same-type `list` of `Tensor`s as the input and
      represents the proposal for the RWM algorithm.
  """

    def _fn(state_parts, seed, experimental_shard_axis_names=None):
        """Adds a normal perturbation to the input state.

    Args:
      state_parts: A list of `Tensor`s of any shape and real dtype representing
        the state parts of the `current_state` of the Markov chain.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.

    Returns:
      perturbed_state_parts: A Python `list` of The `Tensor`s. Has the same
        shape and type as the `state_parts`.

    Raises:
      ValueError: if `scale` does not broadcast with `state_parts`.
    """
        with tf.name_scope(name or 'random_walk_normal_fn'):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError('`scale` must broadcast with `state_parts`.')

            part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
            part_seeds = distribute_lib.fold_in_axis_index(
                part_seeds, experimental_shard_axis_names)

            def _scale_(scale_part, state_part):

                state_part_shape = ps.shape(state_part)[-1]
                rank_scale = ps.rank(scale_part)
                size_scale = ps.size(scale_part)

                if rank_scale == 1 and (size_scale != state_part_shape and size_scale != 1):
                    raise ValueError('Scale must be length 1, or the same as the state.')

                scale_with_shape = tf.where(rank_scale < 2,
                                            tf.eye(state_part_shape)*scale_part,
                                            scale_part)

                return scale_with_shape

            next_state_parts = [
                tfd.MultivariateNormalTriL(
                    loc=state_part,
                    scale_tril=_scale_(scale_part, state_part)
                ).sample(seed=seed_part)
                for scale_part, state_part, seed_part
                in zip(scales, state_parts, part_seeds)
            ]

            return next_state_parts

    return _fn


def random_walk_uniform_fn(scale=1., name=None):
    """Returns a callable that adds a random uniform perturbation to the input.

  For more details on `random_walk_uniform_fn`, see
  `random_walk_normal_fn`. `scale` might
  be a `Tensor` or a list of `Tensor`s that should broadcast with state parts
  of the `current_state`. The generated uniform perturbation is sampled as a
  uniform point on the rectangle `[-scale, scale]`.

  Args:
    scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
      controlling the upper and lower bound of the uniform proposal
      distribution.
    name: Python `str` name prefixed to Ops created by this function.
        Default value: 'random_walk_uniform_fn'.

  Returns:
    random_walk_uniform_fn: A callable accepting a Python `list` of `Tensor`s
      representing the state parts of the `current_state` and an `int`
      representing the random seed used to generate the proposal. The callable
      returns the same-type `list` of `Tensor`s as the input and represents the
      proposal for the RWM algorithm.
  """

    def _fn(state_parts, seed, experimental_shard_axis_names=None):
        """Adds a uniform perturbation to the input state.

    Args:
      state_parts: A list of `Tensor`s of any shape and real dtype representing
        the state parts of the `current_state` of the Markov chain.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.

    Returns:
      perturbed_state_parts: A Python `list` of The `Tensor`s. Has the same
        shape and type as the `state_parts`.

    Raises:
      ValueError: if `scale` does not broadcast with `state_parts`.
    """
        with tf.name_scope(name or 'random_walk_uniform_fn'):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError('`scale` must broadcast with `state_parts`.')

            part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
            part_seeds = distribute_lib.fold_in_axis_index(
                part_seeds, experimental_shard_axis_names)

            next_state_parts = [
                samplers.uniform(  # pylint: disable=g-complex-comprehension
                    minval=state_part - scale_part,
                    maxval=state_part + scale_part,
                    shape=ps.shape(state_part),
                    dtype=dtype_util.base_dtype(state_part.dtype),
                    seed=seed_part)
                for scale_part, state_part, seed_part
                in zip(scales, state_parts, part_seeds)
            ]
            return next_state_parts

    return _fn


class RandomWalkMetropolis(RandomWalkMetropolisBase):
    """Runs one step of the RWM algorithm with symmetric proposal.

  Random Walk Metropolis is a gradient-free Markov chain Monte Carlo
  (MCMC) algorithm. The algorithm involves a proposal generating step
  `proposal_state = current_state + perturb` by a random
  perturbation, followed by Metropolis-Hastings accept/reject step. For more
  details see [Section 2.1 of Roberts and Rosenthal (2004)](
  http://dx.doi.org/10.1214/154957804100000024).

  Current class implements RWM for normal and uniform proposals. Alternatively,
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
                 new_state_fn=None,
                 random_walk_cov=1.,
                 experimental_shard_axis_names=None,
                 name=None):
        super().__init__(target_log_prob_fn,
                         new_state_fn=None,
                         experimental_shard_axis_names=None,
                         name=None)
        """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      new_state_fn: Python callable which takes a list of state parts and a
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
        if new_state_fn is None:
            new_state_fn = random_walk_normal_fn

        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=UncalibratedRandomWalk(
                target_log_prob_fn=target_log_prob_fn,
                random_walk_cov=random_walk_cov,
                new_state_fn=new_state_fn,
                name=name)).experimental_with_shard_axes(
            experimental_shard_axis_names)
        self._parameters = self._impl.inner_kernel.parameters.copy()


class UncalibratedRandomWalk(RandomWalkBase):
    """Generate proposal for the Random Walk Metropolis algorithm.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `tfp.mcmc.RandomWalkMetropolisNormal(...)` or
  `tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedRandomWalk(...))`.

  For more details on `UncalibratedRandomWalk`, see
  `RandomWalkMetropolis`.
  """

    @mcmc_util.set_doc(RandomWalkMetropolis.__init__.__doc__)
    def __init__(self,
                 target_log_prob_fn,
                 new_state_fn=None,
                 random_walk_cov=1.,
                 experimental_shard_axis_names=None,
                 name=None):
        super().__init__(target_log_prob_fn,
                         new_state_fn=None,
                         experimental_shard_axis_names=None,
                         name=None)

        new_state_fn = random_walk_normal_fn

        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=new_state_fn,
            experimental_shard_axis_names=experimental_shard_axis_names,
            random_walk_cov=random_walk_cov,
            name=name)

    @property
    def random_walk_cov(self):
        return self._parameters['random_walk_cov']

    @property
    def new_state_fn(self):
        return self._parameters['new_state_fn']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @mcmc_util.set_doc(RandomWalkMetropolis.one_step.__doc__)
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

            new_state_fn = self.new_state_fn(scale=previous_kernel_results.random_walk_cov)

            next_state_parts = new_state_fn(  # pylint: disable=not-callable
                current_state_parts, seed, **state_fn_kwargs)

            # User should be using a new_state_fn that does not alter the state size.
            # This will fail noisily if that is not the case.
            for next_part, current_part in zip(next_state_parts, current_state_parts):
                tensorshape_util.set_shape(next_part, current_part.shape)

            # Compute `target_log_prob` so its available to MetropolisHastings.
            next_target_log_prob = self.target_log_prob_fn(*next_state_parts)  # pylint: disable=not-callable

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            return [
                maybe_flatten(next_state_parts),
                UncalibratedRandomWalkResults(
                    log_acceptance_correction=tf.zeros_like(next_target_log_prob),
                    target_log_prob=next_target_log_prob,
                    seed=seed,
                    random_walk_cov=previous_kernel_results.random_walk_cov,
                ),
            ]

    @mcmc_util.set_doc(RandomWalkMetropolis.bootstrap_results.__doc__)
    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(
                self.name, 'rwm', 'bootstrap_results')):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]
            init_target_log_prob = self.target_log_prob_fn(*init_state)  # pylint:disable=not-callable
            return UncalibratedRandomWalkResults(
                log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                target_log_prob=init_target_log_prob,
                # Allow room for one_step's seed.
                seed=samplers.zeros_seed(),
                random_walk_cov=self.random_walk_cov)
