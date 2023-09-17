"""Gibbs sampling kernel"""
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.internal import prefer_static

tfd = tfp.distributions  # pylint: disable=no-member
tfb = tfp.bijectors  # pylint: disable=no-member
mcmc = tfp.mcmc  # pylint: disable=no-member

__all__ = [
    'ParticleGibbsKernel',
    'ParticleGibbsKernelResults',
]


class ParticleGibbsKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "GibbsKernelResults",
        [
            "inner_results",
            "step",
        ],
    ),
):
    __slots__ = ()


def _flatten_results(results):
    """Results structures from nested Gibbs samplers sometimes
    need flattening for writing out purposes.
    """

    def recurse(r):
        for i in iter(r):
            if isinstance(i, list):
                for j in _flatten_results(i):
                    yield j
            else:
                yield i

    return [r for r in recurse(results)]


def _has_gradients(results):
    return unnest.has_nested(results, "grads_target_log_prob")


def _get_target_log_prob(results):
    """Fetches a target log prob from a results structure"""
    return unnest.get_innermost(results, "target_log_prob")


def _update_target_log_prob(results, target_log_prob):
    """Puts a target log prob into a results structure"""
    if isinstance(results, GibbsKernelResults):
        replace_fn = unnest.replace_outermost
    else:
        replace_fn = unnest.replace_innermost
    return replace_fn(results, target_log_prob=target_log_prob)


def _maybe_transform_value(tlp, state, kernel, direction):
    if not isinstance(kernel, tfp.mcmc.TransformedTransitionKernel):
        return tlp

    tlp_rank = prefer_static.rank(tlp)
    event_ndims = prefer_static.rank(state) - tlp_rank

    if direction == "forward":
        return tlp + kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    if direction == "inverse":
        return tlp - kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    raise AttributeError("`direction` must be `forward` or `inverse`")


class ParticleGibbsKernel(mcmc.TransitionKernel):
    """Particle Gibbs For State Space Model
    Gibbs sampling may be useful when the joint distribution is explicitly unknown
    or difficult to sample from directly, but the conditional distribution for each
    variable is known and can be sampled directly. The Gibbs sampling algorithm
    generates a realisation from each variable's conditional distribution in turn,
    conditional on the current realisations of the other variables.
    The resulting sequence of samples forms a Markov chain whose stationary
    distribution represents the joint distribution.

    """

    def __init__(self, kernel_list, name=None):
        """Build a Gibbs sampling scheme from component kernels.

        :param target_log_prob_fn: a function that takes `state` arguments
                                   and returns the target log probability
                                   density.
        :param kernel_list: a list of tuples `(state_part_idx, kernel``_make_fn)`.
                            `state_part_idx` denotes the index (relative to
                            positional args in `target_log_prob_fn`) of the
                            state the kernel updates.  `kernel_make_fn` takes
                            arguments `target_log_prob_fn` and `state`, returning
                            a `tfp.mcmc.TransitionKernel`.
                            The length of the kernel list should be the same as the number of parameters
        :returns: an instance of `GibbsKernel`
        """
        # Require to check if all kernel.is_calibrated is True
        self._parameters = dict(
            kernel_list=kernel_list,
            name=name,
        )

    @property
    def is_calibrated(self):
        return True

    @property
    def kernel_list(self):
        return self._parameters["kernel_list"]

    @property
    def name(self):
        return self._parameters["name"]

    def one_step(self, current_state, previous_results, seed=None):
        """We iterate over the state elements, calling each kernel in turn.

        The `target_log_prob` is forwarded to the next `previous_results`
        such that each kernel has a current `target_log_prob` value.
        Transformations are automatically performed if the kernel is of
        type tfp.mcmc.TransformedTransitionKernel.

        In graph and XLA modes, the for loop should be unrolled.
        """
        if mcmc_util.is_list_like(current_state):
            state_parts = list(current_state)
        else:
            state_parts = [current_state]
            # A list, containing all states in Tensor

        state_parts = [
            tf.convert_to_tensor(s, name="current_state") for s in state_parts
        ]

        next_results = []
        untransformed_target_log_prob = previous_results.target_log_prob

        for (state_part_idx, kernel_fn), previous_step_results in zip(
                self.kernel_list, previous_results.inner_results
        ):

            def target_log_prob_fn(state_part):
                # This function is defined before the update, so need extra step to update the state_parts
                state_parts[
                    state_part_idx  # pylint: disable=cell-var-from-loop
                ] = state_part

                # Update the latest index
                return self.target_log_prob_fn(*state_parts)

            # kernel_fn is the kernel_make_fn
            # The target_log_prob is computed after the update of the parameter
            kernel = kernel_fn(target_log_prob_fn, state_parts, state_part_idx)

            # Forward the current tlp to the kernel.  If the kernel is gradient-based,
            # we need to calculate fresh gradients, as these cannot easily be forwarded
            # from the previous Gibbs step.
            if _has_gradients(previous_step_results):

                fresh_previous_results = unnest.UnnestingWrapper(
                    kernel.bootstrap_results(state_parts[state_part_idx])
                )
                previous_step_results = unnest.replace_innermost(
                    previous_step_results,
                    target_log_prob=fresh_previous_results.target_log_prob,
                    grads_target_log_prob=fresh_previous_results.grads_target_log_prob,
                )

            else:
                previous_step_results = _update_target_log_prob(
                    previous_step_results,
                    _maybe_transform_value(
                        tlp=untransformed_target_log_prob,
                        state=state_parts[state_part_idx],
                        kernel=kernel,
                        direction="inverse",
                    ),
                )
            # For full conditional sampling, we need to pass the full states
            state_parts[state_part_idx], next_kernel_results = kernel.one_step(
                state_parts[state_part_idx], previous_step_results, seed
            )

            next_results.append(next_kernel_results)

            # Cache the new tlp for use in the next Gibbs step
            untransformed_target_log_prob = _maybe_transform_value(
                tlp=_get_target_log_prob(next_kernel_results),
                state=state_parts[state_part_idx],
                kernel=kernel,
                direction="forward",
            )
        return (
            state_parts
            if mcmc_util.is_list_like(current_state)
            else state_parts[0],
            ParticleGibbsKernelResults(
                inner_results=next_results,  # All intermediate results are recorded, is it necessary? - Qiuliang
                step=1 + previous_results.step,
            ),
        )

    def bootstrap_results(self, current_state):

        if mcmc_util.is_list_like(current_state):
            state_parts = list(current_state)
        else:
            state_parts = [tf.convert_to_tensor(current_state)]
        state_parts = [
            tf.convert_to_tensor(s, name="current_state") for s in state_parts
        ]

        inner_results = []
        untransformed_target_log_prob = 0.0
        for state_part_idx, kernel_fn in self.kernel_list:
            def target_log_prob_fn(state_part):
                state_parts[
                    state_part_idx  # pylint: disable=cell-var-from-loop
                ] = state_part
                return self.target_log_prob_fn(*state_parts)

            kernel = kernel_fn(target_log_prob_fn, current_state, state_part_idx)
            kernel_results = kernel.bootstrap_results(
                state_parts[state_part_idx]
            )
            inner_results.append(kernel_results)
            untransformed_target_log_prob = _maybe_transform_value(
                tlp=_get_target_log_prob(kernel_results),
                state=state_parts[state_part_idx],
                kernel=kernel,
                direction="forward",
            )

        return ParticleGibbsKernelResults(
            target_log_prob=untransformed_target_log_prob,
            inner_results=inner_results,
            step=tf.constant(0, dtype=tf.int32),
        )
