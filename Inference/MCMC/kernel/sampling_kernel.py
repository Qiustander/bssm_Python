"""The minimum Conditional Sampling sampling kernel"""
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
    'SamplingKernel',
    'SamplingKernelResults',
]


class SamplingKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "SamplingKernelResults",
        [
            "target_log_prob",
            "inner_results",
        ],
    ),
):
    __slots__ = ()


class SamplingKernel(mcmc.TransitionKernel):
    """Minimal Conditional Sampling Kernel
    """

    def __init__(self, target_log_prob_fn, kernel_list, name=None):
        """Build a Conditional Sampling scheme from component kernels.

        :param target_log_prob_fn: a function that takes `state` arguments
                                   and returns the target log probability
                                   density.
        :param kernel_list: a list of tuples `(state_part_idx, kernel_make_fn)`.
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
            target_log_prob_fn=target_log_prob_fn,
            kernel_list=kernel_list,
            name=name,
        )

    @property
    def is_calibrated(self):
        return False

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

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
                #TODO: check this function, this happens before the upddate, why update the state_parts -Qiuliang
                state_parts[
                    state_part_idx  # pylint: disable=cell-var-from-loop
                ] = state_part
                # Update the latest index
                return self.target_log_prob_fn(*state_parts)

            # kernel_fn is the kernel_make_fn
            kernel = kernel_fn(target_log_prob_fn, state_parts)

            # Forward the current tlp to the kernel.  If the kernel is gradient-based,
            # we need to calculate fresh gradients, as these cannot easily be forwarded
            # from the previous Gibbs step.
            if _has_gradients(previous_step_results):
                # TODO would be better to avoid re-calculating the whole of
                # `bootstrap_results` when we just need to calculate gradients.
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
            GibbsKernelResults(
                target_log_prob=untransformed_target_log_prob,
                inner_results=next_results,  # All intermediate results are recorded, is it necessary?
            ),
        )

    def bootstrap_results(self, current_state):

        return SamplingKernelResults(
            target_log_prob=untransformed_target_log_prob,
            inner_results=inner_results,
        )
