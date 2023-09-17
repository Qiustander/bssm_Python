import tensorflow as tf
from collections import namedtuple
from tensorflow_probability.python.internal import unnest
from Inference.SMC.particle_filter import particle_filter
from Inference.SMC.filter_smoother import filter_smoother
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import prefer_static as ps


def _accept_getter_fn(kernel_results):
    """Getter for `random walk state function` so it can be inspected."""
    return unnest.get_innermost(kernel_results, 'is_accepted')


def importance_sampling_correction(mcmc_states,
                                   mcmc_trace,
                                   ssm_model,
                                   observations,
                                   particles_num,
                                   particle_filter_name='bsf',
                                   is_jump_chain=False,
                                   seed=None):
    """Importance sampling correction for MCMC with approximate likelihood
    Args:
        is_jump_chain:
        ssm_model:
        mcmc_states:
        mcmc_trace:
    Returns:

    """
    if mcmc_util.is_list_like(mcmc_states):
        mcmc_states = [mcmc_states[i].numpy() for i in range(len(mcmc_states))]
    else:
        mcmc_states = [mcmc_states.numpy()]
        # A list, containing all states in Tensor

    if ps.rank(mcmc_states[0]) > 2:
        # combine chain
        mcmc_states = [mcmc_states[i].mean(
            axis=tuple(range(1, ps.rank(mcmc_states[i]) - 2))) for i in range(len(mcmc_states))]

    # jump chain
    if is_jump_chain:
        accept_results = _get_accepted(mcmc_trace)
        num_mcmc_samples = tf.math.count_nonzero(accept_results, axis=1)
    else:
        num_mcmc_samples = ps.size0(mcmc_states[0])

    log_likelihood = tf.zeros(dtype=mcmc_states[0].dtype)
    backward_fn = _is_correction_step(ssm_model=ssm_model,
                                      particle_nums=particles_num,
                                      seed=seed,
                                      particle_filter_name=particle_filter_name,
                                      observations=observations)
    traced_results = tf.scan(backward_fn,
                               elems=mcmc_states,
                               initializer=(log_likelihood))

    return traced_results


def _is_correction_step(ssm_model, particle_nums,
                        particle_filter_name,
                        observations,
                        seed=None):
    with tf.name_scope('_is_correction_step'):
        def _one_step(stored_corrected_weights, mcmc_parameters):
            # trajectory_num, (chain)
            log_likelihood = stored_corrected_weights

            current_parameters = mcmc_parameters

            new_ssm = ssm_model.update_model(current_parameters)

            smoother_results = filter_smoother(ssm_model=new_ssm,
                                               num_particles=particle_nums,
                                               particle_filter_name=particle_filter_name,
                                               seed=seed,
                                               observations=observations)

            return (smoother_results.accumulated_log_marginal_likelihood[-1],
                    )

    return _one_step


def _get_accepted(results):
    return unnest.get_innermost(results, "is_accepted")