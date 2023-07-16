import tensorflow as tf
from tensorflow_probability.python.internal import samplers
from .particle_filter import particle_filter
from Utils.smc_utils.smc_utils import default_trace_fn
from Utils.smc_utils.smc_utils import posterior_mean_var
from Utils.smc_utils.smc_utils import extended_kalman_particle_step, extended_kalman_particle_initial

from collections import namedtuple

"""
Extended Kalman particle filter
"""

return_results = namedtuple(
    'EKPFResult', ['filtered_mean', 'predicted_mean',
                        'filtered_variance', 'predicted_variance',
                        'incremental_log_marginal_likelihoods', 'particles',
                        'log_weights', 'parent_indices', 'accumulated_log_marginal_likelihood'])


def extended_kalman_particle_filter(ssm_model,
                              observations,
                              num_particles,
                              initial_state_proposal=None,
                              proposal_fn=extended_kalman_particle_step,
                              resample_fn='stratified',
                              resample_ess=0.5,
                              unbiased_gradients=True,
                              num_transitions_per_observation=1,
                              seed=None,
                              name=None):
    """

    Args:
        ssm_model: state space model
        observations: observed points
        num_particles: the number of particles
        initial_state_proposal: default none, could be useful in EKPF
        proposal_fn: in bootstrap filter is None, that is, use the transiton prior
        resample_fn: 'stratified', 'systematic', 'residual', 'multinomial'
        resample_ess: 1.0 - resampling every step, 0.0 - skip the resampling
        unbiased_gradients:
        num_transitions_per_observation:
        seed:
        name:

    Returns:

    """

    with tf.name_scope(name or 'extended_kalman_particle_filter') as name:
        pf_seed, resample_seed = samplers.split_seed(
            seed)
        (particles,  # num_time_step, particle_num, state_dim
         log_weights,
         parent_indices,
         incremental_log_marginal_likelihoods,
         accumulated_log_marginal_likelihood) = particle_filter(
            observations=observations,
            initial_state_prior=ssm_model.initial_state_prior,
            transition_fn=ssm_model.transition_dist,
            observation_fn=ssm_model.observation_dist,
            filter_method='ekf',
            transition_fn_grad=ssm_model.transition_fn_grad,
            observation_fn_grad=ssm_model.observation_fn_grad,
            num_particles=num_particles,
            initial_state_proposal=initial_state_proposal,
            proposal_fn=proposal_fn,
            resample_fn=resample_fn,
            resample_ess_num=resample_ess,
            unbiased_gradients=unbiased_gradients,
            num_transitions_per_observation=num_transitions_per_observation,
            trace_fn=default_trace_fn,
            trace_criterion_fn=lambda *_: True,
            seed=pf_seed,
            name=name)

        # state_dim = ssm_model.initial_state_prior.sample().shape[0]
        filtered_mean, predicted_mean, \
            filtered_variance, predicted_variance = posterior_mean_var(particles,
                                                                       log_weights)
        return return_results(filtered_mean=filtered_mean, predicted_mean=predicted_mean,
                              filtered_variance=filtered_variance, predicted_variance=predicted_variance,
                              incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods,
                              accumulated_log_marginal_likelihood=accumulated_log_marginal_likelihood,
                              particles=particles, log_weights=log_weights, parent_indices=parent_indices)


