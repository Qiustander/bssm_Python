import tensorflow as tf
from tensorflow_probability.python.internal import samplers
from .particle_filter import particle_filter
from Utils.smc_utils.smc_utils import default_trace_fn
from Utils.smc_utils.smc_utils import posterior_mean_var
from collections import namedtuple

"""
Auxiliary particle filter
"""

return_results = namedtuple(
    'APFResult', ['filtered_mean', 'predicted_mean',
                  'filtered_variance', 'predicted_variance',
                  'incremental_log_marginal_likelihoods', 'particles',
                  'log_weights', 'parent_indices', 'accumulated_log_marginal_likelihood'])


def auxiliary_particle_filter(ssm_model,
                              observations,
                              num_particles,
                              is_guided=False,
                              initial_state_proposal=None,
                              resample_fn='stratified',
                              resample_ess=0.5,
                              unbiased_gradients=True,
                              conditional_sample=None,
                              is_conditional=False,
                              num_transitions_per_observation=1,
                              trace_fn=default_trace_fn,
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

    with tf.name_scope(name or 'auxiliary_particle_filter'):
        if seed is None:
            seed = samplers.sanitize_seed(seed, name='auxiliary_particle_filter')

        if ssm_model.auxiliary_fn() is None and not is_guided:
            raise NotImplementedError('No Auxiliary Function!')

        (particles,  # num_time_step, particle_num, state_dim
         log_weights,
         parent_indices,
         incremental_log_marginal_likelihoods,
         accumulated_log_marginal_likelihood) = particle_filter(
            observations=observations,
            initial_state_prior=ssm_model.initial_state_prior,
            auxiliary_fn=ssm_model.auxiliary_fn if not is_guided else None,
            transition_fn=ssm_model.transition_dist,
            observation_fn=ssm_model.observation_dist,
            num_particles=num_particles,
            initial_state_proposal=initial_state_proposal,
            proposal_fn=ssm_model.proposal_dist,
            resample_fn=resample_fn,
            resample_ess_num=resample_ess,
            unbiased_gradients=unbiased_gradients,
            conditional_sample=conditional_sample,
            is_conditional=is_conditional,
            num_transitions_per_observation=num_transitions_per_observation,
            trace_fn=trace_fn,
            trace_criterion_fn=lambda *_: True,
            seed=seed,
            name=name)

        filtered_mean, predicted_mean, \
            filtered_variance, predicted_variance = posterior_mean_var(particles,
                                                                       log_weights,
                                                                       tf.get_static_value(tf.shape(observations))[0])

        return return_results(filtered_mean=filtered_mean, predicted_mean=predicted_mean,
                              filtered_variance=filtered_variance, predicted_variance=predicted_variance,
                              incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods,
                              accumulated_log_marginal_likelihood=accumulated_log_marginal_likelihood,
                              particles=particles, log_weights=log_weights, parent_indices=parent_indices)
