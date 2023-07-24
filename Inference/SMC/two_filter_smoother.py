import tensorflow as tf
from collections import namedtuple
from Utils.smc_utils.smc_utils import posterior_mean_var
from tensorflow_probability.python.internal import samplers

return_results = namedtuple(
    'TwoFilterSmoother', ['filtered_mean', 'predicted_mean',
                          'filtered_variance', 'predicted_variance', 'smoother_mean',
                          'incremental_log_marginal_likelihoods', 'particles',
                          'log_weights', 'parent_indices', 'accumulated_log_marginal_likelihood'])


@tf.function
def two_filter_smoother(ssm_model,
                                     observations,
                                     num_particles,
                                     particle_filter_name,
                                     resample_ess=0.5,
                                     resample_fn='systematic',
                                     seed=None,
                                     name=None):  # pylint: disable=g-doc-args
    """Generalized Two-Filter Smoother Algorithm

     Estimate the marginal posterior distribution p(x_{t} | y_{0:T}).

  ${particle_filter_arg_str}
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'infer_trajectories'`).
  Returns:

  #### References

  [1] Kitagawa, G (1996). Monte Carlo filter and smoother for non-Gaussian
     nonlinear state space models.
     Journal of Computational and Graphical Statistics, 5, 1-25.
     https://doi.org/10.2307/1390750

  """
    with tf.name_scope(name or 'forward_filter_backward_smoother') as name:
        pf_seed, resample_seed = samplers.split_seed(
            seed, salt='forward_filter_backward_smoother')

        num_time_step = tf.get_static_value(tf.shape(observations))[0]
        if particle_filter_name == 'ekf':
            from .extend_kalman_particle_filter import extended_kalman_particle_filter
            infer_result = extended_kalman_particle_filter(ssm_model=ssm_model,
                                                           observations=observations,
                                                           resample_fn='systematic',
                                                           resample_ess=resample_ess,
                                                           num_particles=num_particles,
                                                           seed=pf_seed)
        elif particle_filter_name == 'bsf':
            from .bootstrap_filter import bootstrap_particle_filter
            infer_result = bootstrap_particle_filter(ssm_model=ssm_model,
                                                     resample_fn='systematic',
                                                     observations=observations,
                                                     resample_ess=resample_ess,
                                                     num_particles=num_particles,
                                                     seed=pf_seed)
        elif particle_filter_name == 'apf':
            from .auxiliary_particle_filter import auxiliary_particle_filter
            infer_result = auxiliary_particle_filter(ssm_model=ssm_model,
                                                     resample_fn='systematic',
                                                     observations=observations,
                                                     resample_ess=resample_ess,
                                                     num_particles=num_particles,
                                                     seed=pf_seed)
        else:
            raise NotImplementedError('No particle method')

        particles = infer_result.particles
        parent_indices = infer_result.parent_indices
        log_weights = infer_result.log_weights
        incremental_log_marginal_likelihoods = infer_result.incremental_log_marginal_likelihoods

        pass