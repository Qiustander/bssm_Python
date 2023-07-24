import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from collections import namedtuple
from .infer_trajectories import reconstruct_trajectories
from .particle_filter import _check_resample_fn
from tensorflow_probability.python.internal import samplers

return_results = namedtuple(
    'FilterSmoother', ['filtered_mean', 'predicted_mean',
                   'filtered_variance', 'predicted_variance', 'smoother_mean',
                   'incremental_log_marginal_likelihoods',  'particles',
                   'log_weights', 'parent_indices', 'accumulated_log_marginal_likelihood'])


def filter_smoother(ssm_model,
                    observations,
                    num_particles,
                    particle_filter_name,
                    resample_ess=0.5,
                    resample_fn='stratified',
                    seed=None,
                    name=None):  # pylint: disable=g-doc-args
    """Filter smoother algorithm, a simple path retracting smoothing algorithmby
     Kitagawa (1996).

     Estimate the full-path posterior distribution p(x_{0:T} | y_{0:T}) with simple
     tracing.

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
    with tf.name_scope(name or 'filter_smoother') as name:
        pf_seed, resample_seed = samplers.split_seed(
            seed, salt='filter_smoother')
        try:
            if particle_filter_name == 'ekf':
                from .extend_kalman_particle_filter import extended_kalman_particle_filter
                infer_result = extended_kalman_particle_filter(ssm_model=ssm_model,
                                                               observations=observations,
                                                               resample_fn=resample_fn,
                                                               resample_ess=resample_ess,
                                                               num_particles=num_particles,
                                                               seed=pf_seed)
            elif particle_filter_name == 'bsf':
                from .bootstrap_filter import bootstrap_particle_filter
                infer_result = bootstrap_particle_filter(ssm_model=ssm_model,
                                                         resample_fn=resample_fn,
                                                         observations=observations,
                                                         resample_ess=resample_ess,
                                                         num_particles=num_particles,
                                                         seed=pf_seed)
            elif particle_filter_name == 'apf':
                from .auxiliary_particle_filter import auxiliary_particle_filter
                infer_result = auxiliary_particle_filter(ssm_model=ssm_model,
                                                         resample_fn=resample_fn,
                                                         observations=observations,
                                                         resample_ess=resample_ess,
                                                         num_particles=num_particles,
                                                         seed=pf_seed)
        except:
            raise NotImplementedError('No particle method')

        particles = infer_result.particles
        parent_indices = infer_result.parent_indices
        log_weights = infer_result.log_weights
        incremental_log_marginal_likelihoods = infer_result.incremental_log_marginal_likelihoods
        resample_fn = _check_resample_fn(resample_fn)

        weighted_trajectories = reconstruct_trajectories(particles, parent_indices)

        # Resample all steps of the trajectories using the final weights.
        resample_indices = resample_fn(weights=log_weights[-1],
                                       resample_num=num_particles,
                                       seed=resample_seed)
        trajectories = tf.nest.map_structure(
            lambda x: mcmc_util.index_remapping_gather(x,  # pylint: disable=g-long-lambda
                                                        resample_indices,
                                                       axis=1),
            weighted_trajectories)
        smoother_mean = tf.reduce_mean(trajectories, axis=1)

        return return_results(filtered_mean=infer_result.filtered_mean, predicted_mean=infer_result.predicted_mean, smoother_mean=smoother_mean,
                              filtered_variance=infer_result.filtered_variance, predicted_variance=infer_result.predicted_variance,
                              incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods,
                              accumulated_log_marginal_likelihood=infer_result.accumulated_log_marginal_likelihood,
                              particles=particles, log_weights=log_weights, parent_indices=parent_indices)
