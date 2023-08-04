from . import smc_kernel
import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from .particle_filter import _check_resample_fn
from tensorflow_probability.python.internal import samplers
from collections import namedtuple

return_results = namedtuple(
    'InferTrajectories', ['trajectories', 'incremental_log_marginal_likelihoods',
                   ])


def infer_trajectories(ssm_model,
                       observations,
                       num_particles,
                       particle_filter_name,
                       resample_ess=0.5,
                       resample_fn='systematic',
                       seed=None,
                       name=None):  # pylint: disable=g-doc-args
    """Use particle filtering to sample from the posterior over trajectories.

  ${particle_filter_arg_str}
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'infer_trajectories'`).
  Returns:
    trajectories: a (structure of) Tensor(s) matching the latent state, each
      of shape
      `concat([[num_timesteps, num_particles, b1, ..., bN], event_shape])`,
      representing unbiased samples from the posterior distribution
      `p(latent_states | observations)`.
    incremental_log_marginal_likelihoods: float `Tensor` of shape
      `[num_observation_steps, b1, ..., bN]`,
      giving the natural logarithm of an unbiased estimate of
      `p(observations[t] | observations[:t])` at each timestep `t`. Note that
      (by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true
      `log p(observations[t] | observations[:t])`.

  #### References

  [1] Arnaud Doucet and Adam M. Johansen. A tutorial on particle
      filtering and smoothing: Fifteen years later.
      _Handbook of nonlinear filtering_, 12(656-704), 2009.
      https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
  [2] Adam Scibior, Vaden Masrani, and Frank Wood. Differentiable Particle
      Filtering without Modifying the Forward Pass. _arXiv preprint
      arXiv:2106.10314_, 2021. https://arxiv.org/abs/2106.10314

  """
    with tf.name_scope(name or 'infer_trajectories') as name:
        pf_seed, resample_seed = samplers.split_seed(
            seed, salt='infer_trajectories')
        try:
            if particle_filter_name == 'ekf':
                from .extend_kalman_particle_filter import extended_kalman_particle_filter
                infer_result = extended_kalman_particle_filter(ssm_model=ssm_model,
                                                               observations=observations,
                                                               resample_fn='systematic',
                                                               resample_ess=resample_ess,
                                                               num_particles=num_particles)
            elif particle_filter_name == 'bsf':
                from .bootstrap_particle_filter import bootstrap_particle_filter
                infer_result = bootstrap_particle_filter(ssm_model=ssm_model,
                                                         resample_fn='systematic',
                                                         observations=observations,
                                                         resample_ess=resample_ess,
                                                         num_particles=num_particles)
            elif particle_filter_name == 'apf':
                from .auxiliary_particle_filter import auxiliary_particle_filter
                infer_result = auxiliary_particle_filter(ssm_model=ssm_model,
                                                         resample_fn='systematic',
                                                         observations=observations,
                                                         resample_ess=resample_ess,
                                                         num_particles=num_particles)
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

        return return_results(trajectories=trajectories, incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods)


def reconstruct_trajectories(particles, parent_indices, name=None):
    """Reconstructs the ancestor trajectory that generated each final particle."""
    with tf.name_scope(name or 'reconstruct_trajectories'):
        # Walk backwards to compute the ancestor of each final particle at time t.
        final_indices = smc_kernel._dummy_indices_like(parent_indices[-1])  # pylint: disable=protected-access
        ancestor_indices = tf.scan(
            fn=lambda ancestor, parent: mcmc_util.index_remapping_gather(  # pylint: disable=g-long-lambda
                parent, ancestor, axis=0),
            elems=parent_indices[1:],
            initializer=final_indices,
            reverse=True)
        ancestor_indices = tf.concat([ancestor_indices, [final_indices]], axis=0)

    return tf.nest.map_structure(
        lambda part: mcmc_util.index_remapping_gather(  # pylint: disable=g-long-lambda
            part, ancestor_indices, axis=1, indices_axis=1),
        particles)


