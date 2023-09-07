import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain, TransformedTransitionKernel
# from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn, RandomWalkMetropolis
from tensorflow_probability.python.mcmc import HamiltonianMonteCarlo
from Inference.MCMC.kernel.random_walk_metropolis import RandomWalkMetropolis
from Inference.MCMC.kernel.covariance_adaptation import CovarianceAdaptation
from collections import namedtuple
from tensorflow_probability.python.internal import samplers
import tensorflow_probability.python.internal.prefer_static as ps
import tensorflow_probability as tfp

"""
(Particle) Marginal Metropolis Hasting Algorithm with Random Walk Proposal Distribution
"""

return_results = namedtuple(
    'PMMHResult', ['states', 'trace_results', 'smc_trace_results'])


def particle_marginal_metropolis_hastings(ssm_model,
                                          observations,
                                          num_samples,
                                          num_particles,
                                          num_burnin_steps,
                                          init_state,
                                          target_dist,
                                          transformed_bijector,
                                          num_steps_between_results=0,
                                          seed=None,
                                          name=None):
    """

    Args:
        init_state:
        num_results:
        num_burnin_steps:
        trace_fn:
        ssm_model: state space model
        seed:
        name:

    Returns:

    """

    with tf.name_scope(name or 'particle_metropolis_hastings'):

        if seed is None:
            seed = samplers.sanitize_seed(seed, name='particle_metropolis_hastings')

        if init_state is None:
            raise NotImplementedError("Must initialize the theta!")

        mcmc_kernel = RandomWalkMetropolis(target_dist(observations, num_particles),
                                         random_walk_cov=0.1)
        # mcmc_kernel = HamiltonianMonteCarlo(target_dist(observations, num_particles),
        #                                     step_size=0.01,
        #                                     num_leapfrog_steps=3)

        if transformed_bijector is None:
            transform_kernel = mcmc_kernel
        else:
            transform_kernel = TransformedTransitionKernel(
                inner_kernel=mcmc_kernel,
                bijector=transformed_bijector)

        adapted_kernel = CovarianceAdaptation(
            inner_kernel=transform_kernel,
            num_adaptation_steps=int(0.8* num_burnin_steps),
            target_accept_prob=0.234,
            adaptation_rate=2 / 3)

        # adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     inner_kernel=transform_kernel,
        #     num_adaptation_steps=int(0.8 * num_burnin_steps),
        #     target_accept_prob=0.651)

        states, kernels_results = sample_chain(num_results=num_samples,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=adapted_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results,
                              smc_trace_results=ssm_model.smc_trace_results)
