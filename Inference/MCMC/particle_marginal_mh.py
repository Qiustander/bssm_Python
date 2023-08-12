import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain, TransformedTransitionKernel
from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn, RandomWalkMetropolis
from tensorflow_probability.python.mcmc.simple_step_size_adaptation import SimpleStepSizeAdaptation
from collections import namedtuple
from Inference.MCMC.kernel.particle_mcmc_kernel import ParticleMetropolisHastings
from tensorflow_probability.python.internal import samplers

"""
(Particle) Marginal Metropolis Hasting Algorithm with Random Walk Proposal Distribution
"""

return_results = namedtuple(
    'PMMHResult', ['states', 'trace_results'])


def particle_marginal_metropolis_hastings(ssm_model,
                                          observations,
                                          num_samples,
                                          num_particles,
                                          num_burnin_steps,
                                          init_state,
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

        mh_kernel = RandomWalkMetropolis(ssm_model.log_target_dist(observations, num_particles),
                                               new_state_fn=random_walk_normal_fn(
                                                   scale=0.3))

        if transformed_bijector is None:
            mh_kernel = mh_kernel
        else:
            mh_kernel = TransformedTransitionKernel(
                                inner_kernel=mh_kernel,
                                bijector=transformed_bijector)

        states, kernels_results = sample_chain(num_results=num_samples,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=mh_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
