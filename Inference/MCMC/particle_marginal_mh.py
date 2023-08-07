import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain
from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn
from collections import namedtuple
from Inference.MCMC.kernel.particle_mcmc_kernel import ParticleMetropolisHastings
from tensorflow_probability.python.internal import samplers

"""
(Particle) Marginal Metropolis Hasting Algorithm with Random Walk Proposal Distribution
"""

return_results = namedtuple(
    'PMMHResult', ['states', 'trace_results'])

#TODO: how to assign the initial state more intelligently?
def particle_marginal_metropolis_hastings(ssm_model,
                                          num_samples,
                                          num_particles,
                                          num_burnin_steps,
                                          init_state,
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

        mh_kernel = ParticleMetropolisHastings(ssm_model.log_target_dist,
                                               proposal_theta_fn=random_walk_normal_fn(scale=1.),
                                               num_particles=num_particles)

        states, kernels_results = sample_chain(num_results=num_samples,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=mh_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
