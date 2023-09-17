import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain
from collections import namedtuple
from tensorflow_probability.python.internal import samplers

"""
Particle Gibbs Sampling Algorithm 
"""

return_results = namedtuple(
    'ParticleGibbsResult', ['states', 'trace_results'])


def particle_gibbs_sampling(gibbs_kernel,
                            num_results,
                            num_burnin_steps,
                            num_steps_between_results=0,
                            init_state=None,
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

    with tf.name_scope(name or 'particle_gibbs'):

        if seed is None:
            seed = samplers.sanitize_seed(seed, name='particle_metropolis_hastings')

        if init_state is None:
            raise NotImplementedError("Must initialize the theta!")

        states, kernels_results = sample_chain(num_results=num_results,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=gibbs_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
