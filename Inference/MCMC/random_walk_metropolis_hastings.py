import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain
from tensorflow_probability.python.mcmc.random_walk_metropolis import RandomWalkMetropolis, random_walk_normal_fn
from tensorflow_probability.python.internal import samplers
from collections import namedtuple

"""
Metropolis Hasting Algorithm with Random Walk Proposal Distribution
"""

return_results = namedtuple(
    'RandomWalkMetropolisHastingsResult', ['states', 'trace_results'])


def random_walk_metropolis_hastings(ssm_model,
                                    num_results,
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

    with tf.name_scope(name or 'metropolis_hastings'):
        if seed is None:
            seed = samplers.sanitize_seed(seed, name='particle_metropolis_hastings')

        # if init_state is None:
        #     init_state = tf.zeros_like(ssm_model.target_dist.sample())
        mh_kernel = RandomWalkMetropolis(ssm_model.target_dist,
                                         new_state_fn=random_walk_normal_fn(scale=0.5))

        states, kernels_results = sample_chain(num_results=num_results,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=mh_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
