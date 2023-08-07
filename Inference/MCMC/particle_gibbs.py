import tensorflow as tf
from tensorflow_probability.python.mcmc import sample_chain
from tensorflow_probability.python.mcmc.random_walk_metropolis import RandomWalkMetropolis, random_walk_normal_fn
from collections import namedtuple

"""
Particle Gibbs Sampling Algorithm 
"""

return_results = namedtuple(
    'ParticleGibbsResult', ['states', 'trace_results'])


def particle_gibbs_sampling(ssm_model,
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

    with tf.name_scope(name or 'particle_gibbs') as name:

        if not ssm_model.target_dist:
            raise NotImplementedError("No target distribution exists!")
        if not init_state:
            init_state = tf.zeros_like(ssm_model.target_dist.sample())

        mh_kernel = RandomWalkMetropolis(ssm_model.target_dist,
                                         new_state_fn=random_walk_normal_fn(scale=0.5))

        states, kernels_results = sample_chain(num_results=num_results,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=mh_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
