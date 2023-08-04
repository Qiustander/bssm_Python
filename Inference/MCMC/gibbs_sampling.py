import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc import sample_chain
from collections import namedtuple

"""
Gibbs Sampling Algorithm
"""

return_results = namedtuple(
    'GibbsSamplingResults', ['states', 'trace_results'])


def gibbs_sampling(ssm_model,
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

    with tf.name_scope(name or 'gibbs_sampling') as name:

        if not ssm_model.target_dist:
            raise NotImplementedError("No target distribution exists!")
        if not init_state:
            init_state = tf.zeros_like(ssm_model.target_dist.sample())
        mh_kernel = tfp.mcmc.tfp.mcmc.MetropolisHastings(ssm_model.proposal_dist)

        states, kernels_results = sample_chain(num_results=num_results,
                                               current_state=init_state,
                                               num_burnin_steps=num_burnin_steps,
                                               num_steps_between_results=num_steps_between_results,
                                               kernel=mh_kernel,
                                               seed=seed)

        return return_results(states=states, trace_results=kernels_results)
