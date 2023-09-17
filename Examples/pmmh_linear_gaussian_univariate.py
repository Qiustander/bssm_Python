import copy
from tensorflow_probability.python.mcmc import effective_sample_size, potential_scale_reduction

from Synthetic_Data.linear_gaussian import gen_data
import matplotlib.pyplot as plt
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import arviz as az
import time

tfd = tfp.distributions
tfb = tfp.bijectors
import logging

logging.basicConfig(level=logging.INFO)
"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""
import logging

logging.basicConfig(level=logging.INFO)

VALIDATE_ARGS = True
num_timesteps = 100
num_particles = 600
num_samples = 4000
particle_method = 'bsf'
state_dim = 1
observation_dim = 1
testcase = 'univariate'
N_CHAINS = 8

"""
Generate data 
"""
ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                     state_dim=state_dim, observed_dim=observation_dim)

true_state, observations = ssm_model.simulate()
PRIOR_INV_GAMMA_ALPHA = tf.constant([1.0])
PRIOR_INV_GAMMA_BETA = tf.constant([1.0])
replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

class NewClassInstance:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(NewClassInstance, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))


class LinearGaussianSSM(NewClassInstance):

        # Currently, we could not trace the SMC results in the log_likelihood because of the wrapped transformed transition kernel.

    """Define the log_prior
    """

    def initial_theta(self):
        """
        Returns: initial precision matrix estimation
        """
        return [tf.random.uniform([N_CHAINS, 1]) + 0.5,
                tf.random.uniform([N_CHAINS, 1]) + 2.0]

    def log_theta_prior(self, sigma_x, sigma_y):
        sigma_x_prior = tfd.Gamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                         rate=tf.constant(PRIOR_INV_GAMMA_BETA))
        sigma_y_prior = tfd.Gamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                         rate=tf.constant(PRIOR_INV_GAMMA_BETA))

        return sigma_x_prior.log_prob(sigma_x) + sigma_y_prior.log_prob(sigma_y)

    """Define update function for ssm model
    """

    def update_model(self, sigma_x, sigma_y):
        new_class = LinearGaussianSSM(self)

        new_class._transition_noise_matrix = tf.sqrt(sigma_x[..., tf.newaxis])
        new_class._observation_noise_matrix = tf.sqrt(sigma_y[..., tf.newaxis])

        prior_mean = tf.broadcast_to(new_class._initial_state_prior.mean(), sigma_x.shape)
        prior_std = tf.broadcast_to(new_class._initial_state_prior.stddev(), sigma_x.shape)[..., tf.newaxis]

        new_class._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=prior_std)

        # self._transition_noise_matrix = tf.linalg.cholesky(covariances_x)
        # self._observation_noise_matrix = tf.linalg.cholesky(covariances_y)

        new_class._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=new_class.transition_fn(t, x),
            scale_tril=new_class._transition_noise_matrix)
        new_class._observation_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=new_class.observation_fn(t, x),
            scale_tril=new_class._observation_noise_matrix
        )

        return new_class


linear_gaussian_ssm = LinearGaussianSSM(ssm_model)

"""Define data likelihood for ssm model
"""


def log_target_dist(ssmmodel, observation, num_particle):
    def _log_likelihood(sigma_x, sigma_y):
        # update model
        new_class = ssmmodel.update_model(sigma_x=sigma_x, sigma_y=sigma_y)

        def _smc_trace_fn(state, kernel_results):
            return (state.particles,
                    state.log_weights,
                    kernel_results.accumulated_log_marginal_likelihood)

        # Conduct SMC
        def _run_smc(ssm_class):

            result = particle_filter(
                observations=observation,
                initial_state_prior=ssm_class.initial_state_prior,
                transition_fn=ssm_class.transition_dist,
                observation_fn=ssm_class.observation_dist,
                num_particles=num_particle,
                trace_fn=_smc_trace_fn,
                trace_criterion_fn=lambda *_: True)
            return result

        # start = time.time()
        traced_results = _run_smc(new_class)
        # end = time.time()
        # logging.info(f"{end - start}")
        # self.smc_trace_results.append(traced_results)
        # logging.info(f"likelihood {traced_results[-1][-1]}")

        return traced_results[-1][-1] \
            + tf.squeeze(new_class.log_theta_prior(sigma_x, sigma_y))

    return _log_likelihood


@tf.function(jit_compile=True)
def run_mcmc():
    unconstrained_to_precision = tfb.JointMap(
        bijectors=[tfb.Exp(), tfb.Exp()]
    )

    result = particle_marginal_metropolis_hastings(
                                                   num_samples,
                                                   target_dist=log_target_dist(linear_gaussian_ssm,
                                                                               replicate_observations,
                                                                               num_particles),
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=linear_gaussian_ssm.initial_theta(),
                                                   num_burnin_steps=int(num_samples // 2),
                                                   num_steps_between_results=2,
                                                   seed=None,
                                                   name=None)
    return result


@tf.function
def run_smc():
    result = bootstrap_particle_filter(linear_gaussian_ssm,
                                       observations,
                                       num_particles)
    return result


if __name__ == '__main__':
    mcmc_result = run_mcmc()
    # smc_result = run_smc()

    mcmc_trace = mcmc_result.trace_results
    mcmc_state = tf.sqrt(mcmc_result.states)


    parameter_names = ['noise_x', 'noise_y']
    posterior_trace = az.from_dict(
        posterior={
            k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_state)
        },
        # sample_stats={k: np.swapaxes(v, 0, 1) for k, v in mcmc_result.items()},
        observed_data={"observations": observations},
        coords={"coefficient": np.arange(1)},
        dims={"intercept": ["coefficient"]},
    )

    xxx = 1
