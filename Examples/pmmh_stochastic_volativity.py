import types
from Synthetic_Data.stochastic_volatility import gen_ssm
import matplotlib.pyplot as plt
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import arviz as az
from tensorflow_probability.python.internal import prefer_static as ps
from Inference.MCMC.kernel.random_walk_metropolis import RandomWalkMetropolis

tfd = tfp.distributions
tfb = tfp.bijectors

"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""
import logging

logging.basicConfig(level=logging.INFO)

VALIDATE_ARGS = True
num_timesteps = 100
num_particles = 1000
num_samples = 10000
particle_method = 'bsf'
state_dim = 1
observation_dim = 1
import time

"""
Generate data 
"""
true_sigma = tf.constant([0.178])
true_mu = tf.constant([-1.02])
true_rho = tf.constant([0.97])
ssm_model = gen_ssm(num_timesteps=num_timesteps,
                    state_dim=state_dim,
                    observed_dim=observation_dim,
                    mu=true_mu,
                    rho=true_rho,
                    state_mtx_noise=true_sigma)


true_state, observations = ssm_model.simulate()

# prior specification
PRIOR_MEAN_MU = tf.constant([0.])
PRIOR_STD_MU = tf.constant([[2.0]])
PRIOR_INV_GAMMA_ALPHA = tf.constant([1.])
PRIOR_INV_GAMMA_BETA = tf.constant([1.])
PRIOR_BETA_A = tf.constant([9.0])
PRIOR_BETA_B = tf.constant([1.0])
N_CHAINS = 10

print(f'sample state mean: {tf.reduce_mean(true_state)}')
print(f'sample state std: {tf.math.reduce_std(true_state)}')
print(f'sample observation mean: {tf.math.reduce_mean(observations)}')
print(f'sample observation std: {tf.math.reduce_std(observations)}')
replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])


class NewClassInstance:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(NewClassInstance, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))


def initial_theta():
    """
    Returns: initial precision matrix estimation
    """
    return [tf.random.uniform([N_CHAINS, 1]) + 0.5,  # sigma
            tf.random.normal([N_CHAINS, 1]),  # mu
            tf.random.uniform([N_CHAINS, 1]) - 0.2  # rho
            ]


def log_theta_prior(sigma, mu, rho):
    return tfd.JointDistributionNamed(dict(
        sigma=tfd.Gamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                        rate=PRIOR_INV_GAMMA_BETA),
        mu=tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_MU, scale_tril=PRIOR_STD_MU),
        # rho=tfd.Uniform(low=-1., high=1.)
        rho=tfd.Beta(concentration1=PRIOR_BETA_A, concentration0=PRIOR_BETA_B)
    )
    )


class StochasticVolativitySSM(NewClassInstance):
    """Define the log_prior
    """

    """Define update function for ssm model
    """

    def update_model(self, sigma, mu, rho, observations):
        sigma = sigma[..., tf.newaxis]

        prior_mean = mu
        prior_cov = sigma / tf.sqrt(1 - rho[..., tf.newaxis] ** 2)

        initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=prior_cov)

        transition_fn = lambda t, x: tf.cast(
            mu * (1. - rho) + rho * x, dtype=sigma.dtype)
        transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=transition_fn(t, x),
            scale_tril=sigma)

        return [initial_state_prior, transition_dist, self._observation_dist]

    """Define data likelihood for ssm model
    """


def _smc_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.accumulated_log_marginal_likelihood)


def _run_smc(initial_state_prior, transition_dist, observation_dist):
    result = particle_filter(
        observations=observations,
        initial_state_prior=initial_state_prior,
        transition_fn=transition_dist,
        observation_fn=observation_dist,
        num_particles=num_particles,
        trace_fn=_smc_trace_fn,
        trace_criterion_fn=lambda *_: True)
    return result


def log_target_dist(ssm_models, observations, num_particles):
    def _log_likelihood(sigma, mu, rho):
        # update model
        new_update_fn = ssm_models.update_model(sigma, mu, rho, observations)

        traced_results = _run_smc(initial_state_prior=new_update_fn[0],
                                  transition_dist=new_update_fn[1],
                                  observation_dist=new_update_fn[2])
        return traced_results[-1][-1] \
            + log_theta_prior(sigma, mu, rho).log_prob({'sigma': sigma[:, 0],
                                                        'mu': mu,
                                                        'rho': (rho[:, 0] +1)/2 })

    return _log_likelihood


sv_ssm = StochasticVolativitySSM(ssm_model)


@tf.function(jit_compile=True)
def run_mcmc():
    unconstrained_to_precision = tfb.JointMap(
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()]
    )

    result = particle_marginal_metropolis_hastings(num_samples,
                                                   target_dist=log_target_dist(sv_ssm, replicate_observations,
                                                                               num_particles),
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=initial_theta(),
                                                   num_burnin_steps=int(num_samples),
                                                   num_steps_between_results=0,
                                                   seed=None,
                                                   name=None)
    return result


@tf.function
def run_smc():
    result = bootstrap_particle_filter(ssm_model,
                                       observations,
                                       num_particles)
    return result


if __name__ == '__main__':
    start = time.time()
    mcmc_result = run_mcmc()
    # smc_result = run_smc()
    end = time.time()
    print(f" execute time {end - start}")
    mcmc_trace = mcmc_result.trace_results
    mcmc_state = mcmc_result.states

    parameter_names = ['noise_std', 'mean', 'coefficients']
    posterior_trace = az.from_dict(
        posterior={
            k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_state)
        },
        # sample_stats={k: np.swapaxes(v, 0, 1) for k, v in mcmc_result.items()},
        observed_data={"observations": observations},
        coords={"coefficient": np.arange(1)},
        dims={"intercept": ["coefficient"]},
    )
    print(az.summary(posterior_trace))
    az.plot_trace(posterior_trace)
    az.plot_rank(posterior_trace)
    plt.show()
