import types
from Synthetic_Data.stochastic_volatility import gen_ssm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import arviz as az
from tensorflow_probability.python.internal import prefer_static as ps

tfd = tfp.distributions
tfb = tfp.bijectors

"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""
import logging

logging.basicConfig(level=logging.INFO)

VALIDATE_ARGS = True
num_timesteps = 100
num_particles = 200
num_samples = 5000
particle_method = 'bsf'
state_dim = 1
observation_dim = 1

"""
Generate data 
"""
ssm_model = gen_ssm(num_timesteps=num_timesteps,
                    state_dim=state_dim, observed_dim=observation_dim)

true_state, observations = ssm_model.simulate()

# prior specification
PRIOR_MEAN_MU = tf.constant([1.0])
PRIOR_STD_MU = tf.constant([[0.8]])
PRIOR_INV_GAMMA_ALPHA = tf.constant([1.0])
PRIOR_INV_GAMMA_BETA = tf.constant([1.0])
N_CHAINS = 5

print(f'sample state mean: {tf.reduce_mean(true_state)}')
print(f'sample state std: {tf.math.reduce_std(true_state)}')
replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])


class StochasticVolativitySSM:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(StochasticVolativitySSM, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))

        # Currently, we could not trace the SMC results in the log_likelihood because of the wrapped transformed transition kernel.

    """Define the log_prior
    """

    def initial_theta(self):
        """
        Returns: initial precision matrix estimation
        """
        return [tf.random.uniform([N_CHAINS, 1]) + 0.5,  # sigma
                tf.random.normal([N_CHAINS, 1]),  # mu
                tf.random.uniform([N_CHAINS, 1]) - 0.5  # rho
                ]

    def log_theta_prior(self, sigma, mu, rho):
        return tfd.JointDistributionNamed(dict(
            sigma=tfd.InverseGamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                   scale=PRIOR_INV_GAMMA_BETA),
            mu=tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_MU, scale_tril=PRIOR_STD_MU),
            rho=tfd.Uniform(low=-0.99, high=0.99)))

    """Define update function for ssm model
    """

    def update_model(self, sigma, mu, rho, observations):

        sigma = sigma[..., tf.newaxis]

        prior_mean = mu
        prior_cov = sigma ** 2 / (1 - rho[..., tf.newaxis] ** 2)

        self._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=tf.sqrt(prior_cov))

        self._transition_noise_matrix = sigma
        self._transition_fn = lambda t, x: tf.cast(
            mu * (1 - rho) + rho * x, dtype=sigma.dtype)
        self._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=self._transition_fn(t, x),
            scale_tril=sigma)

    """Define data likelihood for ssm model
    """

    def log_target_dist(self, observations, num_particles):

        def _log_likelihood(sigma, mu, rho):
            # update model
            self.update_model(sigma, mu, rho, observations)

            # Conduct SMC
            def _run_smc():
                def _smc_trace_fn(state, kernel_results):
                    return (state.particles,
                            state.log_weights,
                            kernel_results.accumulated_log_marginal_likelihood)

                result = particle_filter(
                    observations=observations,
                    initial_state_prior=self.initial_state_prior,
                    transition_fn=self.transition_dist,
                    observation_fn=self.observation_dist,
                    num_particles=num_particles,
                    trace_fn=_smc_trace_fn,
                    trace_criterion_fn=lambda *_: True)
                return result

            traced_results = _run_smc()
            # TODO: could not trace the results because of out of scope
            # self.smc_trace_results.append(traced_results)
            return traced_results[-1][-1] \
                + self.log_theta_prior(sigma, mu, rho).log_prob({'sigma': sigma[:, 0],
                                                                 'mu': mu,
                                                                 'rho': rho[:, 0]})

        return _log_likelihood


sv_ssm = StochasticVolativitySSM(ssm_model)


# @tf.function
def run_mcmc():
    unconstrained_to_precision = tfb.JointMap(
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()]
    )

    result = particle_marginal_metropolis_hastings(sv_ssm,
                                                   replicate_observations,
                                                   num_samples,
                                                   num_particles,
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=sv_ssm.initial_theta(),
                                                   num_burnin_steps=int(num_samples // 2),
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
    mcmc_result = run_mcmc()
    smc_result = run_smc()

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
    dims={ "intercept": ["coefficient"]},
    )