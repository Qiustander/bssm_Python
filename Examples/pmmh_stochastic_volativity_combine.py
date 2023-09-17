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
num_timesteps = 400
num_particles = 1600
num_samples = 10000
particle_method = 'bsf'
state_dim = 1
observation_dim = 1
import time
"""
Generate data 
"""
ssm_model = gen_ssm(num_timesteps=num_timesteps,
                    state_dim=state_dim, observed_dim=observation_dim)

true_state, observations = ssm_model.simulate()

# prior specification
PRIOR_MEAN_MU = tf.constant([0.])
PRIOR_STD_MU = tf.constant([[2.0]])
PRIOR_INV_GAMMA_ALPHA = tf.constant([1.0])
PRIOR_INV_GAMMA_BETA = tf.constant([1.0])
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

        # Currently, we could not trace the SMC results in the log_likelihood because of the wrapped transformed transition kernel.


class StochasticVolativitySSM(NewClassInstance):
    """Define the log_prior
    """

    def initial_theta(self):
        """
        Returns: initial precision matrix estimation
        """
        # return tf.concat([tf.random.uniform([N_CHAINS, 1]) + 0.2,  # sigma
        #         tf.random.normal([N_CHAINS, 1]),  # mu
        #         tf.random.uniform([N_CHAINS, 1]) -0.2  # rho
        #         ],axis=-1)

        return tf.concat([tf.random.uniform([N_CHAINS, 1]) + 0.2,  # sigma
                          tf.random.normal([N_CHAINS, 1]),  # mu
                          tf.random.uniform([N_CHAINS, 1]) - 0.2  # rho
                          ], axis=-1)

    def log_theta_prior(self, sigma, mu, rho):
        return tfd.JointDistributionNamed(dict(
            sigma=tfd.Gamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                   rate=PRIOR_INV_GAMMA_BETA),
            mu=tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_MU, scale_tril=PRIOR_STD_MU),
            rho=tfd.Uniform(low=-1., high=1.)))

    """Define update function for ssm model
    """

    def update_model(self, sigma, mu, rho, observations):
        new_class = StochasticVolativitySSM(self)

        sigma = sigma[..., tf.newaxis]

        prior_mean = mu
        prior_cov = sigma / tf.sqrt(1 - rho[..., tf.newaxis] ** 2)

        new_class._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=prior_cov)

        transition_fn = lambda t, x: tf.cast(
            mu * (1 - rho) + rho * x, dtype=sigma.dtype)
        new_class._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=transition_fn(t, x),
            scale_tril=sigma)

        return new_class

    """Define data likelihood for ssm model
    """


def _smc_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.accumulated_log_marginal_likelihood)

def log_target_dist(ssm_models, observations, num_particles):
    def _log_likelihood(parameters):
        # update model
        sigma = parameters[..., 0:1]
        mu = parameters[..., 1:2]
        rho = parameters[..., 2:3]

        new_class = ssm_models.update_model(sigma=sigma, mu=mu, rho=rho, observations=observations)

        # Conduct SMC
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

        # start = time.time()
        traced_results = _run_smc(initial_state_prior=new_class.initial_state_prior,
                                  transition_dist=new_class.transition_dist,
                                  observation_dist=new_class.observation_dist)
        # end = time.time()
        # logging.info(f"{end - start}")
        # TODO: could not trace the results because of out of scope
        # self.smc_trace_results.append(traced_results)
        return traced_results[-1][-1] \
            + ssm_models.log_theta_prior(sigma, mu, rho).log_prob({'sigma': sigma[..., 0],
                                                             'mu': mu,
                                                             'rho': rho[..., 0]})

    return _log_likelihood


sv_ssm = StochasticVolativitySSM(ssm_model)


@tf.function(jit_compile=True)
def run_mcmc():
    unconstrained_to_precision = tfb.Blockwise(
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()],
        block_sizes=[1, 1, 1]
    )

    result = particle_marginal_metropolis_hastings(num_samples,
                                                   target_dist=log_target_dist(sv_ssm, replicate_observations, num_particles),
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=sv_ssm.initial_theta(),
                                                   num_burnin_steps=int(num_samples),
                                                   num_steps_between_results=2,
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
    print(f" execute time {end-start}")
    mcmc_trace = mcmc_result.trace_results
    mcmc_state = mcmc_result.states

    mcmc_states = []
    mcmc_states.append(mcmc_state[..., 0:1])
    mcmc_states.append(mcmc_state[..., 1:2])
    mcmc_states.append(mcmc_state[..., 2:3])

    parameter_names = ['noise_std', 'mean', 'coefficients']
    posterior_trace = az.from_dict(
        posterior={
            k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_states)
        },
        # sample_stats={k: np.swapaxes(v, 0, 1) for k, v in mcmc_result.items()},
        observed_data={"observations": observations},
        coords={"coefficient": np.arange(1)},
        dims={"intercept": ["coefficient"]},
    )
    az.summary(posterior_trace)
    az.plot_trace(posterior_trace)
    az.plot_rank(posterior_trace)

xxx = 1