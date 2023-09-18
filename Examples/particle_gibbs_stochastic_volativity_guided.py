from Synthetic_Data.stochastic_volatility import gen_ssm
import matplotlib.pyplot as plt
from Inference.MCMC.particle_gibbs import particle_gibbs_sampling
from Inference.SMC.infer_trajectories import infer_trajectories
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter
from Inference.MCMC.kernel.gibbs_kernel import GibbsKernel
from Inference.MCMC.kernel.sampling_kernel import SamplingKernel, cond_sample_fn
from Inference.SMC.forward_filter_backward_sampling import forward_filter_backward_sampling
from Inference.MCMC.run_diagostic import run_pmcmc_diagnostic

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import arviz as az

# tf.config.optimizer.set_jit(True)
tfd = tfp.distributions
tfb = tfp.bijectors

"""
1D SV model: parameters has sigma, mu, rho 
"""
import logging

logging.basicConfig(level=logging.INFO)

VALIDATE_ARGS = True
num_timesteps = 100
num_particles = 200
num_samples = 2000
state_dim = 1
observation_dim = 1
import time

"""
Generate data 
"""
ssm_model = gen_ssm(num_timesteps=num_timesteps,
                    state_dim=state_dim, observed_dim=observation_dim)

true_state, observations = ssm_model.simulate(len_time_step=4*num_timesteps)
true_state = true_state[-num_timesteps:]
observations = observations[-num_timesteps:]


# prior specification
PRIOR_MEAN_MU = tf.constant([0.])
PRIOR_STD_MU = tf.constant([[2.0]])
PRIOR_INV_GAMMA_ALPHA = tf.constant([3.])
PRIOR_INV_GAMMA_BETA = tf.constant([1.])
N_CHAINS = 10
replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

# @title Analytical Posterior Mean
true_sigma = tf.constant([0.178])
true_mu = tf.constant([-1.02])
true_rho = tf.constant([0.97])
true_mean = true_mu
true_std = true_sigma / tf.sqrt(1 - true_rho ** 2)
sample_std = tf.math.reduce_std(true_state)
sample_mean = tf.reduce_mean(true_state)
print(f'true state mean: {true_mean}')
print(f'true state std: {true_std}')
print(f'sample state mean: {tf.reduce_mean(true_state)}')
print(f'sample state std: {tf.math.reduce_std(true_state)}')

# coeff_mu = 1/(1/PRIOR_STD_MU**2 + (observations.shape[0]+1)*(1-true_rho)**2/true_sigma**2)
coeff_mu = true_sigma ** 2 / ((1 - true_rho ** 2) + (observations.shape[0] + 1) * (1 - true_rho) ** 2)
sum_xt = tf.reduce_sum((true_state[:-1] - true_mu) * (true_state[1:] - true_mu), axis=0)
sum_xt2 = tf.reduce_sum((true_state[:-1] - true_mu) ** 2, axis=0)
reduce_sum = tf.reduce_sum(true_state[1:] - true_rho * true_state[:-1], axis=0)
true_posterior_mean_mu = coeff_mu * (
            reduce_sum * (1 - true_rho) / true_sigma ** 2 + (1 - true_rho ** 2) / true_sigma ** 2 * true_state[0])
print(f"posterior mu: {true_posterior_mean_mu}")
true_posterior_mean_rho = sum_xt / sum_xt2
print(f"posterior rho: {true_posterior_mean_rho}")
posterior_alpha = PRIOR_INV_GAMMA_ALPHA + observations.shape[0] / 2
posterior_beta = PRIOR_INV_GAMMA_BETA + \
                 tf.reduce_sum((true_state[1:] - true_mu - true_rho * (true_state[:-1] - true_mu)) ** 2, axis=0) / 2
poster_dist = tfd.InverseGamma(concentration=posterior_alpha,
                               scale=posterior_beta)
true_posterior_sigma_mean = tf.sqrt(poster_dist.mean())
print(f"posterior noise std: {true_posterior_sigma_mean}")


class NewClassInstance:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(NewClassInstance, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))


class StochasticVolativitySSM(NewClassInstance):
    """Define the log_prior
    """

    """Define update function for ssm model
    """

    @tf.function
    def simulate(self, seed=None):
        """
        Simulate true state and observations for filtering and smoothing, and parameter estimation.
        Args:
            seed: generated seed
        Returns:
            observations
        """
        len_time_step = self._num_timesteps

        def _generate_signal(transition_fn, observation_fn):
            def _inner_wrap(gen_data, current_step):
                last_state, _ = gen_data

                current_state = transition_fn(current_step, last_state).sample(seed=seed)
                current_observation = observation_fn(current_step, current_state).sample(seed=seed)

                return current_state, current_observation

            return _inner_wrap

        gen_data = _generate_signal(self._transition_dist, self._observation_dist)

        initial_state = self._initial_state_prior.sample()
        init_obs = self._observation_dist(0, initial_state).sample()
        overall_step = tf.range(1, len_time_step)

        true_state, observations = tf.scan(gen_data,
                                           elems=overall_step,
                                           initializer=(initial_state,
                                                        init_obs),
                                           )
        true_state = tf.concat([initial_state[tf.newaxis], true_state], axis=0)
        observations = tf.concat([init_obs[tf.newaxis], observations], axis=0)

        return true_state, observations

    def auxiliary_fn(self):
        pass

    def update_model(self, sigma, mu, rho):
        new_class = StochasticVolativitySSM(self)

        sigma = sigma[..., tf.newaxis]

        prior_mean = mu
        prior_std = sigma / tf.sqrt(1 - rho[..., tf.newaxis] ** 2)

        new_class._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=prior_std)

        transition_fn = lambda t, x: tf.cast(
            mu * (1. - rho) + rho * x, dtype=sigma.dtype)

        def _xhat(xst, sig, yt):
            return xst + 0.5 * sig ** 2 * (yt ** 2 * tf.exp(-xst) - 1.0)

        new_class.proposal_dist = lambda t, x, y: tfd.MultivariateNormalTriL(
            loc=_xhat(transition_fn(t, x), sigma[..., 0], y),
            scale_tril=sigma)

        new_class._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=transition_fn(t, x),
            scale_tril=sigma)

        return new_class

sv_ssm = StochasticVolativitySSM(ssm_model)


def _run_smc(ssmmodel, conditional_trajectory, is_conditional):
    result = forward_filter_backward_sampling(ssmmodel,
                                              replicate_observations,
                                              particle_filter_name='apf',
                                              resample_ess=0.5,
                                              is_guided=True,
                                              is_conditional=is_conditional,
                                              conditional_sample=conditional_trajectory,
                                              resample_fn='multinomial',
                                              num_particles=num_particles, is_one_trajectory=True)
    # tf.print(tf.reduce_sum(result.incremental_log_marginal_likelihoods, axis=0))
    return result.trajectories


def log_theta_prior():
    return tfd.JointDistributionNamed(dict(
        sigma=tfd.Gamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                        rate=PRIOR_INV_GAMMA_BETA),
        mu=tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_MU, scale_tril=PRIOR_STD_MU),
        rho=tfd.Uniform(low=-1., high=1.))
    )

# from Inference.MCMC.prior_predictive_check import prior_predictive_check
# prior_predictive_check(ssm_model=sv_ssm, observations=true_state, sample_sizes=300,
#                        prior_dist=log_theta_prior)


def initial_theta():
    """
    Returns: initial precision matrix estimation
    """

    def _get_initial_trajectory():
        prior_samples = log_theta_prior().sample(N_CHAINS)
        new_svm_model = sv_ssm.update_model(sigma=prior_samples['sigma'],
                                            mu=prior_samples['mu'],
                                            rho=prior_samples['rho'][..., tf.newaxis])

        return _run_smc(new_svm_model, is_conditional=False, conditional_trajectory=None)

    return [[tf.random.uniform([N_CHAINS, 1]) +0.1,  # sigma
             tf.random.normal([N_CHAINS, 1]),  # mu
             tf.random.uniform([N_CHAINS, 1]) - 0.2  # rho
             ],
            _get_initial_trajectory()
            ]


def parameters_sample_fn(sampling_idx, current_state, rest_state_parts, seed=None):
    """ Update parameters given the reference trajectory
    """

    # current parameters
    parameters = current_state
    sigma = parameters[0]
    mu = parameters[1]
    rho = parameters[2]

    # time_step, chain, state_dim
    reference_states = rest_state_parts[0]
    reference_states_shift = tf.concat([mu[tf.newaxis], reference_states[:-1, ...]], axis=0)

    # Update mu - Normal
    coeff_mu_inner = 1 / (1 / PRIOR_STD_MU ** 2 + replicate_observations.shape[0] * (1 - rho) ** 2 / sigma ** 2)

    reduce_sum_mu = tf.reduce_sum(reference_states[1:] - rho * reference_states[:-1], axis=0)
    posterior_mean_mu = coeff_mu_inner * (reduce_sum_mu * (1. - rho) / sigma ** 2 + PRIOR_MEAN_MU / PRIOR_STD_MU ** 2)
    posterior_std_mu = tf.sqrt(coeff_mu_inner)
    posterior_mu = tfd.MultivariateNormalTriL(loc=posterior_mean_mu,
                                              scale_tril=posterior_std_mu[..., tf.newaxis]).sample(seed=seed)

    # Update rho - truncated normal

    sum_xt2 = tf.reduce_sum((reference_states[:-1] - posterior_mu) ** 2, axis=0)
    sum_xt = tf.reduce_sum((reference_states[1:] - posterior_mu) * (reference_states[:-1] - posterior_mu), axis=0)
    posterior_mean_rho = sum_xt / sum_xt2
    # posterior_mean_rho = posterior_mean_rho*tf.greater_equal(posterior_mean_rho, -1.0) * tf.less_equal(posterior_mean_rho, 1.0)
    posterior_std_rho = sigma / tf.sqrt(sum_xt2)
    posterior_rho = tfd.TruncatedNormal(loc=posterior_mean_rho,
                                        scale=posterior_std_rho,
                                        low=-0.99,
                                        high=1.0).sample(seed=seed)

    # Update sigma - sqrt of Inverse Gamma (std)
    posterior_alpha = PRIOR_INV_GAMMA_ALPHA + (replicate_observations.shape[0] - 1) / 2
    posterior_beta = PRIOR_INV_GAMMA_BETA + \
                     tf.reduce_sum((reference_states[1:] - posterior_mu - posterior_rho * (
                                 reference_states[:-1] - posterior_mu)) ** 2, axis=0) / 2
    gamma_dist = tfd.InverseGamma(concentration=posterior_alpha,
                                  scale=posterior_beta)
    posterior_sigma = tf.sqrt(gamma_dist.sample(seed=seed))

    return [posterior_sigma, posterior_mu, posterior_rho]


def states_sample_fn(sampling_idx, current_state, rest_state_parts, seed=None):
    """ Update the reference trajectory given the parameters
    """
    current_reference_trajectory = current_state

    parameters = rest_state_parts[0]
    sigma = parameters[0]
    mu = parameters[1]
    rho = parameters[2]

    new_svm_model = sv_ssm.update_model(sigma=sigma, mu=mu, rho=rho)

    traced_results = _run_smc(new_svm_model, current_reference_trajectory, is_conditional=True)
    # traced_results = tf.tile(tf.expand_dims(true_state, axis=1), multiples=[1, N_CHAINS, 1])
    # traced_results = traced_results + 0.1*tf.random.uniform(traced_results.shape)

    return traced_results


def kernel_make_fn_parameters(target_log_prob_fn, states, state_part_idx):
    return SamplingKernel(target_log_prob_fn=target_log_prob_fn,
                          full_cond_dist=cond_sample_fn(parameters_sample_fn),
                          sampling_idx=state_part_idx,
                          current_full_states=states)


def kernel_make_fn_states(target_log_prob_fn, states, state_part_idx):
    return SamplingKernel(target_log_prob_fn=target_log_prob_fn,
                          full_cond_dist=cond_sample_fn(states_sample_fn),
                          sampling_idx=state_part_idx,
                          current_full_states=states)


def log_target_dist(observations):
    def _log_likelihood(parameters, states):
        # currently dont need to calculate the likelihood
        return tf.constant([0.])

    return _log_likelihood


# @tf.function(jit_compile=True)
def run_mcmc():
    unconstrained_to_precision = tfb.Blockwise(
        bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()]
    )

    kernel_list = [(0, kernel_make_fn_parameters),
                   (1, kernel_make_fn_states)]
    particle_gibbs_kernel = GibbsKernel(
        target_log_prob_fn=log_target_dist(replicate_observations),
        kernel_list=kernel_list
    )

    result = particle_gibbs_sampling(num_results=num_samples,
                                     gibbs_kernel=particle_gibbs_kernel,
                                     init_state=initial_theta(),
                                     num_burnin_steps=int(num_samples),
                                     num_steps_between_results=0,
                                     seed=None,
                                     name=None)
    return result


if __name__ == '__main__':
    start = time.time()
    mcmc_result = run_mcmc()
    # smc_result = run_smc()
    end = time.time()
    print(f" execute time {end - start}")
    mcmc_trace = mcmc_result.trace_results
    mcmc_state = mcmc_result.states[0]

    parameter_names = ['noise_std', 'mean', 'coefficients']
    run_pmcmc_diagnostic(mcmc_states=mcmc_state, mcmc_traces=mcmc_trace, observations=true_state,
                         ssm_model=sv_ssm, sample_sizes=100, parameter_names=parameter_names)
    # posterior_trace = az.from_dict(
    #     posterior={
    #         k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_state)
    #     },
    #     # sample_stats={k: np.swapaxes(v, 0, 1) for k, v in mcmc_result.items()},
    #     observed_data={"observations": observations},
    #     coords={"coefficient": np.arange(1)},
    #     dims={"intercept": ["coefficient"]},
    # )
    # az.summary(posterior_trace)
    # az.plot_trace(posterior_trace)
    # az.plot_rank(posterior_trace)
