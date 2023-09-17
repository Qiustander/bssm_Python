import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from Models.check_argument import *
from Models.ssm_nlg import NonlinearSSM
from Inference.MCMC.kernel.delayed_acceptance import DelayedAcceptance
from Inference.MCMC.kernel.random_walk_metropolis import UncalibratedRandomWalk
from tensorflow_probability.python.mcmc import TransformedTransitionKernel
from tensorflow_probability.python.mcmc import effective_sample_size, potential_scale_reduction
from tensorflow_probability.python.mcmc import sample_chain
from tensorflow_probability.python.mcmc import RandomWalkMetropolis as RandomWalkMetropolisOriginal
import pytest
import matplotlib.pyplot as plt
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions
tfb = tfp.bijectors

seed = 123
# Number of samples.
num_mcmc_samples = 2000
num_burn_in_steps = int(num_mcmc_samples // 2)
# The number of chains is determined by the shape of the initial values.
# Here we'll generate N_CHAINS chains, so we'll need a tensor of N_CHAINS initial values.
N_CHAINS = 8

"""Toy AR1"""
n = 100
# True parameters
def_varphi = 0.6
true_varphi = tf.constant([def_varphi])
def_delta = -0.6
true_ar_mean = tf.constant([[def_delta]])
def_noise_std = 1.2
true_noise_std = tf.constant([[def_noise_std]])

observations = []
observations.append(tfd.MultivariateNormalTriL(loc=true_ar_mean / (1 - true_varphi),
                                               scale_tril=true_noise_std / tf.sqrt(1 - true_varphi ** 2)).sample(
    seed=seed)
)
for i in range(8 * n):
    observations.append(true_ar_mean + observations[-1] * true_varphi +
                        tfd.MultivariateNormalTriL(loc=tf.constant([0.]),
                                                   scale_tril=true_noise_std).sample(seed=seed))

observations = tf.concat(observations, axis=0)[-n:]

PRIOR_MEAN_DELTA = tf.constant([1.0])
PRIOR_STD_DELTA = tf.constant([[0.8]])
PRIOR_INV_GAMMA_ALPHA = tf.constant([1.0])
PRIOR_INV_GAMMA_BETA = tf.constant([1.0])

"""
"""


class TestDelayedAcceptance:

    @pytest.mark.parametrize(("seed", "observations"),
                             [(seed, observations)])
    def test_delayed_acceptance_univariate(self, seed, observations):

        replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

        def prior_dist(sigma, delta, varphi):
            return tf.squeeze(tfd.InverseGamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                               scale=PRIOR_INV_GAMMA_BETA).log_prob(sigma[..., 0])) + \
                tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_DELTA, scale_tril=PRIOR_STD_DELTA).log_prob(delta) + \
                tf.squeeze(tfd.Uniform(low=-0.99, high=0.99).log_prob(varphi))

        def posterior_dist(observations):
            def _compute_posterior(sigma, delta, varphi):
                first_mean = delta / (1 - varphi)
                first_std = sigma / tf.sqrt(1 - varphi[..., tf.newaxis] ** 2)
                first_likelihood = tfd.MultivariateNormalTriL(loc=first_mean,
                                                              scale_tril=first_std).log_prob(observations[0, ...])
                # x_t = delta + phi x_t-1 + e_t
                data_mean = varphi * observations[:-1, ...] + delta
                data_std = sigma
                rest_likelihood = tfd.MultivariateNormalTriL(loc=data_mean,
                                                             scale_tril=data_std).log_prob(observations[1:, ...])
                return first_likelihood + tf.reduce_sum(rest_likelihood, axis=0) + prior_dist(sigma, delta, varphi)

            return _compute_posterior

        """
        Sampling
        """
        init_state = [tf.random.uniform([N_CHAINS, 1, 1]) * 2,
                      tf.random.normal([N_CHAINS, 1]),
                      tf.random.uniform([N_CHAINS, 1]) - 0.5]

        unconstrained_to_original = tfb.JointMap(
            bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()]
        )

        @tf.function
        def run_delayed_metropolis_hasting():
            mh_kernel = DelayedAcceptance(exact_target_prob=posterior_dist(replicate_observations),
                                          approx_target_prob=posterior_dist(replicate_observations),
                                          inner_kernel=UncalibratedRandomWalk(target_log_prob_fn=posterior_dist(replicate_observations),
                                                                              random_walk_cov=0.2))
            states, kernels_results = sample_chain(num_results=num_mcmc_samples,
                                                   current_state=init_state,  # constant start
                                                   num_burnin_steps=num_burn_in_steps,
                                                   num_steps_between_results=0,
                                                   kernel=TransformedTransitionKernel(mh_kernel,
                                                                                      bijector=unconstrained_to_original),
                                                   seed=seed)
            return states, kernels_results

        @tf.function
        def run_original_metropolis_hasting():
            mh_kernel = RandomWalkMetropolisOriginal(posterior_dist(replicate_observations),
                                                     new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.2))
            states, kernels_results = sample_chain(num_results=num_mcmc_samples,
                                                   current_state=init_state,  # constant start
                                                   num_burnin_steps=num_burn_in_steps,
                                                   num_steps_between_results=0,
                                                   kernel=TransformedTransitionKernel(mh_kernel,
                                                                                      bijector=unconstrained_to_original),
                                                   seed=seed)
            return states, kernels_results

        # mh_original_states, mh_original_results = run_original_metropolis_hasting()
        mh_revised_states, mh_revised_results = run_delayed_metropolis_hasting()
        r_hat_mh = tfp.mcmc.potential_scale_reduction(
            mh_original_states,
            independent_chain_ndims=1,
            split_chains=True)
        r_hat_mh_revised = tfp.mcmc.potential_scale_reduction(
            mh_revised_states,
            independent_chain_ndims=1,
            split_chains=True)
        [
            tf.debugging.assert_near(r_hat_mh_part, r_hat_mh_revised_part, atol=1e-6)
            for r_hat_mh_part, r_hat_mh_revised_part
            in zip(r_hat_mh, r_hat_mh_revised)
        ]

        [
            tf.debugging.assert_near(mh_original_state, mh_revised_state, atol=1e-6)
            for mh_original_state, mh_revised_state
            in zip(mh_original_states, mh_revised_states)
        ]

    @pytest.mark.parametrize(("seed", "observations"),
                             [(seed, observations)])
    def test_delayed_acceptance_multivariate(self, seed, observations):
        # Revision: 1. random walk function -> multivariate normal to multiple with covariance matrix
        #   2. also trace the covariance matrix

        replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

        def prior_dist(parameter):
            return tfd.InverseGamma(concentration=PRIOR_INV_GAMMA_ALPHA,
                                    scale=PRIOR_INV_GAMMA_BETA).log_prob(parameter[..., 0]) + \
                tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_DELTA, scale_tril=PRIOR_STD_DELTA).log_prob(
                    parameter[..., 1:2]) + \
                tf.squeeze(tfd.Uniform(low=-1.1, high=1.1).log_prob(parameter[..., -1:]))

        def posterior_dist(observations):
            def _compute_posterior(parameter):
                # assume the first observations is the stationary state
                # print(f"current parameter {parameter}")

                first_mean = parameter[..., 1:2] / (1 - parameter[..., -1:])
                first_std = parameter[..., 0:1] / tf.sqrt(1 - parameter[..., -1:] ** 2)
                first_likelihood = tfd.MultivariateNormalTriL(loc=first_mean,
                                                              scale_tril=first_std[..., tf.newaxis]).log_prob(
                    observations[0, ...])
                # x_t = delta + phi x_t-1 + e_t
                data_mean = parameter[..., -1:] * observations[:-1, ...] + parameter[..., 1:2]
                data_std = parameter[..., 0:1]
                rest_likelihood = tfd.MultivariateNormalTriL(loc=data_mean,
                                                             scale_tril=data_std[..., tf.newaxis]).log_prob(
                    observations[1:, ...])

                return first_likelihood + tf.reduce_sum(rest_likelihood, axis=0) + prior_dist(parameter)

            return _compute_posterior

        """
        Sampling
        """
        init_state = tf.concat([tf.random.uniform([N_CHAINS, 1]),
                                tf.random.normal([N_CHAINS, 1]),
                                tf.random.uniform([N_CHAINS, 1]) / 2], axis=-1)

        unconstrained_to_original = tfb.Blockwise(
            bijectors=[tfb.Exp(), tfb.Identity(), tfb.Tanh()],
            block_sizes=[1, 1, 1]
        )

        @tf.function
        def run_revised_metropolis_hasting():
            mh_kernel = RandomWalkMetropolis(posterior_dist(replicate_observations),
                                             random_walk_cov=0.2)
            states, kernels_results = sample_chain(num_results=num_mcmc_samples,
                                                   current_state=init_state,  # constant start
                                                   num_burnin_steps=num_burn_in_steps,
                                                   num_steps_between_results=0,
                                                   kernel=TransformedTransitionKernel(mh_kernel,
                                                                                      bijector=unconstrained_to_original),
                                                   seed=seed)
            return states, kernels_results

        @tf.function
        def run_original_metropolis_hasting():
            mh_kernel = RandomWalkMetropolisOriginal(posterior_dist(replicate_observations),
                                                     new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.2))
            states, kernels_results = sample_chain(num_results=num_mcmc_samples,
                                                   current_state=init_state,  # constant start
                                                   num_burnin_steps=num_burn_in_steps,
                                                   num_steps_between_results=0,
                                                   kernel=TransformedTransitionKernel(mh_kernel,
                                                                                      bijector=unconstrained_to_original),
                                                   seed=seed)
            return states, kernels_results

        mh_original_states, mh_original_results = run_original_metropolis_hasting()
        mh_revised_states, mh_revised_results = run_revised_metropolis_hasting()
        r_hat_mh = tfp.mcmc.potential_scale_reduction(
            mh_original_states,
            independent_chain_ndims=1,
            split_chains=True)
        r_hat_mh_revised = tfp.mcmc.potential_scale_reduction(
            mh_revised_states,
            independent_chain_ndims=1,
            split_chains=True)

        tf.debugging.assert_near(r_hat_mh, r_hat_mh_revised, atol=1e-1)
        tf.debugging.assert_near(mh_original_states.numpy().mean(axis=(0, 1)),
                                 mh_revised_states.numpy().mean(axis=(0, 1)), atol=1e-1)

